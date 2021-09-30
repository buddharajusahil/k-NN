/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */
/*
 *   Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License").
 *   You may not use this file except in compliance with the License.
 *   A copy of the License is located at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   or in the "license" file accompanying this file. This file is distributed
 *   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *   express or implied. See the License for the specific language governing
 *   permissions and limitations under the License.
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import org.opensearch.common.xcontent.DeprecationHandler;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.knn.index.JNIService;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.KNNCodecUtil;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelCache;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.DocValuesConsumer;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.opensearch.knn.index.KNNVectorFieldMapper;
import org.opensearch.knn.common.KNNConstants;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Paths;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.index.codec.KNNCodecUtil.buildEngineFileName;

/**
 * This class writes the KNN docvalues to the segments
 */
class KNN80DocValuesConsumer extends DocValuesConsumer implements Closeable {

    private final Logger logger = LogManager.getLogger(KNN80DocValuesConsumer.class);

    private final String TEMP_SUFFIX = "tmp";
    private DocValuesConsumer delegatee;
    private SegmentWriteState state;

    KNN80DocValuesConsumer(DocValuesConsumer delegatee, SegmentWriteState state) throws IOException {
        this.delegatee = delegatee;
        this.state = state;
    }

    @Override
    public void addBinaryField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        delegatee.addBinaryField(field, valuesProducer);
        addKNNBinaryField(field, valuesProducer);
    }

    public void addKNNBinaryField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        KNNCounter.GRAPH_INDEX_REQUESTS.increment();
        if (field.attributes().containsKey(KNNVectorFieldMapper.KNN_FIELD)) {
            // Get engine to be used for indexing
            String engineName = field.attributes().getOrDefault(KNNConstants.KNN_ENGINE, KNNEngine.DEFAULT.getName());
            KNNEngine knnEngine = KNNEngine.getEngine(engineName);

            // Get values to be indexed
            BinaryDocValues values = valuesProducer.getBinary(field);
            KNNCodecUtil.Pair pair = KNNCodecUtil.getFloats(values);
            if (pair.vectors.length == 0 || pair.docs.length == 0) {
                logger.info("Skipping engine index creation as there are no vectors or docs in the documents");
                return;
            }

            // Create library index either from model or from scratch
            String engineFileName = buildEngineFileName(state.segmentInfo.name, knnEngine.getLatestBuildVersion(),
                    field.name, knnEngine.getExtension());
            String indexPath = Paths.get(((FSDirectory) (FilterDirectory.unwrap(state.directory))).getDirectory().toString(),
                    engineFileName).toString();
            String tmpEngineFileName = engineFileName + TEMP_SUFFIX;
            String tempIndexPath = indexPath + TEMP_SUFFIX;
            if (field.attributes().containsKey(MODEL_ID)) {
                String modelId = field.attributes().get(MODEL_ID);
                Model model = ModelCache.getInstance().get(modelId);

                if (model.getModelBlob() == null) {
                    throw new RuntimeException("There is no model with id \"" + modelId + "\"");
                }

                if (model.getModelMetadata().getKnnEngine() != knnEngine) {
                    throw new RuntimeException("Model Engine \"" + model.getModelMetadata().getKnnEngine().getName()
                            + "\" cannot be different than index engine \"" + knnEngine.getName() + "\"");
                }

                String spaceName = field.getAttribute(KNNConstants.SPACE_TYPE);
                if (spaceName == null) {
                    throw new RuntimeException("Space Type cannot be null");
                }

                SpaceType spaceType = SpaceType.getSpace(spaceName);
                if (model.getModelMetadata().getSpaceType() != spaceType) {
                    throw new RuntimeException("Model Space Type \"" + model.getModelMetadata().getSpaceType().getValue()
                            + "\" cannot be different than index Space Type \"" + spaceType.getValue() + "\"");
                }

                int dimension = Integer.parseInt(field.attributes().getOrDefault(DIMENSION, "-1"));
                if (model.getModelMetadata().getDimension() != dimension) {
                    throw new RuntimeException("Model dimension \"" + model.getModelMetadata().getDimension()
                            + "\" cannot be different than index dimension \"" + dimension + "\"");
                }

                createKNNIndexFromTemplate(model.getModelBlob(), pair, knnEngine, tempIndexPath);
            } else {
                createKNNIndexFromScratch(field, pair, knnEngine, tempIndexPath);
            }

            /*
             * Adds Footer to the serialized graph
             * 1. Copies the serialized graph to new file.
             * 2. Adds Footer to the new file.
             *
             * We had to create new file here because adding footer directly to the
             * existing file will miss calculating checksum for the serialized graph
             * bytes and result in index corruption issues.
             */
            //TODO: I think this can be refactored to avoid this copy and then write
            // https://github.com/opendistro-for-elasticsearch/k-NN/issues/330
            try (IndexInput is = state.directory.openInput(tmpEngineFileName, state.context);
                 IndexOutput os = state.directory.createOutput(engineFileName, state.context)) {
                os.copyBytes(is, is.length());
                CodecUtil.writeFooter(os);
            } catch (Exception ex) {
                KNNCounter.GRAPH_INDEX_ERRORS.increment();
                throw new RuntimeException("[KNN] Adding footer to serialized graph failed: " + ex);
            } finally {
                IOUtils.deleteFilesIgnoringExceptions(state.directory, tmpEngineFileName);
            }
        }
    }

    private void createKNNIndexFromTemplate(byte[] model, KNNCodecUtil.Pair pair, KNNEngine knnEngine,
                                            String indexPath) {
        AccessController.doPrivileged(
                (PrivilegedAction<Void>) () -> {
                    JNIService.createIndexFromTemplate(pair.docs, pair.vectors, indexPath, model, knnEngine.getName());
                    return null;
                }
        );
    }

    private void createKNNIndexFromScratch(FieldInfo fieldInfo, KNNCodecUtil.Pair pair, KNNEngine knnEngine,
                                           String indexPath) throws IOException {
        Map<String, Object> parameters = new HashMap<>();
        Map<String, String> fieldAttributes = fieldInfo.attributes();
        String parametersString = fieldAttributes.get(KNNConstants.PARAMETERS);

        // parametersString will be null when legacy mapper is used
        if (parametersString == null) {
            parameters.put(KNNConstants.SPACE_TYPE, fieldAttributes.getOrDefault(KNNConstants.SPACE_TYPE,
                    SpaceType.DEFAULT.getValue()));

            String efConstruction = fieldAttributes.get(KNNConstants.HNSW_ALGO_EF_CONSTRUCTION);
            if (efConstruction != null) {
                parameters.put(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, Integer.parseInt(efConstruction));
            }

            String m = fieldAttributes.get(KNNConstants.HNSW_ALGO_M);
            if (m != null) {
                parameters.put(KNNConstants.METHOD_PARAMETER_M, Integer.parseInt(m));
            }

        } else {
            parameters.putAll(
                    XContentFactory.xContent(XContentType.JSON).createParser(NamedXContentRegistry.EMPTY,
                            DeprecationHandler.THROW_UNSUPPORTED_OPERATION, parametersString).map()
            );
        }

        // Pass the path for the nms library to save the file
        AccessController.doPrivileged(
                (PrivilegedAction<Void>) () -> {
                    JNIService.createIndex(pair.docs, pair.vectors, indexPath, parameters, knnEngine.getName());
                    return null;
                }
        );
    }

    /**
     * Merges in the fields from the readers in mergeState
     *
     * @param mergeState Holds common state used during segment merging
     */
    @Override
    public void merge(MergeState mergeState) {
        try {
            delegatee.merge(mergeState);
            assert mergeState != null;
            assert mergeState.mergeFieldInfos != null;
            for (FieldInfo fieldInfo : mergeState.mergeFieldInfos) {
                DocValuesType type = fieldInfo.getDocValuesType();
                if (type == DocValuesType.BINARY) {
                    addKNNBinaryField(fieldInfo, new KNN80DocValuesReader(mergeState));
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void addSortedSetField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        delegatee.addSortedSetField(field, valuesProducer);
    }

    @Override
    public void addSortedNumericField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        delegatee.addSortedNumericField(field, valuesProducer);
    }

    @Override
    public void addSortedField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        delegatee.addSortedField(field, valuesProducer);
    }

    @Override
    public void addNumericField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        delegatee.addNumericField(field, valuesProducer);
    }

    @Override
    public void close() throws IOException {
        delegatee.close();
    }
}