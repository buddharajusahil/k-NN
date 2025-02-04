/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import com.google.common.base.Predicates;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NonNull;
import lombok.Value;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.HitQueue;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TaskExecutor;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.common.util.concurrent.OpenSearchExecutors;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.query.iterators.BinaryVectorIdsKNNIterator;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.iterators.ByteVectorIdsKNNIterator;
import org.opensearch.knn.index.query.iterators.NestedBinaryVectorIdsKNNIterator;
import org.opensearch.knn.index.query.iterators.VectorIdsKNNIterator;
import org.opensearch.knn.index.query.iterators.KNNIterator;
import org.opensearch.knn.index.query.iterators.NestedByteVectorIdsKNNIterator;
import org.opensearch.knn.index.query.iterators.NestedVectorIdsKNNIterator;
import org.opensearch.knn.index.vectorvalues.KNNBinaryVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNByteVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.threadpool.ThreadPool;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.function.Predicate;
import java.util.logging.Logger;

import static org.opensearch.knn.common.KNNConstants.EXACT_SEARCH_THREAD_POOL;
import static org.opensearch.knn.common.KNNConstants.TRAIN_THREAD_POOL;

@Log4j2
// @AllArgsConstructor
public class ExactSearcher {

    private final ModelDao modelDao;
    private final TaskExecutor taskExecutor;
    int numThreads;
    private static ThreadPool threadPool;

    public ExactSearcher(ModelDao modelDao, ThreadPool threadPool) {
        this.modelDao = modelDao;
        this.threadPool = threadPool;
        numThreads = threadPool.info(EXACT_SEARCH_THREAD_POOL).getMax();
        this.taskExecutor = new TaskExecutor(Executors.newFixedThreadPool(numThreads));
    }

    /**
     * Execute an exact search on a subset of documents of a leaf
     *
     * @param leafReaderContext {@link LeafReaderContext}
     * @param exactSearcherContext {@link ExactSearcherContext}
     * @return Map of re-scored results
     * @throws IOException exception during execution of exact search
     */
    public Map<Integer, Float> searchLeaf(final LeafReaderContext leafReaderContext, final ExactSearcherContext exactSearcherContext)
        throws IOException {
        final KNNIterator iterator = getKNNIterator(leafReaderContext, exactSearcherContext);
        // because of any reason if we are not able to get KNNIterator, return an empty map
        if (iterator == null) {
            return Collections.emptyMap();
        }
        if (exactSearcherContext.getKnnQuery().getRadius() != null) {
            return doRadialSearch(leafReaderContext, exactSearcherContext, iterator);
        }
        if (exactSearcherContext.getMatchedDocsIterator() != null
            && exactSearcherContext.numberOfMatchedDocs <= exactSearcherContext.getK()) {
            return scoreAllDocs(iterator, leafReaderContext, exactSearcherContext);
        }

        return searchTopCandidates(iterator, exactSearcherContext.getK(), Predicates.alwaysTrue(), leafReaderContext, exactSearcherContext);
    }

    /**
     * Perform radial search by comparing scores with min score. Currently, FAISS from native engine supports radial search.
     * Hence, we assume that Radius from knnQuery is always distance, and we convert it to score since we do exact search uses scores
     * to filter out the documents that does not have given min score.
     * @param leafReaderContext {@link LeafReaderContext}
     * @param exactSearcherContext {@link ExactSearcherContext}
     * @param iterator {@link KNNIterator}
     * @return Map of docId and score
     * @throws IOException exception raised by iterator during traversal
     */
    private Map<Integer, Float> doRadialSearch(
        LeafReaderContext leafReaderContext,
        ExactSearcherContext exactSearcherContext,
        KNNIterator iterator
    ) throws IOException {
        final SegmentReader reader = Lucene.segmentReader(leafReaderContext.reader());
        final KNNQuery knnQuery = exactSearcherContext.getKnnQuery();
        final FieldInfo fieldInfo = FieldInfoExtractor.getFieldInfo(reader, knnQuery.getField());
        if (fieldInfo == null) {
            return Collections.emptyMap();
        }
        final KNNEngine engine = FieldInfoExtractor.extractKNNEngine(fieldInfo);
        if (KNNEngine.FAISS != engine) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Engine [%s] does not support radial search", engine));
        }
        final SpaceType spaceType = FieldInfoExtractor.getSpaceType(modelDao, fieldInfo);
        final float minScore = spaceType.scoreTranslation(knnQuery.getRadius());
        return filterDocsByMinScore(exactSearcherContext, leafReaderContext, iterator, minScore);
    }

    private void gatherScoresForIds (ConcurrentMap<Integer, Float> docToScore, List<Integer> ids, final LeafReaderContext leafReaderContext,
                                     final ExactSearcherContext exactSearcherContext) throws IOException {
        KNNIterator iterator1 = getCopyKNNIterator(leafReaderContext, exactSearcherContext);
        for (Integer id : ids) {
            iterator1.advanceToId(id);
            ScoreDoc idAndScore = iterator1.score();
            docToScore.put(idAndScore.doc, idAndScore.score);
        }
    }

    private Map<Integer, Float> scoreAllDocs(KNNIterator iterator, final LeafReaderContext leafReaderContext,
                                             final ExactSearcherContext exactSearcherContext) throws IOException {
        final ConcurrentMap<Integer, Float> docToScore = new ConcurrentHashMap<>();
        int docId;
        ArrayList<Integer> allDocIds = new ArrayList<>();

        while ((docId = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
            allDocIds.add(docId);
        }

        int totalIds = allDocIds.size();
        int idsPerThread = totalIds / numThreads;
        int remainingIds = totalIds % numThreads;
        List<Callable<Void>> tasks = new ArrayList<>(numThreads);

        int startIndex = 0;
        for (int i = 0; i < numThreads; i++) {
            int endIndex = startIndex + idsPerThread + (i < remainingIds ? 1 : 0);
            int startIndexCopy = startIndex;

            tasks.add(() -> {
                gatherScoresForIds(docToScore, allDocIds.subList(startIndexCopy, endIndex), leafReaderContext, exactSearcherContext);
                return null;
            });

            startIndex = endIndex;
        }

        try {
            threadPool.executor(EXACT_SEARCH_THREAD_POOL).invokeAll(tasks);
        } catch (InterruptedException e) {}

        return docToScore;
    }

    private void gatherAndAddScoresToHeap (HitQueue hitQueue, Predicate<Float> filterScore, List<Integer> ids, final LeafReaderContext leafReaderContext,
                                           final ExactSearcherContext exactSearcherContext) throws IOException {
        KNNIterator iterator1 = getCopyKNNIterator(leafReaderContext, exactSearcherContext);
        for (Integer id : ids) {
            iterator1.advanceToId(id);
            ScoreDoc idAndScore = iterator1.score();
            addToHeap(hitQueue, filterScore, idAndScore);
        }
    }

    private synchronized void addToHeap(HitQueue hitQueue, Predicate<Float> filterScore, ScoreDoc docIdScore) {
        ScoreDoc topDoc = hitQueue.top();
        if (filterScore.test(docIdScore.score) && docIdScore.score > topDoc.score) {
            topDoc.score = docIdScore.score;
            topDoc.doc = docIdScore.doc;
            hitQueue.updateTop();
        }
    }

    private Map<Integer, Float> searchTopCandidates(KNNIterator iterator, int limit, @NonNull Predicate<Float> filterScore,
                                                    final LeafReaderContext leafReaderContext, final ExactSearcherContext exactSearcherContext)
        throws IOException {
        // Creating min heap and init with MAX DocID and Score as -INF.
        final HitQueue queue = new HitQueue(limit, true);
        final Map<Integer, Float> docToScore = new HashMap<>();
        int docId;

        ArrayList<Integer> allDocIds = new ArrayList<>();

        while ((docId = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
            allDocIds.add(docId);
        }

        int totalIds = allDocIds.size();
        int idsPerThread = totalIds / numThreads;
        int remainingIds = totalIds % numThreads;
        List<Callable<Void>> tasks = new ArrayList<>(numThreads);

        int startIndex = 0;
        for (int i = 0; i < numThreads; i++) {
            int endIndex = startIndex + idsPerThread + (i < remainingIds ? 1 : 0);
            int startIndexCopy = startIndex;
            tasks.add(() -> {
                gatherAndAddScoresToHeap(queue, filterScore, allDocIds.subList(startIndexCopy, endIndex), leafReaderContext, exactSearcherContext);
                return null;
            });
            startIndex = endIndex;
        }

        try {
            threadPool.executor(EXACT_SEARCH_THREAD_POOL).invokeAll(tasks);
        } catch (InterruptedException e) {}

        while (queue.size() > 0 && queue.top().score < 0) {
            queue.pop();
        }

        while (queue.size() > 0) {
            final ScoreDoc doc = queue.pop();
            docToScore.put(doc.doc, doc.score);
        }
        return docToScore;
    }

    private Map<Integer, Float> filterDocsByMinScore(ExactSearcherContext context, LeafReaderContext leafReaderContext, KNNIterator iterator, float minScore)
        throws IOException {
        int maxResultWindow = context.getKnnQuery().getContext().getMaxResultWindow();
        Predicate<Float> scoreGreaterThanOrEqualToMinScore = score -> score >= minScore;
        return searchTopCandidates(iterator, maxResultWindow, scoreGreaterThanOrEqualToMinScore, leafReaderContext, context);
    }

    private KNNIterator getCopyKNNIterator(LeafReaderContext leafReaderContext, ExactSearcherContext exactSearcherContext) throws IOException {
        final KNNQuery knnQuery = exactSearcherContext.getKnnQuery();
        final SegmentReader reader = Lucene.segmentReader(leafReaderContext.reader());
        final FieldInfo fieldInfo = FieldInfoExtractor.getFieldInfo(reader, knnQuery.getField());
        if (fieldInfo == null) {
            log.debug("[KNN] Cannot get KNNIterator as Field info not found for {}:{}", knnQuery.getField(), reader.getSegmentName());
            return null;
        }
        final SpaceType spaceType = FieldInfoExtractor.getSpaceType(modelDao, fieldInfo);

        boolean isNestedRequired = exactSearcherContext.isParentHits() && knnQuery.getParentsFilter() != null;

        if (VectorDataType.BINARY == knnQuery.getVectorDataType()) {
            final KNNVectorValues<byte[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
            if (isNestedRequired) {
                return new NestedBinaryVectorIdsKNNIterator(
                        knnQuery.getByteQueryVector(),
                        (KNNBinaryVectorValues) vectorValues,
                        spaceType,
                        knnQuery.getParentsFilter().getBitSet(leafReaderContext)
                );
            }
            return new BinaryVectorIdsKNNIterator(
                    knnQuery.getByteQueryVector(),
                    (KNNBinaryVectorValues) vectorValues,
                    spaceType
            );
        }

        if (VectorDataType.BYTE == knnQuery.getVectorDataType()) {
            final KNNVectorValues<byte[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
            if (isNestedRequired) {
                return new NestedByteVectorIdsKNNIterator(
                        knnQuery.getQueryVector(),
                        (KNNByteVectorValues) vectorValues,
                        spaceType,
                        knnQuery.getParentsFilter().getBitSet(leafReaderContext)
                );
            }
            return new ByteVectorIdsKNNIterator(knnQuery.getQueryVector(), (KNNByteVectorValues) vectorValues, spaceType);
        }

        final KNNVectorValues<float[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        if (isNestedRequired) {
            return new NestedVectorIdsKNNIterator(
                    knnQuery.getQueryVector(),
                    (KNNFloatVectorValues) vectorValues,
                    spaceType,
                    knnQuery.getParentsFilter().getBitSet(leafReaderContext)
            );
        }
        return new VectorIdsKNNIterator(
                knnQuery.getQueryVector(),
                (KNNFloatVectorValues) vectorValues,
                spaceType
        );
    }

    private KNNIterator getKNNIterator(LeafReaderContext leafReaderContext, ExactSearcherContext exactSearcherContext) throws IOException {
        final KNNQuery knnQuery = exactSearcherContext.getKnnQuery();
        final DocIdSetIterator matchedDocs = exactSearcherContext.getMatchedDocsIterator();
        final SegmentReader reader = Lucene.segmentReader(leafReaderContext.reader());
        final FieldInfo fieldInfo = FieldInfoExtractor.getFieldInfo(reader, knnQuery.getField());
        if (fieldInfo == null) {
            log.debug("[KNN] Cannot get KNNIterator as Field info not found for {}:{}", knnQuery.getField(), reader.getSegmentName());
            return null;
        }
        final SpaceType spaceType = FieldInfoExtractor.getSpaceType(modelDao, fieldInfo);

        boolean isNestedRequired = exactSearcherContext.isParentHits() && knnQuery.getParentsFilter() != null;

        if (VectorDataType.BINARY == knnQuery.getVectorDataType()) {
            final KNNVectorValues<byte[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
            if (isNestedRequired) {
                return new NestedBinaryVectorIdsKNNIterator(
                    matchedDocs,
                    knnQuery.getByteQueryVector(),
                    (KNNBinaryVectorValues) vectorValues,
                    spaceType,
                    knnQuery.getParentsFilter().getBitSet(leafReaderContext)
                );
            }
            return new BinaryVectorIdsKNNIterator(
                matchedDocs,
                knnQuery.getByteQueryVector(),
                (KNNBinaryVectorValues) vectorValues,
                spaceType
            );
        }

        if (VectorDataType.BYTE == knnQuery.getVectorDataType()) {
            final KNNVectorValues<byte[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
            if (isNestedRequired) {
                return new NestedByteVectorIdsKNNIterator(
                    matchedDocs,
                    knnQuery.getQueryVector(),
                    (KNNByteVectorValues) vectorValues,
                    spaceType,
                    knnQuery.getParentsFilter().getBitSet(leafReaderContext)
                );
            }
            return new ByteVectorIdsKNNIterator(matchedDocs, knnQuery.getQueryVector(), (KNNByteVectorValues) vectorValues, spaceType);
        }
        final byte[] quantizedQueryVector;
        final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo;
        if (exactSearcherContext.isUseQuantizedVectorsForSearch()) {
            // Build Segment Level Quantization info.
            segmentLevelQuantizationInfo = SegmentLevelQuantizationInfo.build(reader, fieldInfo, knnQuery.getField());
            // Quantize the Query Vector Once.
            quantizedQueryVector = SegmentLevelQuantizationUtil.quantizeVector(knnQuery.getQueryVector(), segmentLevelQuantizationInfo);
        } else {
            segmentLevelQuantizationInfo = null;
            quantizedQueryVector = null;
        }

        final KNNVectorValues<float[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        if (isNestedRequired) {
            return new NestedVectorIdsKNNIterator(
                matchedDocs,
                knnQuery.getQueryVector(),
                (KNNFloatVectorValues) vectorValues,
                spaceType,
                knnQuery.getParentsFilter().getBitSet(leafReaderContext),
                quantizedQueryVector,
                segmentLevelQuantizationInfo
            );
        }
        return new VectorIdsKNNIterator(
            matchedDocs,
            knnQuery.getQueryVector(),
            (KNNFloatVectorValues) vectorValues,
            spaceType,
            quantizedQueryVector,
            segmentLevelQuantizationInfo
        );
    }

    /**
     * Stores the context that is used to do the exact search. This class will help in reducing the explosion of attributes
     * for doing exact search.
     */
    @Value
    @Builder
    public static class ExactSearcherContext {
        /**
         * controls whether we should use Quantized vectors during exact search or not. This is useful because when we do
         * re-scoring we need to re-score using full precision vectors and not quantized vectors.
         */
        boolean useQuantizedVectorsForSearch;
        int k;
        DocIdSetIterator matchedDocsIterator;
        long numberOfMatchedDocs;
        KNNQuery knnQuery;
        /**
         * whether the matchedDocs contains parent ids or child ids. This is relevant in the case of
         * filtered nested search where the matchedDocs contain the parent ids and {@link NestedVectorIdsKNNIterator}
         * needs to be used.
         */
        boolean isParentHits;
    }
}
