/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.ScoreDoc;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.vectorvalues.KNNByteVectorValues;

import java.io.IOException;

/**
 * Inspired by DiversifyingChildrenFloatKnnVectorQuery in lucene
 * https://github.com/apache/lucene/blob/7b8aece125aabff2823626d5b939abf4747f63a7/lucene/join/src/java/org/apache/lucene/search/join/DiversifyingChildrenFloatKnnVectorQuery.java#L162
 *
 * The class is used in KNNWeight to score all docs, but, it iterates over filterIdsArray if filter is provided
 */
public class ByteVectorIdsKNNIterator implements KNNIterator {
    protected final DocIdSetIterator filterIdsIterator;
    protected final float[] queryVector;
    protected final KNNByteVectorValues byteVectorValues;
    protected final SpaceType spaceType;
    protected float currentScore = Float.NEGATIVE_INFINITY;
    protected int docId;

    public ByteVectorIdsKNNIterator(
        @Nullable final DocIdSetIterator filterIdsIterator,
        final float[] queryVector,
        final KNNByteVectorValues byteVectorValues,
        final SpaceType spaceType
    ) {
        this.filterIdsIterator = filterIdsIterator;
        this.queryVector = queryVector;
        this.byteVectorValues = byteVectorValues;
        this.spaceType = spaceType;
        // This cannot be moved inside nextDoc() method since it will break when we have nested field, where
        // nextDoc should already be referring to next knnVectorValues
        this.docId = -1;
    }

    public ByteVectorIdsKNNIterator(final float[] queryVector, final KNNByteVectorValues byteVectorValues, final SpaceType spaceType) {
        this(null, queryVector, byteVectorValues, spaceType);
    }

    /**
     * Advance to the next doc and update score value with score of the next doc.
     * DocIdSetIterator.NO_MORE_DOCS is returned when there is no more docs
     *
     * @return next doc id
     */
    @Override
    public int nextDoc() throws IOException {
        docId = getNextDocId();
        return docId;
    }

    @Override
    public ScoreDoc score() throws IOException {
        final byte[] vector = byteVectorValues.getVector();
        return new ScoreDoc(docId, computeScore(vector));
    }

    protected float computeScore(byte[] vector) {
        // Calculates a similarity score between the two vectors with a specified function. Higher similarity
        // scores correspond to closer vectors.

        // The query vector of Faiss byte vector is a Float array because ScalarQuantizer accepts it as float array.
        // To compute the score between this query vector and each vector in KNNByteVectorValues we are casting this query vector into byte
        // array directly.
        // This is safe to do so because float query vector already has validated byte values. Do not reuse this direct cast at any other
        // place.
        final byte[] byteQueryVector = new byte[queryVector.length];
        for (int i = 0; i < queryVector.length; i++) {
            byteQueryVector[i] = (byte) queryVector[i];
        }
        return spaceType.getKnnVectorSimilarityFunction().compare(byteQueryVector, vector);
    }

    protected int getNextDocId() throws IOException {
        if (filterIdsIterator == null) {
            return byteVectorValues.nextDoc();
        }
        int nextDocID = this.filterIdsIterator.nextDoc();
        // For filter case, advance vector values to corresponding doc id from filter bit set
        if (nextDocID != DocIdSetIterator.NO_MORE_DOCS) {
            byteVectorValues.advance(nextDocID);
        }
        return nextDocID;
    }

    public void advanceToId(int advanceDocId) throws IOException {
        byteVectorValues.advance(advanceDocId);
        docId = advanceDocId;
    }
}
