/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.common.Nullable;
import org.apache.lucene.search.ScoreDoc;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.vectorvalues.KNNBinaryVectorValues;

import java.io.IOException;

/**
 * Inspired by DiversifyingChildrenFloatKnnVectorQuery in lucene
 * https://github.com/apache/lucene/blob/7b8aece125aabff2823626d5b939abf4747f63a7/lucene/join/src/java/org/apache/lucene/search/join/DiversifyingChildrenFloatKnnVectorQuery.java#L162
 *
 * The class is used in KNNWeight to score all docs, but, it iterates over filterIdsArray if filter is provided
 */
public class BinaryVectorIdsKNNIterator implements KNNIterator {
    protected final DocIdSetIterator docIdSetIterator;
    protected final byte[] queryVector;
    protected final KNNBinaryVectorValues binaryVectorValues;
    protected final SpaceType spaceType;
    protected float currentScore = Float.NEGATIVE_INFINITY;
    protected int docId;

    public BinaryVectorIdsKNNIterator(
        @Nullable final DocIdSetIterator docIdSetIterator,
        final byte[] queryVector,
        final KNNBinaryVectorValues binaryVectorValues,
        final SpaceType spaceType
    ) {
        this.docIdSetIterator = docIdSetIterator;
        this.queryVector = queryVector;
        this.binaryVectorValues = binaryVectorValues;
        this.spaceType = spaceType;
        // This cannot be moved inside nextDoc() method since it will break when we have nested field, where
        // nextDoc should already be referring to next knnVectorValues
        this.docId = -1;
    }

    public BinaryVectorIdsKNNIterator(final byte[] queryVector, final KNNBinaryVectorValues binaryVectorValues, final SpaceType spaceType) {
        this(null, queryVector, binaryVectorValues, spaceType);
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
        final byte[] vector = binaryVectorValues.getVector();
        return new ScoreDoc(docId, computeScore(vector));
    }

    protected float computeScore(byte[] vector) {
        // Calculates a similarity score between the two vectors with a specified function. Higher similarity
        // scores correspond to closer vectors.
        return spaceType.getKnnVectorSimilarityFunction().compare(queryVector, vector);
    }

    protected int getNextDocId() throws IOException {
        if (docIdSetIterator == null) {
            return binaryVectorValues.nextDoc();
        }
        int nextDocID = this.docIdSetIterator.nextDoc();
        // For filter case, advance vector values to corresponding doc id from filter bit set
        if (nextDocID != DocIdSetIterator.NO_MORE_DOCS) {
            binaryVectorValues.advance(nextDocID);
        }
        return nextDocID;
    }

    public void advanceToId(int advanceDocId) throws IOException {
        binaryVectorValues.advance(advanceDocId);
        docId = advanceDocId;
    }
}
