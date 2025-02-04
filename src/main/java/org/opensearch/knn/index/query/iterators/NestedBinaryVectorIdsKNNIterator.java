/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.util.BitSet;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.vectorvalues.KNNBinaryVectorValues;

import java.io.IOException;

/**
 * This iterator iterates filterIdsArray to scoreif filter is provided else it iterates over all docs.
 * However, it dedupe docs per each parent doc
 * of which ID is set in parentBitSet and only return best child doc with the highest score.
 */
public class NestedBinaryVectorIdsKNNIterator extends BinaryVectorIdsKNNIterator {
    private final BitSet parentBitSet;
    int prevParent;

    public NestedBinaryVectorIdsKNNIterator(
        @Nullable final DocIdSetIterator filterIdsIterator,
        final byte[] queryVector,
        final KNNBinaryVectorValues binaryVectorValues,
        final SpaceType spaceType,
        final BitSet parentBitSet
    ) {
        super(filterIdsIterator, queryVector, binaryVectorValues, spaceType);
        this.parentBitSet = parentBitSet;
        this.docId = -1;
        prevParent = -1;
    }

    public NestedBinaryVectorIdsKNNIterator(
        final byte[] queryVector,
        final KNNBinaryVectorValues binaryVectorValues,
        final SpaceType spaceType,
        final BitSet parentBitSet
    ) {
        super(null, queryVector, binaryVectorValues, spaceType);
        this.parentBitSet = parentBitSet;
        this.docId = -1;
        prevParent = -1;
    }

    @Override
    public ScoreDoc score() throws IOException {
        if (prevParent + 1 != binaryVectorValues.docId()) {
            binaryVectorValues.advance(prevParent + 1);
        }
        int iterDocId = prevParent + 1;
        int bestChild = -1;
        currentScore = Float.NEGATIVE_INFINITY;
        while (iterDocId != DocIdSetIterator.NO_MORE_DOCS && iterDocId < docId) {
            float score = computeScore(binaryVectorValues.getVector());
            if (score > currentScore) {
                bestChild = iterDocId;
                currentScore = score;
            }
            iterDocId = getNextDocId();
        }

        return new ScoreDoc(bestChild, currentScore);
    }

    /**
     * Advance to the next best child doc per parent and update score with the best score among child docs from the parent.
     * DocIdSetIterator.NO_MORE_DOCS is returned when there is no more docs
     *
     * @return next best child doc id
     */
    @Override
    public int nextDoc() throws IOException {
        if (docId == DocIdSetIterator.NO_MORE_DOCS) {
            return DocIdSetIterator.NO_MORE_DOCS;
        }
        int currentParent;

        if (docId == -1) {
            currentParent = parentBitSet.nextSetBit(docId + 1);
        } else {
            currentParent = parentBitSet.nextSetBit(docId);
        }

        // In order to traverse all children for given parent, we have to use docId < parentId, because,
        // kNNVectorValues will not have parent id since DocId is unique per segment. For ex: let's say for doc id 1, there is one child
        // and for doc id 5, there are three children. In that case knnVectorValues iterator will have [0, 2, 3, 4]
        // and parentBitSet will have [1,5]
        // Hence, we have to iterate till docId from knnVectorValues is less than parentId instead of till equal to parentId
        /*
        while (docId != DocIdSetIterator.NO_MORE_DOCS && docId < currentParent) {
            float score = computeScore();
            if (score > currentScore) {
                bestChild = docId;
                currentScore = score;
            }
            docId = getNextDocId();
        }

        */
        while (docId != DocIdSetIterator.NO_MORE_DOCS && docId < currentParent) {
            docId = getNextDocId();
        }

        return currentParent;
    }

    @Override
    public void advanceToId(int advanceDocId) throws IOException {
        prevParent = parentBitSet.prevSetBit(advanceDocId - 1);
        docId = advanceDocId;
    }
}
