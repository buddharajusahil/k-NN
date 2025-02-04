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
import org.opensearch.knn.index.query.SegmentLevelQuantizationInfo;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.search.suggest.term.TermSuggestion;

import javax.print.Doc;
import java.io.IOException;

/**
 * This iterator iterates filterIdsArray to score if filter is provided else it iterates over all docs.
 * However, it dedupe docs per each parent doc
 * of which ID is set in parentBitSet and only return best child doc with the highest score.
 */
public class NestedVectorIdsKNNIterator extends VectorIdsKNNIterator {
    private final BitSet parentBitSet;
    private int prevParent;

    public NestedVectorIdsKNNIterator(
        @Nullable final DocIdSetIterator filterIdsIterator,
        final float[] queryVector,
        final KNNFloatVectorValues knnFloatVectorValues,
        final SpaceType spaceType,
        final BitSet parentBitSet
    ) {
        this(filterIdsIterator, queryVector, knnFloatVectorValues, spaceType, parentBitSet, null, null);
    }

    public NestedVectorIdsKNNIterator(
        final float[] queryVector,
        final KNNFloatVectorValues knnFloatVectorValues,
        final SpaceType spaceType,
        final BitSet parentBitSet
    ) throws IOException {
        this(null, queryVector, knnFloatVectorValues, spaceType, parentBitSet, null, null);
    }

    public NestedVectorIdsKNNIterator(
        @Nullable final DocIdSetIterator filterIdsIterator,
        final float[] queryVector,
        final KNNFloatVectorValues knnFloatVectorValues,
        final SpaceType spaceType,
        final BitSet parentBitSet,
        final byte[] quantizedVector,
        final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo
    ) {
        super(filterIdsIterator, queryVector, knnFloatVectorValues, spaceType, quantizedVector, segmentLevelQuantizationInfo);
        this.parentBitSet = parentBitSet;
        this.docId = -1;
        prevParent = -1;
    }

    @Override
    public ScoreDoc score() throws IOException {
        if (prevParent + 1 != knnFloatVectorValues.docId()) {
            knnFloatVectorValues.advance(prevParent + 1);
        }
        int iterDocId = prevParent + 1;
        int bestChild = -1;
        currentScore = Float.NEGATIVE_INFINITY;
        while (iterDocId != DocIdSetIterator.NO_MORE_DOCS && iterDocId < docId) {
            float score = computeScore(knnFloatVectorValues.getVector());
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
        while (docId != DocIdSetIterator.NO_MORE_DOCS && docId < currentParent) {
            docId = getNextDocId();
        }
        return currentParent;
    }

    @Override
    public void advanceToId(int advanceDocId) {
        prevParent = parentBitSet.prevSetBit(advanceDocId - 1);
        docId = advanceDocId;
    }
}
