/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import junit.framework.TestCase;
import lombok.SneakyThrows;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class NestedVectorIdsKNNIteratorTests extends TestCase {
    @SneakyThrows
    public void testNextDoc_whenIterate_ReturnBestChildDocsPerParent() {
        final SpaceType spaceType = SpaceType.L2;
        final float[] queryVector = { 1.0f, 2.0f, 3.0f };
        final int[] filterIds = { 0, 2, 3 };
        // Parent id for 0 -> 1
        // Parent id for 2, 3 -> 4
        // In bit representation, it is 10010. In long, it is 18.
        final BitSet parentBitSet = new FixedBitSet(new long[] { 18 }, 5);
        final List<float[]> dataVectors = Arrays.asList(
            new float[] { 11.0f, 12.0f, 13.0f },
            new float[] { 17.0f, 18.0f, 19.0f },
            new float[] { 14.0f, 15.0f, 16.0f }
        );
        final List<Float> expectedScores = dataVectors.stream()
            .map(vector -> spaceType.getKnnVectorSimilarityFunction().compare(queryVector, vector))
            .collect(Collectors.toList());

        KNNFloatVectorValues values = mock(KNNFloatVectorValues.class);
        when(values.getVector()).thenReturn(dataVectors.get(0), dataVectors.get(1), dataVectors.get(2));
        // final List<BytesRef> byteRefs = dataVectors.stream()
        // .map(vector -> new BytesRef(new KNNVectorAsArraySerializer().floatToByteArray(vector)))
        // .collect(Collectors.toList());
        // when(values.binaryValue()).thenReturn(byteRefs.get(0), byteRefs.get(1), byteRefs.get(2));

        FixedBitSet filterBitSet = new FixedBitSet(4);
        for (int id : filterIds) {
            when(values.advance(id)).thenReturn(id);
            filterBitSet.set(id);
        }

        // Execute and verify
        NestedVectorIdsKNNIterator iterator = new NestedVectorIdsKNNIterator(
            new BitSetIterator(filterBitSet, filterBitSet.length()),
            queryVector,
            values,
            spaceType,
            parentBitSet
        );

        iterator.advanceToId(1);
        assertEquals(expectedScores.get(0), iterator.score().score);
        iterator.advanceToId(4);
        assertEquals(expectedScores.get(2), iterator.score().score);
    }

    @SneakyThrows
    public void testNextDoc_whenIterateWithoutFilters_thenReturnBestChildDocsPerParent() {
        final SpaceType spaceType = SpaceType.L2;
        final float[] queryVector = { 1.0f, 2.0f, 3.0f };
        // Parent id for 0 -> 1
        // Parent id for 2, 3 -> 4
        // In bit representation, it is 10010. In long, it is 18.
        final BitSet parentBitSet = new FixedBitSet(new long[] { 18 }, 5);
        final List<float[]> dataVectors = Arrays.asList(
            new float[] { 11.0f, 12.0f, 13.0f },
            new float[] { 17.0f, 18.0f, 19.0f },
            new float[] { 14.0f, 15.0f, 16.0f }
        );
        final List<Float> expectedScores = dataVectors.stream()
            .map(vector -> spaceType.getKnnVectorSimilarityFunction().compare(queryVector, vector))
            .collect(Collectors.toList());

        KNNFloatVectorValues values = mock(KNNFloatVectorValues.class);
        when(values.getVector()).thenReturn(dataVectors.get(0), dataVectors.get(1), dataVectors.get(2));
        when(values.nextDoc()).thenReturn(0, 2, 3, Integer.MAX_VALUE);

        // Execute and verify
        NestedVectorIdsKNNIterator iterator = new NestedVectorIdsKNNIterator(queryVector, values, spaceType, parentBitSet);
        iterator.advanceToId(1);
        assertEquals(expectedScores.get(0), iterator.score().score);
        iterator.advanceToId(4);
        assertEquals(expectedScores.get(2), iterator.score().score);
    }
}
