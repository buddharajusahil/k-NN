/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import org.apache.lucene.search.ScoreDoc;

import java.io.IOException;

public interface KNNIterator {
    int nextDoc() throws IOException;

    ScoreDoc score() throws IOException;

    void advanceToId(int advanceDocId) throws IOException;
}
