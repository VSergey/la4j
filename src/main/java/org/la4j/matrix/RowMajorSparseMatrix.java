/*
 * Copyright 2011-2014, by Vladimir Kostyukov and Contributors.
 *
 * This file is part of la4j project (http://la4j.org)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Contributor(s): -
 *
 */

package org.la4j.matrix;

import org.la4j.Matrices;
import org.la4j.iterator.MatrixIterator;
import org.la4j.iterator.VectorIterator;
import org.la4j.Matrix;
import org.la4j.operation.MatrixMatrixOperation;
import org.la4j.operation.MatrixOperation;
import org.la4j.operation.MatrixVectorOperation;
import org.la4j.Vector;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public abstract class RowMajorSparseMatrix extends SparseMatrix {

    public RowMajorSparseMatrix(int rows, int columns) {
        super(rows, columns);
    }

    public RowMajorSparseMatrix(int rows, int columns, int cardinality) {
        super(rows, columns, cardinality);
    }

    @Override
    public MatrixFactory factory() {
        return Matrices.SPARSE_ROW_MAJOR;
    }

    public boolean isRowMajor() {
        return true;
    }

    @Override
    public Matrix transpose() {
        Matrix result = Matrices.CCS.zero(columns, rows);
        MatrixIterator it = nonZeroRowMajorIterator();

        while (it.hasNext()) {
            double x = it.next();
            int i = it.rowIndex();
            int j = it.columnIndex();
            result.set(j, i, x);
        }

        return result;
    }

    @Override
    public Matrix rotate() {
        Matrix result = Matrices.CCS.zero(columns, rows);

        Iterator<Integer> nzRows = iteratorOfNonZeroRows();
        List<Integer> reversedNzRows = new LinkedList<>();
        while (nzRows.hasNext()) {
            reversedNzRows.add(0, nzRows.next());
        }

        for (int i: reversedNzRows) {
            VectorIterator it = nonZeroIteratorOfRow(i);
            while (it.hasNext()) {
                double x = it.next();
                int j = it.index();
                result.set(j, rows - 1 - i, x);
            }
        }

        return result;
    }

    public abstract Iterator<Integer> iteratorOfNonZeroRows();

    public <T> T apply(MatrixOperation<T> operation) {
        operation.ensureApplicableTo(this);
        return operation.apply(this);
    }

    public <T> T apply(MatrixMatrixOperation<T> operation, Matrix that) {
        return that.apply(operation.partiallyApply(this));
    }

    public <T> T apply(MatrixVectorOperation<T> operation, Vector that) {
        return that.apply(operation.partiallyApply(this));
    }
}
