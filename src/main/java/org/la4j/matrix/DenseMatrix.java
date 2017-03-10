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

import org.la4j.*;
import org.la4j.operation.MatrixMatrixOperation;
import org.la4j.operation.MatrixOperation;
import org.la4j.operation.MatrixVectorOperation;

import java.text.NumberFormat;

public abstract class DenseMatrix extends AbstractMatrix {

    public DenseMatrix(int rows, int columns) {
        super(rows, columns);
    }

    @Override
    public MatrixFactory factory() {
        return Matrices.DENSE;
    }
    /**
     * Converts this dense matrix to double array.
     * 
     * @return an array representation of this matrix
     */
    public abstract double[][] toArray();

    public Vector getRow(int i) {
        Vector result = Vectors.DENSE.zero(columns);

        for (int j = 0; j < columns; j++) {
            result.set(j, get(i, j));
        }

        return result;
    }

    public Vector getColumn(int j) {
        Vector result = Vectors.DENSE.zero(rows);

        for (int i = 0; i < rows; i++) {
            result.set(i, get(i, j));
        }

        return result;
    }

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

    public String toMatrixMarket(NumberFormat formatter) {
        StringBuilder out = new StringBuilder();

        out.append("%%MatrixMarket matrix array real general\n");
        out.append(rows).append(' ').append(columns).append('\n');
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                out.append(formatter.format(get(i, j))).append('\n');
            }
        }

        return out.toString();
    }
}
