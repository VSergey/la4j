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

package org.la4j.vector;

import org.la4j.*;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.operation.VectorMatrixOperation;
import org.la4j.operation.VectorOperation;
import org.la4j.operation.VectorVectorOperation;

import java.text.NumberFormat;

/**
 * A dense vector.
 * 
 * A vector represents an array of elements. It can be re-sized.
 * 
 * A dense data structure usually stores data in an underlying array. Zero elements
 * take up memory space. If you want a data structure that will not have zero
 * elements take up memory space, try a sparse structure.
 * 
 * However, fetch/store operations on dense data structures only take O(1) time,
 * instead of the O(log n) time on sparse structures.
 * 
 */
public abstract class DenseVector extends AbstractVector {

    public DenseVector(int length) {
        super(length);
    }

    public <T> T apply(VectorOperation<T> operation) {
        operation.ensureApplicableTo(this);
        return operation.apply(this);
    }

    public <T> T apply(VectorVectorOperation<T> operation, Vector that) {
        return that.apply(operation.partiallyApply(this));
    }

    public <T> T apply(VectorMatrixOperation<T> operation, Matrix that) {
        return that.apply(operation.partiallyApply(this));
    }

    /**
     * Converts this dense vector to a double array.
     *
     * @return an array representation of this vector
     */
    public abstract double[] toArray();

    public Matrix toRowMatrix() {
        Matrix result = Matrices.BASIC_2D.zero(1, length);

        for (int j = 0; j < length; j++) {
            result.set(0, j, get(j));
        }

        return result;
    }

    public Matrix toColumnMatrix() {
        Matrix result = Matrices.BASIC_2D.zero(length, 1);

        for (int i = 0; i < length; i++) {
            result.set(i, 0, get(i));
        }

        return result;
    }

    public Matrix toDiagonalMatrix() {
        Matrix result = Matrices.BASIC_2D.zero(length, length);

        for (int i = 0; i < length; i++) {
            result.set(i, i, get(i));
        }

        return result;
    }

    public String toMatrixMarket(NumberFormat formatter) {
        StringBuilder out = new StringBuilder();

        out.append("%%MatrixMarket vector array real\n");
        out.append(length).append('\n');
        for (int i = 0; i < length; i++) {
            out.append(formatter.format(get(i))).append('\n');
        }

        return out.toString();
    }
}
