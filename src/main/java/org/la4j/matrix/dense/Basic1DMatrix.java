/*
 * Copyright 2011-2013, by Vladimir Kostyukov and Contributors.
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
 * Contributor(s): Wajdy Essam
 * 
 */

package org.la4j.matrix.dense;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Random;

import org.la4j.*;
import org.la4j.matrix.DenseMatrix;
import org.la4j.matrix.MatrixFactory;
import org.la4j.vector.dense.BasicVector;

public class Basic1DMatrix extends DenseMatrix {

    private double[] self;

    public Basic1DMatrix() {
        this(0, 0);
    }

    public Basic1DMatrix(int rows, int columns) {
        this(rows, columns, new double[rows * columns]);
    }

    public Basic1DMatrix(int rows, int columns, double[] array) {
        super(rows, columns);
        this.self = array;
    }

    /**
     * Creates a constant {@link Basic1DMatrix} of the given shape and {@code value}.
     */
    public static Basic1DMatrix constant(int rows, int columns, double constant) {
        double[] array = new double[rows * columns];
        Arrays.fill(array, constant);
        return new Basic1DMatrix(rows, columns, array);
    }

    /**
     * Creates a random {@link Basic1DMatrix} of the given shape:
     * {@code rows} x {@code columns}.
     */
    public static Basic1DMatrix random(int rows, int columns, Random random) {
        double[] array = new double[rows * columns];
        for (int i = 0; i < rows * columns; i++) {
            array[i] = random.nextDouble();
        }
        return new Basic1DMatrix(rows, columns, array);
    }

    /**
     * Creates a random symmetric {@link Basic1DMatrix} of the given {@code size}.
     */
    public static Basic1DMatrix randomSymmetric(int size, Random random) {
        double[] array = new double[size * size];
        for (int i = 0; i < size; i++) {
            for (int j = i; j < size; j++) {
                double value = random.nextDouble();
                array[i * size + j] = value;
                array[j * size + i] = value;
            }
        }
        return new Basic1DMatrix(size, size, array);
    }

    @Override
    public MatrixFactory factory() {
        return Matrices.BASIC_1D;
    }

    public double get(int i, int j) {
        ensureIndexesAreInBounds(i, j);
        return self[i * columns + j];
    }

    public void set(int i, int j, double value) {
        ensureIndexesAreInBounds(i, j);
        self[i * columns + j] = value;
    }

    @Override
    public void setAll(double value) {
        Arrays.fill(self, value);
    }

    @Override
    public void swapRows(int i, int j) {
        if (i != j) {
            for (int k = 0; k < columns; k++) {
                double tmp = self[i * columns + k];
                self[i * columns + k] = self[j * columns + k];
                self[j * columns + k] = tmp;
            }
        }
    }

    @Override
    public void swapColumns(int i, int j) {
        if (i != j) {
            for (int k = 0; k < rows; k++) {
                double tmp  = self[k * columns + i];
                self[k * columns + i] = self[k * columns + j];
                self[k * columns + j] = tmp;
            }
        }
    }

    @Override
    public Vector getRow(int i) {
        double[] result = new double[columns];
        System.arraycopy(self, i * columns , result, 0, columns);

        return new BasicVector(result);
    }

    public Matrix copyOfShape(int rows, int columns) {
        ensureDimensionsAreCorrect(rows, columns);

        if (this.rows < rows && this.columns == columns) {
            double[] $self = new double[rows * columns];
            System.arraycopy(self, 0, $self, 0, this.rows * columns);

            return new Basic1DMatrix(rows, columns, $self);
        }

        double[] $self = new double[rows * columns];

        int columnSize = columns < this.columns ? columns : this.columns;
        int rowSize =  rows < this.rows ? rows : this.rows;

        for (int i = 0; i < rowSize; i++) {
            System.arraycopy(self, i * this.columns, $self, i * columns, 
                             columnSize);
        }

        return new Basic1DMatrix(rows, columns, $self);
    }

    public double[][] toArray() {

        double[][] result = new double[rows][columns];

        int offset = 0;
        for (int i = 0; i < rows; i++) {
            System.arraycopy(self, offset, result[i], 0, columns);
            offset += columns;
        }

        return result;
    }

    public byte[] toBinary() {
        int size = 1 +                  // 1 byte: class tag
                   4 +                  // 4 bytes: rows
                   4 +                  // 4 bytes: columns
                  (8 * rows * columns); // 8 * rows * columns bytes: values

        ByteBuffer buffer = ByteBuffer.allocate(size);

        buffer.put(MatrixFactory.BASIC_1_MATRIX_TAG);
        buffer.putInt(rows);
        buffer.putInt(columns);
        for (double value: self) {
            buffer.putDouble(value);
        }

        return buffer.array();
    }
}
