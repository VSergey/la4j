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
 * Contributor(s): Chandler May
 *                 Maxim Samoylov
 *                 Anveshi Charuvaka
 *                 Clement Skau
 *                 Catherine da Graca
 * 
 */

package org.la4j.matrix.sparse;

import java.nio.ByteBuffer;
import java.util.*;

import org.la4j.*;
import org.la4j.Vector;
import org.la4j.iterator.ColumnMajorMatrixIterator;
import org.la4j.iterator.VectorIterator;
import org.la4j.matrix.ColumnMajorSparseMatrix;
import org.la4j.matrix.MatrixFactory;
import org.la4j.matrix.functor.MatrixFunction;
import org.la4j.matrix.functor.MatrixProcedure;
import org.la4j.vector.functor.VectorProcedure;
import org.la4j.vector.sparse.CompressedVector;

/**
 * This is a CCS (Compressed Column Storage) matrix class.
 */
public class CCSMatrix extends ColumnMajorSparseMatrix {

    private static final int MINIMUM_SIZE = 32;

    private double[] values;
    private int[] rowIndices;
    private int[] columnPointers;

    public CCSMatrix() {
        this(0, 0);
    }

    public CCSMatrix(int rows, int columns) {
        this (rows, columns, 0);
    }

    public CCSMatrix(int rows, int columns, int capacity) {
        super(rows, columns);
        ensureCardinalityIsCorrect(rows, columns, capacity);

        int alignedSize = align(capacity);
        this.values = new double[alignedSize];
        this.rowIndices = new int[alignedSize];
        this.columnPointers = new int[columns + 1];
    }

    public CCSMatrix(int rows, int columns, int cardinality, double[] values, int[] rowIndices, int[] columnPointers) {
        super(rows, columns, cardinality);
        ensureCardinalityIsCorrect(rows, columns, cardinality);

        this.values = values;
        this.rowIndices = rowIndices;
        this.columnPointers = columnPointers;
    }

    @Override
    public MatrixFactory factory() {
        return Matrices.CCS;
    }

    /**
     * Creates a zero {@link CCSMatrix} of the given shape:
     * {@code rows} x {@code columns} with the given {@code capacity}.
     */
    public static CCSMatrix zero(int rows, int columns, int capacity) {
        return new CCSMatrix(rows, columns, capacity);
    }

    /**
     * Creates a random {@link CCSMatrix} of the given shape:
     * {@code rows} x {@code columns}.
     */
    public static CCSMatrix random(int rows, int columns, double density, Random random) {
        if (density < 0.0 || density > 1.0) {
            throw new IllegalArgumentException("The density value should be between 0 and 1.0");
        }

        int cardinality = Math.max((int)((rows * columns) * density), columns);

        double[] values = new double[cardinality];
        int[] rowIndices = new int[cardinality];
        int[] columnPointers = new int[columns + 1];

        int kk = cardinality / columns;
        int[] indices = new int[kk];

        int k = 0;
        for (int j = 0; j < columns; j++) {

            columnPointers[j] = k;

            for (int jj = 0; jj < kk; jj++) {
                indices[jj] = random.nextInt(rows);
            }

            Arrays.sort(indices);

            int previous = -1;
            for (int jj = 0; jj < kk; jj++) {

                if (indices[jj] == previous) {
                    continue;
                }

                values[k] = random.nextDouble();
                rowIndices[k++] = indices[jj];
                previous = indices[jj];
            }
        }

        columnPointers[columns] = cardinality;

        return new CCSMatrix(rows, columns, cardinality, values, rowIndices, columnPointers);
    }

    /**
     * Creates a random symmetric {@link CCSMatrix} of the given {@code size}.
     */
    public static CCSMatrix randomSymmetric(int size, double density, Random random) {
        int cardinality = (int) ((size * size) * density);

        // TODO: Issue 15
        // We can do better here. All we need to is to make sure
        // that all the writes to CCS matrix are done in a serial
        // order (column-major). This will give us O(1) performance
        // per write.

        CCSMatrix matrix = new CCSMatrix(size, size, cardinality);

        for (int k = 0; k < cardinality / 2; k++) {
            int i = random.nextInt(size);
            int j = random.nextInt(size);
            double value = random.nextDouble();

            matrix.set(i, j, value);
            matrix.set(j, i, value);
        }

        return matrix;
    }

    public double getOrElse(int i, int j, double defaultValue) {
        ensureIndexesAreInBounds(i, j);
        int k = searchForRowIndex(i, columnPointers[j], columnPointers[j + 1]);

        if (k < columnPointers[j + 1] && rowIndices[k] == i) {
            return values[k];
        }

        return defaultValue;
    }

    public void set(int i, int j, double value) {
        ensureIndexesAreInBounds(i, j);
        int k = searchForRowIndex(i, columnPointers[j], columnPointers[j + 1]);

        if (k < columnPointers[j + 1] && rowIndices[k] == i) {
            // if (Math.abs(value) < Matrices.EPS && value >= 0.0) {
            if (value == 0.0) {
                remove(k, j);
            } else {
                values[k] = value;
            }
        } else {
            insert(k, i, j, value);
        }
    }

    @Override
    public void setAll(double value) {
        if (value == 0.0) {
            cardinality = 0;
        } else {
            int size = (int) capacity();

            if (values.length < size) {
                values = new double[size];
                rowIndices = new int[size];
                columnPointers = new int[columns + 1];
            }

            for (int j = 0; j < columns; j++) {
                for (int i = 0; i < rows; i++) {
                    values[j * rows + i] = value;
                    rowIndices[j * rows + i] = i;
                }
                columnPointers[j] = rows * j;
            }

            columnPointers[columns] = size;
            cardinality = size;
        }
    }

    @Override
    public Vector getColumn(int j) {
        int columnCardinality = columnPointers[j + 1] - columnPointers[j];
        double[] columnValues = new double[columnCardinality];
        int[] columnIndices = new int[columnCardinality];

        System.arraycopy(values, columnPointers[j], columnValues, 0, columnCardinality);
        System.arraycopy(rowIndices, columnPointers[j], columnIndices, 0, columnCardinality);

        return new CompressedVector(rows, columnCardinality, columnValues, columnIndices);
    }

    @Override
    public Vector getRow(int i) {
        Vector result = Vectors.COMPRESSED.zero(columns);
        int j = 0;

        while (columnPointers[j] < cardinality) {
            int k = searchForRowIndex(i, columnPointers[j], columnPointers[j + 1]);
            if (k < columnPointers[j + 1] && rowIndices[k] == i) {
                result.set(j, values[k]);
            }
            j++;
        }

        return result;
    }

    public Matrix copyOfShape(int rows, int columns) {
        ensureDimensionsAreCorrect(rows, columns);

        if (rows >= this.rows && columns >= this.columns) {
            double[] $values = new double[align(cardinality)];
            int[] $rowIndices = new int[align(cardinality)];
            int[] $columnPointers = new int[columns + 1];

            System.arraycopy(values, 0, $values, 0, cardinality);
            System.arraycopy(rowIndices, 0, $rowIndices, 0, cardinality);
            System.arraycopy(columnPointers, 0, $columnPointers, 0, this.columns + 1);

            for (int i = this.columns; i < columns + 1; i++) {
                $columnPointers[i] = cardinality;
            }

            return new CCSMatrix(rows, columns, cardinality, $values, $rowIndices, $columnPointers);
        }

        double[] $values = new double[align(cardinality)];
        int[] $rowIndices = new int[align(cardinality)];
        int[] $columnPointers = new int[columns + 1];

        int $cardinality = 0;

        int k = 0;
        int j = 0;
        while (k < cardinality && j < columns) {

            $columnPointers[j] = $cardinality;

            for (int i = columnPointers[j]; i < columnPointers[j + 1]
                    && rowIndices[i] < rows; i++, k++) {

                $values[$cardinality] = values[i];
                $rowIndices[$cardinality] = rowIndices[i];
                $cardinality++;
            }
            j++;
        }

        for (; j < columns + 1; j++) {
            $columnPointers[j] = $cardinality;
        }

        return new CCSMatrix(rows, columns, $cardinality, $values, $rowIndices, $columnPointers);
    }

    @Override
    public void eachNonZero(MatrixProcedure procedure) {
        int k = 0;
        int j = 0;
        while (k < cardinality) {
            for (int i = columnPointers[j]; i < columnPointers[j + 1]; i++, k++) {
                procedure.apply(rowIndices[i], j, values[i]);
            }
            j++;
        }
    }

    @Override
    public void each(MatrixProcedure procedure) {
        int k = 0;
        for (int i = 0; i < rows; i++) {
            int valuesSoFar = columnPointers[i + 1];
            for (int j = 0; j < columns; j++) {
                if (k < valuesSoFar && j == rowIndices[k]) {
                    procedure.apply(i, j, values[k++]);
                } else {
                    procedure.apply(i, j, 0.0);
                }
            }
        }
    }

    @Override
    public void eachInColumn(int j, VectorProcedure procedure) {
        int k = columnPointers[j];
        int valuesSoFar = columnPointers[j + 1];
        for (int i = 0; i < rows; i++) {
            if (k < valuesSoFar && i == rowIndices[k]) {
                procedure.apply(i, values[k++]);
            } else {
                procedure.apply(i, 0.0);
            }
        }
    }

    @Override
    public void eachNonZeroInColumn(int j, VectorProcedure procedure) {
        for (int i = columnPointers[j]; i < columnPointers[j + 1]; i++) {
            procedure.apply(rowIndices[i], values[i]);
        }
    }

    @Override
    public void updateAt(int i, int j, MatrixFunction function) {
        int k = searchForRowIndex(i, columnPointers[j], columnPointers[j + 1]);
        if (k < columnPointers[j + 1] && rowIndices[k] == i) {
            double value = function.evaluate(i, j, values[k]);
            // if (Math.abs(value) < Matrices.EPS && value >= 0.0) {
            if (value == 0.0) {
                remove(k, j);
            } else {
                values[k] = value;
            }
        } else {
            insert(k, i, j, function.evaluate(i, j, 0.0));
        }
    }

    public boolean nonZeroAt(int i, int j) {
        int k = searchForRowIndex(i, columnPointers[j], columnPointers[j + 1]);
        return k < columnPointers[j + 1] && rowIndices[k] == i;
    }

    private int searchForRowIndex(int i, int left, int right) {
        if (right - left == 0 || i > rowIndices[right - 1]) {
            return right;
        }
        while (left < right) {
            int p = (left + right) / 2;
            if (rowIndices[p] > i) {
                right = p;
            } else if (rowIndices[p] < i) {
                left = p + 1;
            } else {
                return p;
            }
        }
        return left;
    }

    private void insert(int k, int i, int j, double value) {

        // if (Math.abs(value) < Matrices.EPS && value >= 0.0) {
        if (value == 0.0) {
            return;
        }
        if (values.length < cardinality + 1) {
            growUp();
        }
        if (cardinality - k > 0) {
            System.arraycopy(values, k, values, k + 1, cardinality - k);
            System.arraycopy(rowIndices, k, rowIndices, k + 1, cardinality - k);
        }
//        for (int k = cardinality; k > position; k--) {
//            values[k] = values[k - 1];
//            rowIndices[k] = rowIndices[k - 1];
//        }

        values[k] = value;
        rowIndices[k] = i;

        for (int jj = j + 1; jj < columns + 1; jj++) {
            columnPointers[jj]++;
        }

        cardinality++;
    }

    private void remove(int k, int j) {
        cardinality--;

        if (cardinality - k > 0) {
            System.arraycopy(values, k + 1, values, k, cardinality - k);
            System.arraycopy(rowIndices, k + 1, rowIndices, k, cardinality - k);
        }
//        for (int kk = k; kk < cardinality; kk++) {
//            values[kk] = values[kk + 1];
//            rowIndices[kk] = rowIndices[kk + 1];
//        }

        for (int jj = j + 1; jj < columns + 1; jj++) {
            columnPointers[jj]--;
        }
    }

    private void growUp() {

        if (values.length == capacity()) {
            // This should never happen
            throw new IllegalStateException("This matrix can't grow up.");
        }
        int min = (
            (rows != 0 && columns > Integer.MAX_VALUE / rows) ?
            Integer.MAX_VALUE :
            (rows * columns)
            );
        int capacity = Math.min(min, (cardinality * 3) / 2 + 1);

        double[] $values = new double[capacity];
        int[] $rowIndices = new int[capacity];

        System.arraycopy(values, 0, $values, 0, cardinality);
        System.arraycopy(rowIndices, 0, $rowIndices, 0, cardinality);

        values = $values;
        rowIndices = $rowIndices;
    }

    private int align(int cardinality) {
        return ((cardinality / MINIMUM_SIZE) + 1) * MINIMUM_SIZE;
    }

    @Override
    public double max() {
        double max = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < cardinality; i++) {
            if (values[i] > max) {
                max = values[i];
            }
        }
        return (max > 0.0) ? max : 0.0;
    }

    @Override
    public double min() {
        double min = Double.POSITIVE_INFINITY;

        for (int i = 0; i < cardinality; i++) {
            if (values[i] < min) {
                min = values[i];
            }
        }
        return (min < 0.0) ? min : 0.0;
    }

    @Override
    public double maxInColumn(int j) {
        double max = Double.NEGATIVE_INFINITY;

        for (int k = columnPointers[j]; k < columnPointers[j + 1]; k++) {
            if (values[k] > max) {
                max = values[k];
            }
        }
        return (max > 0.0) ? max : 0.0;
    }

    @Override
    public double minInColumn(int j) {
        double min = Double.POSITIVE_INFINITY;

        for (int k = columnPointers[j]; k < columnPointers[j + 1]; k++) {
            if (values[k] < min) {
                min = values[k];
            }
        }
        return (min < 0.0) ? min : 0.0;
    }

    /**
     * Returns a CCSMatrix with the selected rows and columns.
     */
    @Override
    public Matrix select(int[] rowIndices, int[] columnIndices) {
        int newRows = rowIndices.length;
        int newCols = columnIndices.length;

        if (newRows == 0 || newCols == 0) {
            fail("No rows or columns selected.");
        }

        // determine number of non-zero values (cardinality)
        // before allocating space, this is perhaps more efficient
        // than single pass and calling grow() when required.
        int newCardinality = 0;
        for (int row : rowIndices) {
            for (int column : columnIndices) {
                if (get(row, column) != 0.0) {
                    newCardinality++;
                }
            }
        }

        // Construct the raw structure for the sparse matrix
        double[] newValues = new double[newCardinality];
        int[] newRowIndices = new int[newCardinality];
        int[] newColumnPointers = new int[newCols + 1];

        newColumnPointers[0] = 0;
        int endPtr = 0;
        for (int j = 0; j < newCols; j++) {
            newColumnPointers[j + 1] = newColumnPointers[j];
            for (int i = 0; i < newRows; i++) {
                double val = get(rowIndices[i], columnIndices[j]);
                if (val != 0.0) {
                    newValues[endPtr] = val;
                    newRowIndices[endPtr] = i;
                    endPtr++;
                    newColumnPointers[j + 1]++;
                }
            }
        }
        return new CCSMatrix(newRows, newCols, newCardinality, newValues,
                             newRowIndices, newColumnPointers);
    }

    public Iterator<Integer> iteratorOrNonZeroColumns() {
        return new Iterator<Integer>() {
            private int j = -1;

            public boolean hasNext() {
                while (j + 1 < columns &&
                       columnPointers[j + 1] < cardinality &&
                       columnPointers[j + 1] == columnPointers[j + 2]) {
                    j++;
                }
                return j + 1 < columns && columnPointers[j + 1] < cardinality;
            }

            public Integer next() {
                if(!hasNext()) {
                    throw new NoSuchElementException();
                }
                j++;
                return j;
            }
        };
    }

    @Override
    public ColumnMajorMatrixIterator columnMajorIterator() {
        return new ColumnMajorMatrixIterator(rows, columns) {
            private long limit = (long) rows * columns;
            private boolean currentNonZero = false;
            private int i = -1;
            private int k = 0;

            public int rowIndex() {
                return i - columnIndex() * rows;
            }

            public int columnIndex() {
                return i / rows;
            }

            public double get() {
                return currentNonZero ? values[k] : 0.0;
            }

            public void set(double value) {
                if (currentNonZero) {
                    if (value == 0.0) {
                        CCSMatrix.this.remove(k, columnIndex());
                        currentNonZero = false;
                    } else {
                        values[k] = value;
                    }
                } else {
                    CCSMatrix.this.insert(k, rowIndex(), columnIndex(), value);
                    currentNonZero = true;
                }
            }

            public boolean hasNext() {
                return i + 1 < limit;
            }

            public Double next() {
                if(!hasNext()) {
                    throw new NoSuchElementException();
                }
                if (currentNonZero) {
                    k++;
                }
                i++;
                currentNonZero = k < columnPointers[columnIndex() + 1] && rowIndices[k] == rowIndex();
                return get();
            }
        };
    }

    @Override
    public ColumnMajorMatrixIterator nonZeroColumnMajorIterator() {
        return new ColumnMajorMatrixIterator(rows, columns) {
            private int j = 0;
            private int k = -1;
            private boolean currentIsRemoved = false;
            private int removedIndex = -1;

            public int rowIndex() {
                return currentIsRemoved ? removedIndex : rowIndices[k];
            }

            public int columnIndex() {
                return j;
            }

            public double get() {
                return currentIsRemoved ? 0.0 : values[k];
            }

            public void set(double value) {
                if (value == 0.0 && !currentIsRemoved) {
                    currentIsRemoved = true;
                    removedIndex = rowIndices[k];
                    CCSMatrix.this.remove(k--, j);
                } else if (value != 0.0 && !currentIsRemoved) {
                    values[k] = value;
                } else {
                    currentIsRemoved = false;
                    CCSMatrix.this.insert(++k, removedIndex, j, value);
                }
            }

            public boolean hasNext() {
                return k + 1 < cardinality;
            }

            public Double next() {
                if(!hasNext()) {
                    throw new NoSuchElementException();
                }
                currentIsRemoved = false;
                k++;
                while (columnPointers[j + 1] == k) {
                    j++;
                }
                return get();
            }
        };
    }

    @Override
    public VectorIterator nonZeroIteratorOfColumn(final int j) {
        return new VectorIterator(rows) {
            private int k = columnPointers[j] - 1;
            private boolean currentIsRemoved = false;
            private int removedIndex = -1;

            public int index() {
                return currentIsRemoved ? removedIndex : rowIndices[k];
            }

            public double get() {
                return currentIsRemoved ? 0.0 : values[k];
            }

            public void set(double value) {
                if (value == 0.0 && !currentIsRemoved) {
                    currentIsRemoved = true;
                    removedIndex = rowIndices[k];
                    CCSMatrix.this.remove(k--, j);
                } else if (value != 0.0 && !currentIsRemoved) {
                    values[k] = value;
                } else {
                    currentIsRemoved = false;
                    CCSMatrix.this.insert(++k, removedIndex, j, value);
                }
            }

            public boolean hasNext() {
                return k + 1 < columnPointers[j + 1];
            }

            public Double next() {
                if(!hasNext()) {
                    throw new NoSuchElementException();
                }
                currentIsRemoved = false;
                return values[++k];
            }
        };
    }

    @Override
    public VectorIterator iteratorOfColumn(final int j) {
        return new VectorIterator(rows) {
            private int i = -1;
            private int k = columnPointers[j];

            public int index() {
                return i;
            }

            public double get() {
                if (k < columnPointers[j + 1] && rowIndices[k] == i) {
                    return values[k];
                }
                return 0.0;
            }

            public void set(double value) {
                if (k < columnPointers[j + 1] && rowIndices[k] == i) {
                    if (value == 0.0) {
                        CCSMatrix.this.remove(k, j);
                    } else {
                        values[k] = value;
                    }
                } else {
                    CCSMatrix.this.insert(k, i, j, value);
                }
            }

            public boolean hasNext() {
                return i + 1 < rows;
            }

            public Double next() {
                if(!hasNext()) {
                    throw new NoSuchElementException();
                }
                i++;
                if (k < columnPointers[j + 1] && rowIndices[k] == i - 1) {
                    k++;
                }
                return get();
            }
        };
    }

    @Override
    public byte[] toBinary() {
        int size = 1 +                 // 1 byte: class tag
                   4 +                 // 4 bytes: rows
                   4 +                 // 4 bytes: columns
                   4 +                 // 4 bytes: cardinality
                  (8 * cardinality) +  // 8 * cardinality bytes: values
                  (4 * cardinality) +  // 4 * cardinality bytes: rowPointers
                  (4 * (columns + 1)); // 4 * (columns + 1) bytes: columnIndices

        ByteBuffer buffer = ByteBuffer.allocate(size);

        buffer.put(MatrixFactory.CCS_MATRIX_MATRIX_TAG);
        buffer.putInt(rows);
        buffer.putInt(columns);
        buffer.putInt(cardinality);

        for (int i = 0; i < cardinality; i++) {
            buffer.putInt(rowIndices[i]);
            buffer.putDouble(values[i]);
        }

        for (int i = 0; i < columns + 1; i++) {
            buffer.putInt(columnPointers[i]);
        }

        return buffer.array();
    }
}
