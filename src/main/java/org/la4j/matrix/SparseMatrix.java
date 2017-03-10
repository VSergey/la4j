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

import org.la4j.iterator.ColumnMajorMatrixIterator;
import org.la4j.iterator.MatrixIterator;
import org.la4j.iterator.RowMajorMatrixIterator;
import org.la4j.iterator.VectorIterator;
import org.la4j.Matrices;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.matrix.functor.MatrixAccumulator;
import org.la4j.matrix.functor.MatrixProcedure;
import org.la4j.Vector;
import org.la4j.Vectors;
import org.la4j.vector.functor.VectorAccumulator;
import org.la4j.vector.functor.VectorProcedure;

import java.text.NumberFormat;
import java.util.NoSuchElementException;

public abstract class SparseMatrix extends AbstractMatrix {

    protected int cardinality;

    public SparseMatrix(int rows, int columns) {
        this(rows, columns, 0);
    }

    public SparseMatrix(int rows, int columns, int cardinality) {
        super(rows, columns);
        this.cardinality = cardinality;
    }

    public MatrixFactory factory() {
        return Matrices.SPARSE;
    }

    public double get(int i, int j) {
        return getOrElse(i, j, 0.0);
    }

    /**
     * Gets the specified element, or a {@code defaultValue} if there
     * is no actual element at ({@code i}, {@code j}) in this sparse matrix.
     *
     * @param i the element's row index
     * @param j the element's column index
     * @param defaultValue the default value
     *
     * @return the element of this vector or a default value
     */
    public abstract double getOrElse(int i, int j, double defaultValue);

    /**
     * Checks whether or not this sparse matrix row-major.
     */
    public abstract boolean isRowMajor();

    public boolean isColumnMajor() {
        return !isRowMajor();
    }

    /**
     * Returns the cardinality (the number of non-zero elements)
     * of this sparse matrix.
     * 
     * @return the cardinality of this matrix
     */
    public int cardinality() {
        return cardinality;
    }

    /**
     * Returns the density (non-zero elements divided by total elements)
     * of this sparse matrix.
     * 
     * @return the density of this matrix
     */
    public double density() {
        return cardinality / (double) (rows * columns);
    }

    /**
     * @return a capacity of this sparse matrix
     */
    protected long capacity() {
        return ((long) rows) * columns;
    }

    public Vector getRow(int i) {
        Vector result = Vectors.SPARSE.zero(columns);
        VectorIterator it = nonZeroIteratorOfRow(i);

        while (it.hasNext()) {
            double x = it.next();
            int j = it.index();
            result.set(j, x);
        }

        return result;
    }

    public Vector getColumn(int j) {
        Vector result = Vectors.SPARSE.zero(rows);
        VectorIterator it = nonZeroIteratorOfColumn(j);

        while (it.hasNext()) {
            double x = it.next();
            int i = it.index();
            result.set(i, x);
        }

        return result;
    }

    @Override
    public Matrix multiply(double value) {
        MatrixIterator it = nonZeroIterator();
        Matrix result = blank();

        while (it.hasNext()) {
            double x = it.next();
            int i = it.rowIndex();
            int j = it.columnIndex();
            result.set(i, j, x * value);
        }

        return result;
    }

    @Override
    public Matrix add(double value) {
        MatrixIterator it = nonZeroIterator();
        Matrix result = Basic2DMatrix.constant(rows, columns, value);

        while (it.hasNext()) {
            double x = it.next();
            int i = it.rowIndex();
            int j = it.columnIndex();
            result.set(i, j, x + value);
        }

        return result;
    }

    /**
     * Whether or not the specified element is zero.
     *
     * @param i element's row index
     * @param j element's column index
     *
     * @return {@code true} if specified element is zero, {@code false} otherwise
     */
    public boolean isZeroAt(int i, int j) {
        return !nonZeroAt(i, j);
    }

    /**
     * Whether or not the specified element is not zero.
     *
     * @param i element's row index
     * @param j element's column index
     *
     * @return {@code true} if specified element is not zero, {@code false} otherwise
     */
    public abstract boolean nonZeroAt(int i, int j);

    /**
     * Applies given {@code procedure} to each non-zero element of this matrix.
     *
     * @param procedure the matrix procedure
     */
    public void eachNonZero(MatrixProcedure procedure) {
        MatrixIterator it = nonZeroIterator();

        while (it.hasNext()) {
            double x = it.next();
            int i = it.rowIndex();
            int j = it.columnIndex();
            procedure.apply(i, j, x);
        }
    }

    /**
     * Applies the given {@code procedure} to each non-zero element of the specified row of this matrix.
     * 
     * @param i the row index. 
     * @param procedure the {@link VectorProcedure}. 
     */
    public void eachNonZeroInRow(int i, VectorProcedure procedure) {
        VectorIterator it = nonZeroIteratorOfRow(i);

        while (it.hasNext()) {
            double x = it.next();
            int j = it.index();
            procedure.apply(j, x);
        }
    }

    /**
     * Applies the given {@code procedure} to each non-zero element of the specified column of this matrix.
     * 
     * @param j the column index.
     * @param procedure the {@link VectorProcedure}.
     */
    public void eachNonZeroInColumn(int j, VectorProcedure procedure) {
        VectorIterator it = nonZeroIteratorOfColumn(j);

        while (it.hasNext()) {
            double x = it.next();
            int i = it.index();
            procedure.apply(i, x);
        }
    }

    /**
     * Folds non-zero elements of this matrix with given {@code accumulator}.
     *
     * @param accumulator the matrix accumulator
     *
     * @return the accumulated value
     */
    public double foldNonZero(MatrixAccumulator accumulator) {
        eachNonZero(Matrices.asAccumulatorProcedure(accumulator));
        return accumulator.accumulate();
    }

    /**
     * Folds non-zero elements of the specified row in this matrix with the given {@code accumulator}.
     * 
     * @param i the row index.
     * @param accumulator the {@link VectorAccumulator}.
     * 
     * @return the accumulated value.
     */
    public double foldNonZeroInRow(int i, VectorAccumulator accumulator) {
        eachNonZeroInRow(i, Vectors.asAccumulatorProcedure(accumulator));
        return accumulator.accumulate();
    }

    /**
     * Folds non-zero elements of the specified column in this matrix with the given {@code accumulator}.
     * 
     * @param j the column index.
     * @param accumulator the {@link VectorAccumulator}.
     * 
     * @return the accumulated value.
     */
    public double foldNonZeroInColumn(int j, VectorAccumulator accumulator) {
        eachNonZeroInColumn(j, Vectors.asAccumulatorProcedure(accumulator));
        return accumulator.accumulate();
    }

    /**
     * Folds non-zero elements (in a column-by-column manner) of this matrix with given {@code accumulator}.
     *
     * @param accumulator the matrix accumulator
     *
     * @return the accumulated vector
     */
    public double[] foldNonZeroInColumns(VectorAccumulator accumulator) {
        double[] result = new double[columns];

        for (int j = 0; j < columns; j++) {
            result[j] = foldNonZeroInColumn(j, accumulator);
        }

        return result;
    }

    /**
     * Folds non-zero elements (in a row-by-row manner) of this matrix with given {@code accumulator}.
     *
     * @param accumulator the matrix accumulator
     *
     * @return the accumulated vector
     */
    public double[] foldNonZeroInRows(VectorAccumulator accumulator) {
        double[] result = new double[rows];

        for (int i = 0; i < rows; i++) {
            result[i] = foldNonZeroInRow(i, accumulator);
        }

        return result;
    }

    /**
     * Returns a non-zero matrix iterator.
     *
     * @return a non-zero matrix iterator
     */
    public MatrixIterator nonZeroIterator() {
        return nonZeroRowMajorIterator();
    }

    /**
     * Returns a non-zero row-major matrix iterator.
     *
     * @return a non-zero row-major matrix iterator.
     */
    public RowMajorMatrixIterator nonZeroRowMajorIterator() {
        return new RowMajorMatrixIterator(rows, columns) {
            private long limit = (long) rows * columns;
            private long i = -1;

            public int rowIndex() {
                return (int) (i / columns);
            }

            public int columnIndex() {
                return (int) (i - ((i / columns) * columns));
            }

            public double get() {
                return SparseMatrix.this.get(rowIndex(), columnIndex());
            }

            public void set(double value) {
                SparseMatrix.this.set(rowIndex(), columnIndex(), value);
            }

            public boolean hasNext() {
                while (i + 1 < limit) {
                    i++;
                    if (SparseMatrix.this.nonZeroAt(rowIndex(), columnIndex())) {
                        i--;
                        break;
                    }
                }
                return i + 1 < limit;
            }

            public Double next() {
                if(!hasNext()) {
                    throw new NoSuchElementException();
                }
                i++;
                return get();
            }
        };
    }

    /**
     * Returns a non-zero column-major matrix iterator.
     *
     * @return a non-zero column major matrix iterator.
     */
    public ColumnMajorMatrixIterator nonZeroColumnMajorIterator() {
        return new ColumnMajorMatrixIterator(rows, columns) {
            private long limit = (long) rows * columns;
            private long i = -1;

            public int rowIndex() {
                return (int) (i - ((i / rows) * rows));
            }

            public int columnIndex() {
                return (int) (i / rows);
            }

            public double get() {
                return SparseMatrix.this.get(rowIndex(), columnIndex());
            }

            public void set(double value) {
                SparseMatrix.this.set(rowIndex(), columnIndex(), value);
            }

            public boolean hasNext() {
                while (i + 1 < limit) {
                    i++;
                    if (SparseMatrix.this.nonZeroAt(rowIndex(), columnIndex())) {
                        i--;
                        break;
                    }
                }

                return i + 1 < limit;
            }

            public Double next() {
                if(!hasNext()) {
                    throw new NoSuchElementException();
                }
                i++;
                return get();
            }
        };
    }

    /**
     * Returns a non-zero vector iterator of the given row {@code i}.
     *
     * @return a non-zero vector iterator
     */
    public VectorIterator nonZeroIteratorOfRow(final int i) {
        return new VectorIterator(columns) {
            private int j = -1;

            public int index() {
                return j;
            }

            public double get() {
                return SparseMatrix.this.get(i, j);
            }

            public void set(double value) {
                SparseMatrix.this.set(i, j, value);
            }

            public boolean hasNext() {
                while (j + 1 < columns && SparseMatrix.this.isZeroAt(i, j + 1)) {
                    j++;
                }
                return j + 1 < columns;
            }

            public Double next() {
                if(!hasNext()) {
                    throw new NoSuchElementException();
                }
                j++;
                return get();
            }
        };
    }

    /**
     * Returns a non-zero vector iterator of the given column {@code j}.
     *
     * @return a non-zero vector iterator
     */
    public VectorIterator nonZeroIteratorOfColumn(final int j) {
        return new VectorIterator(rows) {
            private int i = -1;

            public int index() {
                return i;
            }

            public double get() {
                return SparseMatrix.this.get(i, j);
            }

            public void set(double value) {
                SparseMatrix.this.set(i, j, value);
            }

            public boolean hasNext() {
                while (i + 1 < rows && SparseMatrix.this.isZeroAt(i + 1, j)) {
                    i++;
                }
                return i + 1 < rows;
            }

            public Double next() {
                if(!hasNext()) {
                    throw new NoSuchElementException();
                }
                i++;
                return get();
            }
        };
    }

    public String toMatrixMarket(NumberFormat formatter) {
        String majority = isRowMajor() ? "row-major" : "column-major";
        StringBuilder out = new StringBuilder();
        MatrixIterator it = nonZeroIterator();

        out.append("%%MatrixMarket matrix coordinate real general ")
           .append(majority).append('\n');
        out.append(rows).append(' ').append(columns).append(' ')
           .append(cardinality).append('\n');
        while (it.hasNext()) {
            double x = it.next();
            int i = it.rowIndex();
            int j = it.columnIndex();
            out.append(i + 1).append(' ').append(j + 1).append(' ')
               .append(formatter.format(x)).append('\n');
        }

        return out.toString();
    }

    protected void ensureCardinalityIsCorrect(long rows, long columns, long cardinality) {
        if (cardinality < 0) {
            fail("Cardinality should be positive: " + cardinality + ".");
        }

        long capacity = capacity();

        if (cardinality > capacity) {
            fail("Cardinality should be less then or equal to capacity: " + capacity + ".");
        }
    }
}
