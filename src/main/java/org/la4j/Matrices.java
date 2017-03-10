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
 * Contributor(s): Yuriy Drozd
 *                 Ewald Grusk
 *                 Maxim Samoylov
 *                 Miron Aseev
 *                 Todd Brunhoff
 *
 */

package org.la4j;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.ByteBuffer;

import gnu.trove.list.array.TDoubleArrayList;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import org.la4j.matrix.MatrixFactory;
import org.la4j.matrix.dense.Basic1DMatrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.matrix.functor.AdvancedMatrixPredicate;
import org.la4j.matrix.functor.MatrixAccumulator;
import org.la4j.matrix.functor.MatrixFunction;
import org.la4j.matrix.functor.MatrixPredicate;
import org.la4j.matrix.functor.MatrixProcedure;
import org.la4j.matrix.sparse.CCSMatrix;
import org.la4j.matrix.sparse.CRSMatrix;

public final class Matrices {

    /**
     * The machine epsilon, that is calculated at runtime.
     */
    public static final double EPS = LinearAlgebra.EPS;

    /**
     * Exponent of machine epsilon
     */
    public static final int ROUND_FACTOR = LinearAlgebra.ROUND_FACTOR;

    /**
     * Checks whether the matrix is a
     * <a href="http://mathworld.wolfram.com/DiagonalMatrix.html">diagonal
     * matrix</a>.
     */
    public static final MatrixPredicate DIAGONAL_MATRIX = new MatrixPredicate() {
        public boolean test(int rows, int columns) {
            return rows == columns;
        }
        public boolean test(int i, int j, double value) {
            return (i == j) || Math.abs(value) < EPS;
        }
    };

    /**
     * Checks whether the matrix is an
     * <a href="http://mathworld.wolfram.com/IdentityMatrix.html">identity
     * matrix</a>.
     */
    public static final MatrixPredicate IDENTITY_MATRIX = new MatrixPredicate() {
        public boolean test(int rows, int columns) {
            return rows == columns;
        }
        public boolean test(int i, int j, double value) {
            return (i == j) ? Math.abs(1.0 - value) < EPS
                    : Math.abs(value) < EPS;
        }
    };

    /**
     * Checks whether the matrix is a
     * <a href="http://mathworld.wolfram.com/ZeroMatrix.html">zero
     * matrix</a>.
     */
    public static final MatrixPredicate ZERO_MATRIX = new MatrixPredicate() {
        public boolean test(int rows, int columns) {
            return true;
        }
        public boolean test(int i, int j, double value) {
            return Math.abs(value) < EPS;
        }
    };

    /**
     * Checks whether the matrix is a
     * <a href="http://mathworld.wolfram.com/TridiagonalMatrix.html">tridiagonal
     * matrix</a>.
     */
    public static final MatrixPredicate TRIDIAGONAL_MATRIX = new MatrixPredicate() {
        public boolean test(int rows, int columns) {
            return rows == columns;
        }
        public boolean test(int i, int j, double value) {
            return Math.abs(i - j) <= 1 || Math.abs(value) < EPS;
        }
    };

    /**
     * Checks whether the matrix is a
     * <a href="http://mathworld.wolfram.com/PositiveMatrix.html">positive
     * matrix</a>.
     */
    public static final MatrixPredicate POSITIVE_MATRIX = new MatrixPredicate() {
        public boolean test(int rows, int columns) {
            return true;
        }
        public boolean test(int i, int j, double value) {
            return value > 0.0;
        }
    };

    /**
     * Checks whether the matrix is a
     * <a href="http://mathworld.wolfram.com/NegativeMatrix.html">negative
     * matrix</a>.
     */
    public static final MatrixPredicate NEGATIVE_MATRIX = new MatrixPredicate() {
        public boolean test(int rows, int columns) {
            return true;
        }
        public boolean test(int i, int j, double value) {
            return value < 0.0;
        }
    };

    /**
     * Checks whether the matrix is a lower bi-diagonal matrix</a>.
     */
    public static final MatrixPredicate LOWER_BIDIAGONAL_MATRIX = new MatrixPredicate() {
        public boolean test(int rows, int columns) {
            return rows == columns;
        }
        public boolean test(int i, int j, double value) {
            return !((i == j) || (i == j + 1)) || Math.abs(value) < EPS;
        }
    };

    /**
     * Checks whether the matrix is an upper bidiagonal matrix.
     */
    public static final MatrixPredicate UPPER_BIDIAGONAL_MATRIX = new MatrixPredicate() {
        public boolean test(int rows, int columns) {
            return rows == columns;
        }
        public boolean test(int i, int j, double value) {
            return !((i == j) || (i == j - 1)) || Math.abs(value) < EPS;
        }
    };

    /**
     * Checks whether the matrix is a
     * <a href="http://mathworld.wolfram.com/LowerTriangularMatrix.html">lower
     * triangular matrix</a>.
     */
    public static final MatrixPredicate LOWER_TRIANGULAR_MATRIX = new MatrixPredicate() {
        public boolean test(int rows, int columns) {
            return rows == columns;
        }
        public boolean test(int i, int j, double value) {
            return (i <= j) || Math.abs(value) < EPS;
        }
    };

    /**
     * Checks whether the matrix is an
     * <a href="http://mathworld.wolfram.com/UpperTriangularMatrix.html">upper
     * triangular matrix</a>.
     */
    public static final MatrixPredicate UPPER_TRIANGULAR_MATRIX = new MatrixPredicate() {
        public boolean test(int rows, int columns) {
            return rows == columns;
        }
        public boolean test(int i, int j, double value) {
            return (i >= j) || Math.abs(value) < EPS;
        }
    };

    /**
     * Checks whether the matrix is a
     * <a href="http://mathworld.wolfram.com/SymmetricMatrix.html">symmetric
     * matrix</a>.
     */
    public static final AdvancedMatrixPredicate SYMMETRIC_MATRIX =
            new SymmetricMatrixPredicate();

    /**
     * Checks whether the matrix is a
     * <a href="http://en.wikipedia.org/wiki/Diagonally_dominant_matrix">diagonally dominant matrix</a>.
     */
    public static final AdvancedMatrixPredicate DIAGONALLY_DOMINANT_MATRIX =
            new DiagonallyDominantPredicate();

    /**
     * Checks whether the matrix is positive definite.
     */
    public static final AdvancedMatrixPredicate POSITIVE_DEFINITE_MATRIX =
            new PositiveDefiniteMatrixPredicate();

    /**
     * A matrix factory that produces zero {@link Basic2DMatrix}.
     */
    public static final MatrixFactory<Basic2DMatrix> BASIC_2D = new MatrixFactory<Basic2DMatrix>() {
        public Basic2DMatrix zero(int rows, int columns) {
            return new Basic2DMatrix(rows, columns);
        }
        @Override
        public Basic2DMatrix convert(Matrix matrix) {
            if (outputClass == matrix.getClass()) {
                return outputClass.cast(matrix);
            }
            return super.convert(matrix);
        }
        public Basic2DMatrix diagonal(int size, double diagonal) {
            double[][] array = new double[size][size];
            for (int i = 0; i < size; i++) {
                array[i][i] = diagonal;
            }
            return new Basic2DMatrix(array);
        }
        public Basic2DMatrix from1DArray(int rows, int columns, double[] array) {
            double[][] array2D = new double[rows][columns];

            for (int i = 0; i < rows; i++) {
                System.arraycopy(array, i * columns, array2D[i], 0, columns);
            }

            return new Basic2DMatrix(array2D);
        }
        public Basic2DMatrix from2DArray(double[][] array) {
            return new Basic2DMatrix(array);
        }
        public Basic2DMatrix block(Matrix a, Matrix b, Matrix c, Matrix d) {
            if ((a.rows() != b.rows()) || (a.columns() != c.columns()) ||
                    (c.rows() != d.rows()) || (b.columns() != d.columns())) {
                throw new IllegalArgumentException("Sides of blocks are incompatible!");
            }
            int rows = a.rows() + c.rows();
            int columns = a.columns() + b.columns();
            double[][] array = new double[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    if ((i < a.rows()) && (j < a.columns())) {
                        array[i][j] = a.get(i, j);
                    }
                    if ((i < a.rows()) && (j > a.columns())) {
                        array[i][j] = b.get(i, j);
                    }
                    if ((i > a.rows()) && (j < a.columns())) {
                        array[i][j] = c.get(i, j);
                    }
                    if ((i > a.rows()) && (j > a.columns())) {
                        array[i][j] = d.get(i, j);
                    }
                }
            }
            return new Basic2DMatrix(array);
        }
        public Basic2DMatrix fromBinary(byte[] array) {
            ByteBuffer buffer = ByteBuffer.wrap(array);
            if (buffer.get() != BASIC_2_MATRIX_TAG) {
                throw new IllegalArgumentException("Can not decode Basic2DMatrix from the given byte array.");
            }
            int rows = buffer.getInt();
            int columns = buffer.getInt();
            double[][] values = new double[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    values[i][j] = buffer.getDouble();
                }
            }
            return new Basic2DMatrix(values);
        }
    };

    /**
     * A matrix factory that produces zero {@link Basic1DMatrix}.
     */
    public static final MatrixFactory<Basic1DMatrix> BASIC_1D = new MatrixFactory<Basic1DMatrix>() {
        public Basic1DMatrix zero(int rows, int columns) {
            return new Basic1DMatrix(rows, columns);
        }
        @Override
        public Basic1DMatrix convert(Matrix matrix) {
            if (outputClass == matrix.getClass()) {
                return outputClass.cast(matrix);
            }
            return super.convert(matrix);
        }
        public Basic1DMatrix diagonal(int size, double diagonal) {
            double[] array = new double[size * size];
            for (int i = 0; i < size; i++) {
                array[i * size + i] = diagonal;
            }
            return new Basic1DMatrix(size, size, array);
        }
        public Basic1DMatrix from1DArray(int rows, int columns, double[] array) {
            return new Basic1DMatrix(rows, columns, array);
        }
        public Basic1DMatrix from2DArray(double[][] array) {
            int rows = array.length;
            int columns = array[0].length;
            double[] array1D = new double[rows * columns];
            int offset = 0;
            for (double[] arr : array) {
                System.arraycopy(arr, 0, array1D, offset, columns);
                offset += columns;
            }
            return new Basic1DMatrix(rows, columns, array1D);
        }
        public Basic1DMatrix block(Matrix a, Matrix b, Matrix c, Matrix d) {
            if ((a.rows() != b.rows()) || (a.columns() != c.columns()) ||
                    (c.rows() != d.rows()) || (b.columns() != d.columns())) {
                throw new IllegalArgumentException("Sides of blocks are incompatible!");
            }
            int rows = a.rows() + c.rows();
            int columns = a.columns() + b.columns();
            double[] array = new double[rows * columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    if ((i < a.rows()) && (j < a.columns())) {
                        array[i * rows + j] = a.get(i, j);
                    }
                    if ((i < a.rows()) && (j > a.columns())) {
                        array[i * rows + j] = b.get(i, j);
                    }
                    if ((i > a.rows()) && (j < a.columns())) {
                        array[i * rows + j] = c.get(i, j);
                    }
                    if ((i > a.rows()) && (j > a.columns())) {
                        array[i * rows + j] = d.get(i, j);
                    }
                }
            }
            return new Basic1DMatrix(rows, columns, array);
        }
        public Basic1DMatrix fromBinary(byte[] array) {
            ByteBuffer buffer = ByteBuffer.wrap(array);

            if (buffer.get() != BASIC_1_MATRIX_TAG) {
                throw new IllegalArgumentException("Can not decode Basic1DMatrix from the given byte array.");
            }

            int rows = buffer.getInt();
            int columns = buffer.getInt();
            int capacity = rows * columns;
            double[] values = new double[capacity];

            for (int i = 0; i < capacity; i++) {
                values[i] = buffer.getDouble();
            }

            return new Basic1DMatrix(rows, columns, values);
        }
    };

    /**
     * A matrix factory that produces zero {@link CCSMatrix}.
     */
    public static final MatrixFactory<CCSMatrix> CCS = new MatrixFactory<CCSMatrix>() {
        public CCSMatrix zero(int rows, int columns) {
            return new CCSMatrix(rows, columns);
        }
        @Override
        public CCSMatrix convert(Matrix matrix) {
            if (outputClass == matrix.getClass()) {
                return outputClass.cast(matrix);
            }
            return super.convert(matrix);
        }
        public CCSMatrix diagonal(int size, double diagonal) {
            double[] values = new double[size];
            int[] rowIndices = new int[size];
            int[] columnPointers = new int[size + 1];

            for (int i = 0; i < size; i++) {
                rowIndices[i] = i;
                columnPointers[i] = i;
                values[i] = diagonal;

            }
            columnPointers[size] = size;
            return new CCSMatrix(size, size, size, values, rowIndices, columnPointers);
        }
        public CCSMatrix from1DArray(int rows, int columns, double[] array) {
            CCSMatrix result = Matrices.CCS.zero(rows, columns);

            for (int j = 0; j < columns; j++) {
                for (int i = 0; i < rows; i++) {
                    int k = i * columns + j;
                    if (array[k] != 0.0) {
                        result.set(i, j, array[k]);
                    }
                }
            }

            return result;
        }
        public CCSMatrix from2DArray(double[][] array) {
            int rows = array.length;
            int columns = array[0].length;
            CCSMatrix result = Matrices.CCS.zero(rows, columns);

            for (int j = 0; j < columns; j++) {
                for (int i = 0; i < rows; i++) {
                    if (array[i][j] != 0.0) {
                        result.set(i, j, array[i][j]);
                    }
                }
            }

            return result;
        }
        public CCSMatrix block(Matrix a, Matrix b, Matrix c, Matrix d) {
            if ((a.rows() != b.rows()) || (a.columns() != c.columns()) ||
                    (c.rows() != d.rows()) || (b.columns() != d.columns())) {
                throw new IllegalArgumentException("Sides of blocks are incompatible!");
            }
            int rows = a.rows() + c.rows();
            int columns = a.columns() + b.columns();
            TDoubleArrayList values = new TDoubleArrayList();
            IntArrayList rowIndices = new IntArrayList();
            int[] columnPointers = new int[rows + 1];

            int k = 0;
            columnPointers[0] = 0;
            double current = 0;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    if ((i < a.rows()) && (j < a.columns())) {
                        current = a.get(i, j);
                    }
                    if ((i < a.rows()) && (j > a.columns())) {
                        current = b.get(i, j);
                    }
                    if ((i > a.rows()) && (j < a.columns())) {
                        current = c.get(i, j);
                    }
                    if ((i > a.rows()) && (j > a.columns())) {
                        current = d.get(i, j);
                    }
                    if (Math.abs(current) > Matrices.EPS) {
                        values.add(current);
                        rowIndices.add(j);
                        k++;
                    }
                }
                columnPointers[i + 1] = k;
            }
            double[] valuesArray = values.toArray(new double[values.size()]);
            int[] rowIndArray = rowIndices.toIntArray(new int[rowIndices.size()]);
            return new CCSMatrix(rows, columns, k, valuesArray, rowIndArray, columnPointers);
        }
        public CCSMatrix fromBinary(byte[] array) {
            ByteBuffer buffer = ByteBuffer.wrap(array);
            if (buffer.get() != CCS_MATRIX_MATRIX_TAG) {
                throw new IllegalArgumentException("Can not decode CCSMatrix from the given byte array.");
            }
            int rows = buffer.getInt();
            int columns = buffer.getInt();
            int cardinality = buffer.getInt();

            int[] rowIndices = new int[cardinality];
            double[] values = new double[cardinality];
            int[] columnsPointers = new int[columns + 1];
            for (int i = 0; i < cardinality; i++) {
                rowIndices[i] = buffer.getInt();
                values[i] = buffer.getDouble();
            }
            for (int i = 0; i < columns + 1; i++) {
                columnsPointers[i] = buffer.getInt();
            }
            return new CCSMatrix(rows, columns, cardinality, values, rowIndices, columnsPointers);
        }
    };

    /**
     * A matrix factory that produces zero {@link CRSMatrix}.
     */
    public static final MatrixFactory<CRSMatrix> CRS = new MatrixFactory<CRSMatrix>() {
        public CRSMatrix zero(int rows, int columns) {
            return new CRSMatrix(rows, columns);
        }
        @Override
        public CRSMatrix convert(Matrix matrix) {
            if (outputClass == matrix.getClass()) {
                return outputClass.cast(matrix);
            }
            return super.convert(matrix);
        }
        public CRSMatrix diagonal(int size, double diagonal) {
            double[] values = new double[size];
            int[] columnIndices = new int[size];
            int[] rowPointers = new int[size + 1];

            for (int i = 0; i < size; i++) {
                columnIndices[i] = i;
                rowPointers[i] = i;
                values[i] = diagonal;
            }
            rowPointers[size] = size;
            return new CRSMatrix(size, size, size, values, columnIndices, rowPointers);
        }
        public CRSMatrix from1DArray(int rows, int columns, double[] array) {
            CRSMatrix result = Matrices.CRS.zero(rows, columns);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    int k = i * columns + j;
                    if (array[k] != 0.0) {
                        result.set(i, j, array[k]);
                    }
                }
            }
            return result;
        }
        public CRSMatrix from2DArray(double[][] array) {
            int rows = array.length;
            int columns = array[0].length;
            CRSMatrix result = Matrices.CRS.zero(rows, columns);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    if (array[i][j] != 0.0) {
                        result.set(i, j, array[i][j]);
                    }
                }
            }
            return result;
        }
        public CRSMatrix block(Matrix a, Matrix b, Matrix c, Matrix d) {
            if ((a.rows() != b.rows()) || (a.columns() != c.columns()) ||
                    (c.rows() != d.rows()) || (b.columns() != d.columns())) {
                throw new IllegalArgumentException("Sides of blocks are incompatible!");
            }
            int rows = a.rows() + c.rows();
            int columns = a.columns() + b.columns();
            TDoubleArrayList values = new TDoubleArrayList();
            IntArrayList columnIndices = new IntArrayList();
            int[] rowPointers = new int[rows + 1];
            int k = 0;
            rowPointers[0] = 0;
            double current = 0;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    if ((i < a.rows()) && (j < a.columns())) {
                        current = a.get(i, j);
                    }
                    if ((i < a.rows()) && (j > a.columns())) {
                        current = b.get(i, j);
                    }
                    if ((i > a.rows()) && (j < a.columns())) {
                        current = c.get(i, j);
                    }
                    if ((i > a.rows()) && (j > a.columns())) {
                        current = d.get(i, j);
                    }
                    if (Math.abs(current) > Matrices.EPS) {
                        values.add(current);
                        columnIndices.add(j);
                        k++;
                    }
                }
                rowPointers[i + 1] = k;
            }
            double[] valuesArray = values.toArray(new double[values.size()]);
            int[] colIndArray = columnIndices.toIntArray(new int[columnIndices.size()]);
            return new CRSMatrix(rows, columns, k, valuesArray, colIndArray, rowPointers);
        }
        public CRSMatrix fromBinary(byte[] array) {
            ByteBuffer buffer = ByteBuffer.wrap(array);

            if (buffer.get() != CRS_MATRIX_MATRIX_TAG) {
                throw new IllegalArgumentException("Can not decode CRSMatrix from the given byte array.");
            }
            int rows = buffer.getInt();
            int columns = buffer.getInt();
            int cardinality = buffer.getInt();
            int[] columnIndices = new int[cardinality];
            double[] values = new double[cardinality];
            int[] rowPointers = new int[rows + 1];

            for (int i = 0; i < cardinality; i++) {
                columnIndices[i] = buffer.getInt();
                values[i] = buffer.getDouble();
            }
            for (int i = 0; i < rows + 1; i++) {
                rowPointers[i] = buffer.getInt();
            }
            return new CRSMatrix(rows, columns, cardinality, values, columnIndices, rowPointers);
        }
    };

    /**
     * A default matrix factory for dense matrices.
     */
    public static final MatrixFactory<Basic2DMatrix> DENSE = BASIC_2D;

    /**
     * A default factory for sparse matrices.
     */
    public static final MatrixFactory<CRSMatrix> SPARSE = CRS;

    /**
     * A default factory for sparse row-major matrices.
     */
    public static final MatrixFactory<CRSMatrix> SPARSE_ROW_MAJOR = CRS;

    /**
     * A default factory for sparse column-major matrices.
     */
    public static final MatrixFactory<CCSMatrix> SPARSE_COLUMN_MAJOR = CCS;

    /**
     * Increases each element of matrix by <code>1</code>.
     */
    public static final MatrixFunction INC_FUNCTION = (i, j, value) -> value + 1.0;

    /**
     * Decreases each element of matrix by <code>1</code>.
     */
    public static final MatrixFunction DEC_FUNCTION = (i, j, value) -> value - 1.0;

    /**
     * Inverts each element of matrix.
     */
    public static final MatrixFunction INV_FUNCTION = (i, j, value) -> -value;

    private Matrices() {}

    private static class SymmetricMatrixPredicate implements AdvancedMatrixPredicate {

        @Override
        public boolean test(Matrix matrix) {
            if (matrix.rows() != matrix.columns()) {
                return false;
            }
            for (int i = 0; i < matrix.rows(); i++) {
                for (int j = i + 1; j < matrix.columns(); j++) {
                    double a = matrix.get(i, j);
                    double b = matrix.get(j, i);
                    double diff = Math.abs(a - b);
                    if (diff / Math.max(Math.abs(a), Math.abs(b)) > EPS) {
                        return false;
                    }
                }
            }
            return true;
        }
    }

    private static class DiagonallyDominantPredicate implements AdvancedMatrixPredicate {

        public boolean test(Matrix matrix) {
            if (matrix.rows() != matrix.columns()) {
                return false;
            }
            for (int i = 0; i < matrix.rows(); i++) {
                double sum = 0;
                for (int j = 0; j < matrix.columns(); j++) {
                    if (i != j) {
                        sum += Math.abs(matrix.get(i, j));
                    }
                }
                if (sum > Math.abs(matrix.get(i, i)) - EPS) {
                    return false;
                }
            }
            return true;
        }
    }

    private static class PositiveDefiniteMatrixPredicate implements AdvancedMatrixPredicate {

        public boolean test(Matrix matrix) {
            if (matrix.rows() != matrix.columns()) {
                return false;
            }
            int size = matrix.columns();
            int currentSize = 1;

            while (currentSize <= size) {
                Matrix topLeftMatrix = matrix.sliceTopLeft(currentSize, currentSize);

                if (topLeftMatrix.determinant() < 0) {
                    return false;
                }

                currentSize++;
            }
            return true;
        }
    }

    /**
     * Creates a const function that evaluates it's argument to given {@code value}.
     *
     * @param arg a const value
     *
     * @return a closure object that does {@code _}
     */
    public static MatrixFunction asConstFunction(final double arg) {
        return (i, j, value) -> arg;
    }

    /**
     * Creates a plus function that adds given {@code value} to it's argument.
     *
     * @param arg a value to be added to function's argument
     *
     * @return a closure object that does {@code _ + _}
     */
    public static MatrixFunction asPlusFunction(final double arg) {
        return (i, j, value) -> value + arg;
    }

    /**
     * Creates a minus function that subtracts given {@code value} from it's argument.
     *
     * @param arg a value to be subtracted from function's argument
     *
     * @return a closure that does {@code _ - _}
     */
    public static MatrixFunction asMinusFunction(final double arg) {
        return (i, j, value) -> value - arg;
    }

    /**
     * Creates a mul function that multiplies given {@code value} by it's argument.
     *
     * @param arg a value to be multiplied by function's argument
     *
     * @return a closure that does {@code _ * _}
     */
    public static MatrixFunction asMulFunction(final double arg) {
        return (i, j, value) -> value * arg;
    }

    /**
     * Creates a div function that divides it's argument by given {@code value}.
     *
     * @param arg a divisor value
     *
     * @return a closure that does {@code _ / _}
     */
    public static MatrixFunction asDivFunction(final double arg) {
        return (i, j, value) -> value / arg;
    }

    /**
     * Creates a mod function that calculates the modulus of it's argument and given {@code value}.
     *
     * @param arg a divisor value
     *
     * @return a closure that does {@code _ % _}
     */
    public static MatrixFunction asModFunction(final double arg) {
        return (i, j, value) -> value % arg;
    }

    /**
     * Makes a minimum matrix accumulator that accumulates the minimum of matrix elements.
     *
     * @return a minimum vector accumulator
     */
    public static MatrixAccumulator mkMinAccumulator() {
        return new MatrixAccumulator() {
            private double result = Double.POSITIVE_INFINITY;

            public void update(int i, int j, double value) {
                result = Math.min(result, value);
            }

            public double accumulate() {
                double value = result;
                result = Double.POSITIVE_INFINITY;
                return value;
            }
        };
    }

    /**
     * Makes a maximum matrix accumulator that accumulates the maximum of matrix elements.
     *
     * @return a maximum vector accumulator
     */
    public static MatrixAccumulator mkMaxAccumulator() {
        return new MatrixAccumulator() {
            private double result = Double.NEGATIVE_INFINITY;

            public void update(int i, int j, double value) {
                result = Math.max(result, value);
            }

            public double accumulate() {
                double value = result;
                result = Double.NEGATIVE_INFINITY;
                return value;
            }
        };
    }
    
    /**
     * Makes an Euclidean norm accumulator that allows to use
     * {@link org.la4j.Matrix#fold(MatrixAccumulator)}
     * method for norm calculation.
     *
     * @return an Euclidean norm accumulator
     */
    public static MatrixAccumulator mkEuclideanNormAccumulator() {
        return new MatrixAccumulator() {
            private BigDecimal result = BigDecimal.valueOf(0.0);

            public void update(int i, int j, double value) {
                result = result.add(BigDecimal.valueOf(value * value));
            }

            public double accumulate() {
                double value = result.setScale(Matrices.ROUND_FACTOR, RoundingMode.CEILING).doubleValue();
                result = BigDecimal.valueOf(0.0);
                return Math.sqrt(value);
            }
        };
    }
    
    /**
     * Makes an Manhattan norm accumulator that allows to use
     * {@link org.la4j.Matrix#fold(MatrixAccumulator)}
     * method for norm calculation.
     *
     * @return a Manhattan norm accumulator
     */
    public static MatrixAccumulator mkManhattanNormAccumulator() {
        return new MatrixAccumulator() {
            private double result = 0.0;

            public void update(int i, int j, double value) {
                result += Math.abs(value);
            }

            public double accumulate() {
                double value = result;
                result = 0.0;
                return value;
            }
        };
    }
    
    /**
     * Makes an Infinity norm accumulator that allows to use
     * {@link org.la4j.Matrix#fold(MatrixAccumulator)}
     * method for norm calculation.
     *
     * @return an Infinity norm accumulator
     */
    public static MatrixAccumulator mkInfinityNormAccumulator() {
        return new MatrixAccumulator() {
          private double result = Double.NEGATIVE_INFINITY;
          
          public void update(int i, int j, double value) {
            result = Math.max(result, Math.abs(value));
          }
          
          public double accumulate() {
            double value = result;
            result = Double.NEGATIVE_INFINITY;
            return value;
          }
        };
    }

    /**
     * Creates a sum matrix accumulator that calculates the sum of all elements in the matrix.
     *
     * @param neutral the neutral value
     *
     * @return a sum accumulator
     */
    public static MatrixAccumulator asSumAccumulator(final double neutral) {
        return new MatrixAccumulator() {
            private BigDecimal result = BigDecimal.valueOf(neutral);

            public void update(int i, int j, double value) {
                result = result.add(BigDecimal.valueOf(value));
            }

            public double accumulate() {
                double value = result.setScale(Matrices.ROUND_FACTOR, RoundingMode.CEILING).doubleValue();
                result = BigDecimal.valueOf(neutral);
                return value;
            }
        };
    }

    /**
     * Creates a product matrix accumulator that calculates the product of all elements in the matrix.
     *
     * @param neutral the neutral value
     *
     * @return a product accumulator
     */
    public static MatrixAccumulator asProductAccumulator(final double neutral) {
        return new MatrixAccumulator() {
            private BigDecimal result = BigDecimal.valueOf(neutral);

            public void update(int i, int j, double value) {
                result = result.multiply(BigDecimal.valueOf(value));
            }

            public double accumulate() {
                double value = result.setScale(Matrices.ROUND_FACTOR, RoundingMode.CEILING).doubleValue();
                result = BigDecimal.valueOf(neutral);
                return value;
            }
        };
    }

    /**
     * Creates a sum function accumulator, that calculates the sum of all
     * elements in the matrix after applying given {@code function} to each of them.
     *
     * @param neutral the neutral value
     * @param function the matrix function
     *
     * @return a sum function accumulator
     */
    public static MatrixAccumulator asSumFunctionAccumulator(final double neutral, final MatrixFunction function) {
        return new MatrixAccumulator() {
            private final MatrixAccumulator sumAccumulator = Matrices.asSumAccumulator(neutral);

            public void update(int i, int j, double value) {
                sumAccumulator.update(i, j, function.evaluate(i, j, value));
            }

            public double accumulate() {
                return sumAccumulator.accumulate();
            }
        };
    }

    /**
     * Creates a product function accumulator, that calculates the product of
     * all elements in the matrix after applying given {@code function} to
     * each of them.
     *
     * @param neutral the neutral value
     * @param function the matrix function
     *
     * @return a product function accumulator
     */
    public static MatrixAccumulator asProductFunctionAccumulator(final double neutral, final MatrixFunction function) {
        return new MatrixAccumulator() {
            private final MatrixAccumulator productAccumulator = Matrices.asProductAccumulator(neutral);

            public void update(int i, int j, double value) {
                productAccumulator.update(i, j, function.evaluate(i, j, value));
            }

            public double accumulate() {
                return productAccumulator.accumulate();
            }
        };
    }

    /**
     * Creates an accumulator procedure that adapts a matrix accumulator for procedure
     * interface. This is useful for reusing a single accumulator for multiple fold operations
     * in multiple matrices.
     *
     * @param accumulator the matrix accumulator
     *
     * @return an accumulator procedure
     */
    public static MatrixProcedure asAccumulatorProcedure(final MatrixAccumulator accumulator) {
        return accumulator::update;
    }
}