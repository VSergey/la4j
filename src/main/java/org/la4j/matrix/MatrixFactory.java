/*
 * Copyright 2011-2014, by Vladimir Kostyukov and Contributors.
 *
 * This file is part of the la4j project (http://la4j.org)
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
 * Contributor(s):
 */

package org.la4j.matrix;

import org.la4j.*;
import org.la4j.matrix.sparse.CCSMatrix;
import org.la4j.matrix.sparse.CRSMatrix;

import java.lang.reflect.ParameterizedType;
import java.util.StringTokenizer;

/**
 * An abstract matrix factory.
 *
 * @param <T>
 */
public abstract class MatrixFactory<T extends Matrix> {
    public static final byte BASIC_1_MATRIX_TAG = (byte) 0x00;
    public static final byte BASIC_2_MATRIX_TAG = (byte) 0x10;
    public static final byte CRS_MATRIX_MATRIX_TAG = (byte) 0x20;
    public static final byte CCS_MATRIX_MATRIX_TAG = (byte) 0x30;

    @SuppressWarnings("unchecked cast")
    protected final Class<T> outputClass = (Class<T>) ((ParameterizedType)
        getClass().getGenericSuperclass()).getActualTypeArguments()[0];

    public abstract T zero(int rows, int columns);

    /**
     * Converts given matrix using the this factory.
     *
     * @return converted matrix
     */
    public T convert(Matrix matrix) {
        T result = zero(matrix.rows(), matrix.columns());
        matrix.apply(LinearAlgebra.IN_PLACE_COPY_MATRIX_TO_MATRIX, result);
        return result;
    }

    /**
     * Creates a diagonal {@link Matrix} of the given {@code size} whose
     * diagonal elements are equal to {@code diagonal}.
     */
    public abstract T diagonal(int size, double diagonal);

    /**
     * Creates a new {@link Matrix} from the given 1D {@code array} with
     * compressing (copying) the underlying array.
     */
    public abstract T from1DArray(int rows, int columns, double[] array);

    /**
     * Creates a new {@link Matrix} from the given 2D {@code array} with
     * compressing (copying) the underlying array.
     */
    public abstract T from2DArray(double[][] array);

    /**
     * Creates a block {@link Matrix} of the given blocks {@code a},
     * {@code b}, {@code c} and {@code d}.
     */
    public abstract T block(Matrix a, Matrix b, Matrix c, Matrix d);

    /**
     * Decodes {@link Matrix} from the given byte {@code array}.
     *
     * @param array the byte array representing a matrix
     * @return a decoded matrix
     */
    public abstract T fromBinary(byte[] array);

    /**
     * Creates a zero {@link Matrix} of the given shape:
     * {@code rows} x {@code columns}.
     */
    public static Matrix zeroMatrix(int rows, int columns) {
        long size = (long) rows * columns;
        return size > 1000 ? Matrices.SPARSE.zero(rows, columns) : Matrices.DENSE.zero(rows, columns);
    }

    /**
     * Parses {@link Matrix} from the given CSV string.
     *
     * @param csv the string in CSV format
     *
     * @return a parsed matrix
     */
    public static Matrix fromCSV(String csv) {
        StringTokenizer lines = new StringTokenizer(csv, "\n");
        Matrix result = Matrices.DENSE.zero(10, 10);
        int rows = 0;
        int columns = 0;

        while (lines.hasMoreTokens()) {
            if (result.rows() == rows) {
                result = result.copyOfRows((rows * 3) / 2 + 1);
            }

            StringTokenizer elements = new StringTokenizer(lines.nextToken(), ", ");
            int j = 0;
            while (elements.hasMoreElements()) {
                if (j == result.columns()) {
                    result = result.copyOfColumns((j * 3) / 2 + 1);
                }

                double x = Double.parseDouble(elements.nextToken());
                result.set(rows, j++, x);
            }

            rows++;
            columns = j > columns ? j : columns;
        }

        return result.copyOfShape(rows, columns);
    }

    /**
     * Parses {@link Matrix} from the given Matrix Market string.
     *
     * @param mm the string in Matrix Market format
     *
     * @return a parsed matrix
     */
    public static Matrix fromMatrixMarket(String mm) {
        StringTokenizer body = new StringTokenizer(mm, "\n");

        String headerString = body.nextToken();
        StringTokenizer header = new StringTokenizer(headerString);

        if (!"%%MatrixMarket".equals(header.nextToken())) {
            throw new IllegalArgumentException("Wrong input file format: can not read header '%%MatrixMarket'.");
        }

        String object = header.nextToken();
        if (!"matrix".equals(object)) {
            throw new IllegalArgumentException("Unexpected object: " + object + ".");
        }

        String format = header.nextToken();
        if (!"coordinate".equals(format) && !"array".equals(format)) {
            throw new IllegalArgumentException("Unknown format: " + format + ".");
        }

        String field = header.nextToken();
        if (!"real".equals(field)) {
            throw new IllegalArgumentException("Unknown field type: " + field + ".");
        }

        String symmetry = header.nextToken();
        if (!symmetry.equals("general")) {
            throw new IllegalArgumentException("Unknown symmetry type: " + symmetry + ".");
        }

        String majority = (header.hasMoreTokens()) ? header.nextToken() : "row-major";

        String nextToken = body.nextToken();
        while (nextToken.startsWith("%")) {
            nextToken = body.nextToken();
        }

        if ("coordinate".equals(format)) {
            StringTokenizer lines = new StringTokenizer(nextToken);

            int rows = Integer.parseInt(lines.nextToken());
            int columns = Integer.parseInt(lines.nextToken());
            int cardinality = Integer.parseInt(lines.nextToken());

            Matrix result = "row-major".equals(majority) ?
                    CRSMatrix.zero(rows, columns, cardinality) :
                    CCSMatrix.zero(rows, columns, cardinality);

            for (int k = 0; k < cardinality; k++) {
                lines = new StringTokenizer(body.nextToken());

                int i = Integer.valueOf(lines.nextToken());
                int j = Integer.valueOf(lines.nextToken());
                double x = Double.valueOf(lines.nextToken());
                result.set(i - 1, j - 1, x);
            }

            return result;
        } else {
            StringTokenizer lines = new StringTokenizer(nextToken);

            int rows = Integer.valueOf(lines.nextToken());
            int columns = Integer.valueOf(lines.nextToken());
            Matrix result = Matrices.DENSE.zero(rows, columns);

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    result.set(i, j, Double.valueOf(body.nextToken()));
                }
            }

            return result;
        }
    }

}
