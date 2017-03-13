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

package org.la4j.vector;

import org.la4j.Vector;
import org.la4j.Vectors;
import org.la4j.vector.dense.BasicVector;
import org.la4j.vector.sparse.CompressedVector;

import java.lang.reflect.ParameterizedType;
import java.util.*;

/**
 * An abstract vector factory.
 *
 * @param <T>
 */
public abstract class VectorFactory<T extends Vector> {
    public static final byte BASIC_VECTOR_TAG = (byte) 0x00;
    public static final byte COMPRESSED_VECTOR_TAG = (byte) 0x10;

    @SuppressWarnings("unchecked cast")
    protected final Class<T> outputClass = (Class<T>) ((ParameterizedType)
            getClass().getGenericSuperclass()).getActualTypeArguments()[0];

    /**
     * Create new vector instance with length size
     */
    public abstract T zero(int length);

    /**
     * Creates new vector from {@code list}
     */
    public abstract T fromCollection(Collection<? extends Number> list);

    /**
     * Creates new vector from {@code list}
     */
    public abstract T fromMap(Map<Integer, ? extends Number> map, int length);

    /**
     * Creates a new vector from the given {@code array} w/o
     * copying the underlying array.
     */
    public abstract T fromArray(double[] array);

    /**
     * Decodes vector from the given byte {@code array}.
     *
     * @param array the byte array representing a vector
     *
     * @return a decoded vector
     */
    public abstract T fromBinary(byte[] array);

    /**
     * Converts vector
     */
    public abstract T convert(Vector v);


    /**
     * Creates a zero {@link Vector} of the given {@code length}.
     */
    public static Vector anyZero(int length) {
        return length > 1000 ? Vectors.SPARSE.zero(length) : Vectors.DENSE.zero(length);
    }

    /**
     * Creates a random {@link Vector} of the given {@code length} with
     * the given {@code Random}.
     */
    public static BasicVector random(int length, Random random) {
        return BasicVector.random(length, random);
    }

    /**
     * Creates a constant {@link Vector} of the given {@code length} with
     * the given {@code value}.
     */
    public static BasicVector constant(int length, double value) {
        return BasicVector.constant(length, value);
    }

    public static Vector fromBinaryArray(byte[] array) {
        switch(array[0]) {
            case BASIC_VECTOR_TAG:
                return Vectors.BASIC.fromBinary(array);
            case COMPRESSED_VECTOR_TAG:
                return Vectors.COMPRESSED.fromBinary(array);
        }
        throw new IllegalArgumentException("Can't decode Vector from the given byte array. Unknown vector type");
    }

    /**
     * Parses {@link Vector} from the given CSV string.
     *
     * @param csv the CSV string representing a vector
     *
     * @return a parsed vector
     */
    public static Vector fromCSV(String csv) {
        StringTokenizer tokenizer = new StringTokenizer(csv, ", ");
        int estimatedLength = csv.length() / (5 + 2) + 1; // 5 symbols per element "0.000"
        // 2 symbols for delimiter ", "
        Vector result = Vectors.DENSE.zero(estimatedLength);

        int i = 0;
        while (tokenizer.hasMoreTokens()) {
            if (result.length() == i) {
                result = result.copy((i * 3) / 2 + 1);
            }

            double x = Double.parseDouble(tokenizer.nextToken());
            result.set(i++, x);
        }

        return result.copy(i);
    }

    /**
     * Parses {@link Vector} from the given Matrix Market string.
     *
     * @param mm the string in Matrix Market format
     *
     * @return a parsed vector
     */
    public static Vector fromMatrixMarket(String mm) {
        StringTokenizer body = new StringTokenizer(mm);

        if (!"%%MatrixMarket".equals(body.nextToken())) {
            throw new IllegalArgumentException("Wrong input file format: can not read header '%%MatrixMarket'.");
        }

        String object = body.nextToken();
        if (!"vector".equals(object)) {
            throw new IllegalArgumentException("Unexpected object: " + object + ".");
        }

        String format = body.nextToken();
        if (!"coordinate".equals(format) && !"array".equals(format)) {
            throw new IllegalArgumentException("Unknown format: " + format + ".");
        }

        String field = body.nextToken();
        if (!"real".equals(field)) {
            throw new IllegalArgumentException("Unknown field type: " + field + ".");
        }

        int length = Integer.parseInt(body.nextToken());
        Vector result;
        if ("coordinate".equals(format)) {
            int cardinality = Integer.parseInt(body.nextToken());
            result = CompressedVector.zero(length, cardinality);

            for (int k = 0; k < cardinality; k++) {
                int i = Integer.parseInt(body.nextToken());
                double x = Double.parseDouble(body.nextToken());
                result.set(i - 1, x);
            }
        } else {
            result = Vectors.DENSE.zero(length);

            for (int i = 0; i < length; i++) {
                result.set(i, Double.valueOf(body.nextToken()));
            }
        }
        return result;
    }
}
