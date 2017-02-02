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

    @SuppressWarnings("unchecked cast")
    public final Class<T> outputClass = (Class<T>) ((ParameterizedType)
            getClass().getGenericSuperclass()).getActualTypeArguments()[0];

    /**
     * Create new instance with length size
     */
    public abstract T apply(int length);

    /**
     * Creates a zero {@link Vector} of the given {@code length}.
     */
    public static Vector zero(int length) {
        return length > 1000 ? SparseVector.zero(length) : DenseVector.zero(length);
    }

    /**
     * Creates a new {@link Vector} from the given {@code array} w/o
     * copying the underlying array.
     */
    public static Vector fromArray(double[] array) {
        return DenseVector.fromArray(array);
    }

    /**
     * Creates a random {@link Vector} of the given {@code length} with
     * the given {@code Random}.
     */
    public static Vector random(int length, Random random) {
        return DenseVector.random(length, random);
    }

    /**
     * Creates a constant {@link Vector} of the given {@code length} with
     * the given {@code value}.
     */
    public static Vector constant(int length, double value) {
        return DenseVector.constant(length, value);
    }

    /**
     * Creates an unit {@link Vector} of the given {@code length}.
     */
    public static Vector unit(int length) {
        return DenseVector.constant(length, 1.0);
    }

    /**
     * Creates new {@link org.la4j.vector.dense.BasicVector} from {@code list}
     */
    public static Vector fromCollection(Collection<? extends Number> list) {
        return DenseVector.fromCollection(list);
    }

    /**
     * Creates new {@link org.la4j.vector.SparseVector} from {@code list}
     */
    public static Vector fromMap(Map<Integer, ? extends Number> map, int length) {
        return SparseVector.fromMap(map, length);
    }

    public static Vector fromBinary(byte[] array) {
        switch(array[0]) {
            case BasicVector.VECTOR_TAG:
                return BasicVector.fromBinary(array);
            case CompressedVector.VECTOR_TAG:
                return CompressedVector.fromBinary(array);
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
        Vector result = DenseVector.zero(estimatedLength);

        int i = 0;
        while (tokenizer.hasMoreTokens()) {
            if (result.length() == i) {
                result = result.copyOfLength((i * 3) / 2 + 1);
            }

            double x = Double.parseDouble(tokenizer.nextToken());
            result.set(i++, x);
        }

        return result.copyOfLength(i);
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
            result = SparseVector.zero(length, cardinality);

            for (int k = 0; k < cardinality; k++) {
                int i = Integer.parseInt(body.nextToken());
                double x = Double.parseDouble(body.nextToken());
                result.set(i - 1, x);
            }
        } else {
            result = DenseVector.zero(length);

            for (int i = 0; i < length; i++) {
                result.set(i, Double.valueOf(body.nextToken()));
            }
        }
        return result;
    }

}
