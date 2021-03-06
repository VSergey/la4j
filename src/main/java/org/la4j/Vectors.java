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
 * Contributor(s): Maxim Samoylov
 *                 Miron Aseev
 * 
 */

package org.la4j;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.ByteBuffer;
import java.util.Collection;
import java.util.Map;

import org.la4j.iterator.VectorIterator;
import org.la4j.vector.*;
import org.la4j.vector.dense.BasicVector;
import org.la4j.vector.functor.VectorAccumulator;
import org.la4j.vector.functor.VectorFunction;
import org.la4j.vector.functor.VectorPredicate;
import org.la4j.vector.functor.VectorProcedure;
import org.la4j.vector.sparse.CompressedVector;

public final class Vectors {

    public static final double EPS = LinearAlgebra.EPS;
    public static final int ROUND_FACTOR = LinearAlgebra.ROUND_FACTOR;

    public static final VectorFactory<BasicVector> BASIC = new VectorFactory<BasicVector>() {
        public BasicVector zero(int length) {
            return new BasicVector(length);
        }
        public BasicVector fromCollection(Collection<? extends Number> list) {
            BasicVector vector = new BasicVector(list.size());
            int i = 0;
            for (Number x : list) {
                vector.set(i++, x.doubleValue());
            }
            return vector;
        }
        public BasicVector fromMap(Map<Integer, ? extends Number> map, int length) {
            SparseVector vector = Vectors.SPARSE.fromMap(map, length);
            return convert(vector);
        }
        public BasicVector fromArray(double[] array) {
            return new BasicVector(array);
        }
        public BasicVector fromBinary(byte[] array) {
            ByteBuffer buffer = ByteBuffer.wrap(array);
            if (buffer.get() != BASIC_VECTOR_TAG) {
                throw new IllegalArgumentException("Can not decode BasicVector from the given byte array.");
            }
            double[] values = new double[buffer.getInt()];
            for (int i = 0; i < values.length; i++) {
                values[i] = buffer.getDouble();
            }
            return new BasicVector(values);
        }
        public BasicVector convert(Vector v) {
            if (outputClass == v.getClass()) {
                return outputClass.cast(v);
            }
            VectorIterator it = v.iterator();
            BasicVector result = zero(v.length());

            while (it.hasNext()) {
                double x = it.next();
                int i = it.index();
                result.set(i, x);
            }
            return result;
        }
    };

    public static final VectorFactory<CompressedVector> COMPRESSED = new VectorFactory<CompressedVector>() {
        public CompressedVector zero(int length) {
            return new CompressedVector(length);
        }
        public CompressedVector fromCollection(Collection<? extends Number> list) {
            BasicVector vector = Vectors.BASIC.fromCollection(list);
            return convert(vector);
        }
        public CompressedVector fromMap(Map<Integer, ? extends Number> map, int length) {
            //TODO goto lambdas
            int cardinality = map.size();
            int[] indices = new int[cardinality];
            double[] values = new double[cardinality];
            int i = 0;
            for (Map.Entry<Integer, ? extends Number> entry : map.entrySet()) {
                int index = entry.getKey();
                if (index < 0 || index >= length) {
                    throw new IllegalArgumentException("Check your map: Index must be 0..n-1");
                }
                indices[i] = index;
                values[i] = entry.getValue().doubleValue();
                i++;
            }
            return new CompressedVector(length, cardinality, values, indices);
        }
        public CompressedVector fromArray(double[] array) {
            int length  = array.length;
            CompressedVector result = new CompressedVector(length);
            for (int i = 0; i < length; i++) {
                if (array[i] != 0.0) {
                    result.set(i, array[i]);
                }
            }
            return result;
        }
        public CompressedVector fromBinary(byte[] array) {
            ByteBuffer buffer = ByteBuffer.wrap(array);
            if (buffer.get() != COMPRESSED_VECTOR_TAG) {
                throw new IllegalArgumentException("Can not decode CompressedVector from the given byte array.");
            }
            int length = buffer.getInt();
            int cardinality = buffer.getInt();
            double[] values = new double[cardinality];
            int[] indices = new int[cardinality];
            for (int i = 0; i < cardinality; i++) {
                indices[i] = buffer.getInt();
                values[i] = buffer.getDouble();
            }
            return new CompressedVector(length, cardinality, values, indices);
        }
        public CompressedVector convert(Vector v) {
            if (outputClass == v.getClass()) {
                return outputClass.cast(v);
            }
            CompressedVector result = zero(v.length());
            VectorIterator it = (v instanceof SparseVector)? ((SparseVector)v).nonZeroIterator() : v.iterator();

            while (it.hasNext()) {
                double x = it.next();
                int i = it.index();
                result.set(i, x);
            }
            return result;
        }
    };

    public static final VectorFactory<BasicVector> DENSE = BASIC;

    public static final VectorFactory<CompressedVector> SPARSE = COMPRESSED;

    /**
     * Checks whether the vector is a
     * <a href="http://mathworld.wolfram.com/ZeroMatrix.html">zero
     * vector</a>.
     */
    public static final VectorPredicate ZERO_VECTOR = (i, value) -> Math.abs(value) < EPS;

    /**
     * Checks whether the vector is a
     * <a href="http://mathworld.wolfram.com/PositiveMatrix.html">positive
     * vector</a>.
     */
    public static final VectorPredicate POSITIVE_VECTOR = (i, value) -> value > 0.0;

    /**
     * Checks whether the vector is a
     * <a href="http://mathworld.wolfram.com/NegativeMatrix.html">negative
     * vector</a>.
     */
    public static final VectorPredicate NEGATIVE_VECTOR = (i, value) -> value < 0.0;

    /**
     * Increases each element of vector by <code>1</code>.
     */
    public static final VectorFunction INC_FUNCTION = (i, value) -> value + 1.0;

    /**
     * Decreases each element of vectors by <code>1</code>.
     */
    public static final VectorFunction DEC_FUNCTION = (i, value) -> value - 1.0;

    /**
     * Inverts each element of vector.
     */
    public static final VectorFunction INV_FUNCTION = (i, value) -> -value;

    private Vectors() {}

    /**
     * Creates a const function that evaluates it's argument to given {@code value}.
     *
     * @param arg a const value
     *
     * @return a closure object that does {@code _}
     */
    public static VectorFunction asConstFunction(final double arg) {
        return (i, value) -> arg;
    }

    /**
     * Creates a plus function that adds given {@code value} to it's argument.
     *
     * @param arg a value to be added to function's argument
     *
     * @return a closure object that does {@code _ + _}
     */
    public static VectorFunction asPlusFunction(final double arg) {
        return (i, value) -> value + arg;
    }

    /**
     * Creates a minus function that subtracts given {@code value} from it's argument.
     *
     * @param arg a value to be subtracted from function's argument
     *
     * @return a closure that does {@code _ - _}
     */
    public static VectorFunction asMinusFunction(final double arg) {
        return (i, value) -> value - arg;
    }

    /**
     * Creates a mul function that multiplies given {@code value} by it's argument.
     *
     * @param arg a value to be multiplied by function's argument
     *
     * @return a closure that does {@code _ * _}
     */
    public static VectorFunction asMulFunction(final double arg) {
        return (i, value) -> value * arg;
    }

    /**
     * Creates a div function that divides it's argument by given {@code value}.
     *
     * @param arg a divisor value
     *
     * @return a closure that does {@code _ / _}
     */
    public static VectorFunction asDivFunction(final double arg) {
        return (i, value) -> value / arg;
    }

    /**
     * Creates a mod function that calculates the modulus of it's argument and given {@code value}.
     *
     * @param arg a divisor value
     *
     * @return a closure that does {@code _ % _}
     */
    public static VectorFunction asModFunction(final double arg) {
        return (i, value) -> value % arg;
    }

    /**
     * Creates a sum vector accumulator that calculates the sum of all elements in the vector.
     *
     * @param neutral the neutral value
     *
     * @return a sum accumulator
     */
    public static VectorAccumulator asSumAccumulator(final double neutral) {
        return new VectorAccumulator() {
            private BigDecimal result = BigDecimal.valueOf(neutral);

            public void update(int i, double value) {
                result = result.add(BigDecimal.valueOf(value));
            }

            public double accumulate() {
                double value = result.setScale(Vectors.ROUND_FACTOR, RoundingMode.CEILING).doubleValue();
                result = BigDecimal.valueOf(neutral);
                return value;
            }
        };
    }

    /**
     * Creates a product vector accumulator that calculates the product of all elements in the vector.
     *
     * @param neutral the neutral value
     *
     * @return a product accumulator
     */
    public static VectorAccumulator asProductAccumulator(final double neutral) {
        return new VectorAccumulator() {
            private BigDecimal result = BigDecimal.valueOf(neutral);

            public void update(int i, double value) {
                result = result.multiply(BigDecimal.valueOf(value));
            }

            public double accumulate() {
                double value = result.setScale(Vectors.ROUND_FACTOR, RoundingMode.CEILING).doubleValue();
                result = BigDecimal.valueOf(neutral);
                return value;
            }
        };
    }

    /**
     * Makes a minimum vector accumulator that accumulates the minimum across vector elements.
     *
     * @return a minimum vector accumulator
     */
    public static VectorAccumulator mkMinAccumulator() {
        return new VectorAccumulator() {
            private double result = Double.POSITIVE_INFINITY;

            public void update(int i, double value) {
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
     * Makes a maximum vector accumulator that accumulates the maximum across vector elements.
     *
     * @return a maximum vector accumulator
     */
    public static VectorAccumulator mkMaxAccumulator() {
        return new VectorAccumulator() {
            private double result = Double.NEGATIVE_INFINITY;

            public void update(int i, double value) {
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
     * {@link org.la4j.Vector#fold(org.la4j.vector.functor.VectorAccumulator)} method for norm calculation.
     *
     * @return an Euclidean norm accumulator
     */
    public static VectorAccumulator mkEuclideanNormAccumulator() {
        return new VectorAccumulator() {
            private BigDecimal result = BigDecimal.valueOf(0.0);

            public void update(int i, double value) {
                result = result.add(BigDecimal.valueOf(value * value));
            }

            public double accumulate() {
                double value = result.setScale(Vectors.ROUND_FACTOR, RoundingMode.CEILING).doubleValue();
                result = BigDecimal.valueOf(0.0);
                return Math.sqrt(value);
            }
        };
    }

    /**
     * Makes a Manhattan norm accumulator that allows to use
     * {@link org.la4j.Vector#fold(org.la4j.vector.functor.VectorAccumulator)} method for norm calculation.
     *
     * @return a Manhattan norm accumulator
     */
    public static VectorAccumulator mkManhattanNormAccumulator() {
        return new VectorAccumulator() {
            private double result = 0.0;

            public void update(int i, double value) {
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
     * {@link org.la4j.Vector#fold(org.la4j.vector.functor.VectorAccumulator)} method for norm calculation.
     *
     * @return an Infinity norm accumulator
     */
    public static VectorAccumulator mkInfinityNormAccumulator() {
        return new VectorAccumulator() {
            private double result = Double.NEGATIVE_INFINITY;

            @Override
            public void update(int i, double value) {
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
     * Creates a sum function accumulator, that calculates the sum of all
     * elements in the vector after applying given {@code function} to each of them.
     *
     * @param neutral the neutral value
     * @param function the vector function
     *
     * @return a sum function accumulator
     */
    public static VectorAccumulator asSumFunctionAccumulator(final double neutral, final VectorFunction function) {
        return new VectorAccumulator() {
            private final VectorAccumulator sumAccumulator = Vectors.asSumAccumulator(neutral);

            public void update(int i, double value) {
                sumAccumulator.update(i, function.evaluate(i, value));
            }
            public double accumulate() {
                return sumAccumulator.accumulate();
            }
        };
    }

    /**
     * Creates a product function accumulator, that calculates the product of
     * all elements in the vector after applying given {@code function} to
     * each of them.
     *
     * @param neutral the neutral value
     * @param function the vector function
     *
     * @return a product function accumulator
     */
    public static VectorAccumulator asProductFunctionAccumulator(final double neutral, final VectorFunction function) {
        return new VectorAccumulator() {
            private final VectorAccumulator productAccumulator = Vectors.asProductAccumulator(neutral);

            public void update(int i, double value) {
                productAccumulator.update(i, function.evaluate(i, value));
            }
            public double accumulate() {
                return productAccumulator.accumulate();
            }
        };
    }

    /**
     * Creates an accumulator procedure that adapts a vector accumulator for procedure
     * interface. This is useful for reusing a single accumulator for multiple fold operations
     * in multiple vectors.
     *
     * @param accumulator the vector accumulator
     *
     * @return an accumulator procedure
     */
    public static VectorProcedure asAccumulatorProcedure(final VectorAccumulator accumulator) {
        return accumulator::update;
    }
}
