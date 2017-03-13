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
 * Contributor(s): Daniel Renshaw
 *                 Jakob Moellers
 *                 Maxim Samoylov
 *                 Miron Aseev
 *                 Ewald Grusk
 *
 */

package org.la4j;

import java.text.DecimalFormat;
import java.text.NumberFormat;

import org.la4j.iterator.VectorIterator;
import org.la4j.operation.VectorMatrixOperation;
import org.la4j.operation.VectorOperation;
import org.la4j.operation.VectorVectorOperation;
import org.la4j.vector.VectorFactory;
import org.la4j.vector.functor.*;

/**
 * A vector represents an array of elements. It can be re-sized.
 */
public interface Vector extends Iterable<Double> {
    NumberFormat DEFAULT_FORMATTER = new DecimalFormat("0.000");

    /**
     * Returns the length of this vector.
     *
     * @return length of this vector
     */
    int length();

    /**
     * Gets the specified element of this vector.
     * 
     * @param i element's index
     * @return the element of this vector
     */
    double get(int i);

    /**
     * Sets the specified element of this matrix to given {@code value}.
     *
     * @param i element's index
     * @param value element's new value
     */
    void set(int i, double value);

    /**
     * Sets all elements of this vector to given {@code value}.
     *
     * @param value the element's new value
     */
    void setAll(double value);

    /**
     * Swaps the specified elements of this vector.
     *
     * @param i element's index
     * @param j element's index
     */
    void swapElements(int i, int j);

    /**
     * Pipes this vector to a given {@code operation}.
     *
     * @param operation the vector operation
     *                  (an operation that take vector and returns {@code T})
     * @param <T> the result type
     *
     * @return the result of an operation applied to this vector
     */
    <T> T apply(VectorOperation<T> operation);

    /**
     * Pipes this vector to a given {@code operation}.
     *
     * @param operation the vector-vector operation
     *                  (an operation that takes two vectors and returns {@code T})
     * @param <T> the result type
     * @param that the right hand vector for the given operation
     *
     * @return the result of an operation applied to this and {@code that} vector
     */
    <T> T apply(VectorVectorOperation<T> operation, Vector that);

    /**
     * Pipes this vector to a given {@code operation}.
     *
     * @param operation the vector-matrix operation
     *                  (an operation that takes vector and matrix and returns {@code T})
     * @param <T> the result type
     * @param that the right hand matrix for the given operation
     *
     * @return the result of an operation applied to this vector and {@code that} matrix
     */
    <T> T apply(VectorMatrixOperation<T> operation, Matrix that);

    /**
     * Retrieves the specified sub-vector of this vector. The sub-vector is specified by
     * interval of indices.
     *
     * @param from the beginning of indices interval
     * @param until the ending of indices interval
     *
     * @return the sub-vector of this vector
     */
    Vector slice(int from, int until);

    /**
     * Retrieves the specified sub-vector of this vector. The sub-vector is specified by
     * interval of indices. The left point of interval is fixed to zero.
     *
     * @param until the ending of indices interval
     *
     * @return the sub-vector of this vector
     */
    default Vector sliceLeft(int until) {
        return slice(0, until);
    }

    /**
     * Retrieves the specified sub-vector of this vector. The sub-vector is specified by
     * interval of indices. The right point of interval is fixed to vector's length.
     *
     * @param from the beginning of indices interval
     *
     * @return the sub-vector of this vector
     */
    default Vector sliceRight(int from) {
        return slice(from, length());
    }

    /**
     * Returns a new vector with the selected elements.
     *
     * @param indices the array of indices
     *
     * @return the new vector with the selected elements
     */
    Vector select(int ... indices);

    /**
     * Builds a new vector by applying given {@code function} to each element
     * of this vector.
     *
     * @param function the vector function
     *
     * @return the transformed vector
     */
    Vector transform(VectorFunction function);

    /**
     * Updates all elements of this vector by applying given {@code function}.
     *
     * @param function the the vector function
     */
    void update(VectorFunction function);

    /**
     * Updates the specified element of this vector by applying given {@code function}.
     *
     * @param i element's index
     * @param function the vector function
     */
    default void updateAt(int i, VectorFunction function) {
        set(i, function.evaluate(i, get(i)));
    }

    /**
     * Folds all elements of this vector with given {@code accumulator}.
     *
     * @param accumulator the vector accumulator
     *
     * @return the accumulated value
     */
    default double fold(VectorAccumulator accumulator) {
        each(Vectors.asAccumulatorProcedure(accumulator));
        return accumulator.accumulate();
    }

    /**
     * Checks whether this vector compiles with given {@code predicate} or not.
     *
     * @param predicate the vector predicate
     *
     * @return whether this vector compiles with predicate
     */
    boolean is(VectorPredicate predicate);

    /**
     * Checks whether this vector compiles with given {@code predicate} or not.
     *
     * @param predicate the vector predicate
     *
     * @return whether this vector compiles with predicate
     */
    default boolean non(VectorPredicate predicate) {
        return !is(predicate);
    }

    /**
     * Returns a vector iterator.
     *
     * @return a vector iterator.
     */
    @Override
    VectorIterator iterator();

    /**
     * Applies given {@code procedure} to each element of this vector.
     *
     * @param procedure the vector procedure
     */
    void each(VectorProcedure procedure);

    /**
     * Calculates an Euclidean norm of this vector.
     *
     * @return an Euclidean norm
     */
    default double norm() {
        return euclideanNorm();
    }

    /**
     * Calculates an Euclidean norm of this vector.
     *
     * @return an Euclidean norm
     */
    default double euclideanNorm() {
        return fold(Vectors.mkEuclideanNormAccumulator());
    }

    /**
     * Calculates a Manhattan norm of this vector.
     *
     * @return a Manhattan norm
     */
    default double manhattanNorm() {
        return fold(Vectors.mkManhattanNormAccumulator());
    }

    /**
     * Calculates an Infinity norm of this vector.
     *
     * @return an Infinity norm
     */
    default double infinityNorm() {
        return fold(Vectors.mkInfinityNormAccumulator());
    }

    /**
     * Searches for the maximum value of the elements of this vector.
     *
     * @return the maximum value of this vector
     */
    default double max() {
        return fold(Vectors.mkMaxAccumulator());
    }

    /**
     * Searches for the minimum value of the elements of this vector.
     *
     * @return the minimum value of this vector
     */
    default double min() {
        return fold(Vectors.mkMinAccumulator());
    }

    /**
     * Adds given {@code value} (v) to this vector (X).
     *
     * @param value the right hand value for addition
     *
     * @return X + v
     */
    Vector add(double value);

    /**
     * Adds given {@code vector} (X) to this vector (Y).
     *
     * @param that the right hand vector for addition
     *
     * @return X + Y
     */
    default Vector add(Vector that) {
        return apply(LinearAlgebra.OO_PLACE_VECTORS_ADDITION, that);
    }

    /**
     * Multiplies this vector (X) by given {@code value} (v).
     *
     * @param value the right hand value for multiplication
     *
     * @return X * v
     */
    Vector multiply(double value);

    /**
     * Shuffles this vector.
     *
     * <p>
     * Copies this vector in the new vector that contains the same elements but with
     * the elements shuffled around (which might also result in the same vector
     * (all outcomes are equally probable)).
     * </p>
     *
     * @return the shuffled vector
     */
    Vector shuffle();

    /**
     * Calculates the Hadamard (element-wise) product of this vector and given {@code that}.
     *
     * @param that the right hand vector for Hadamard product
     *
     * @return the Hadamard product of two vectors
     */
    default Vector hadamardProduct(Vector that) {
        return apply(LinearAlgebra.OO_PLACE_VECTOR_HADAMARD_PRODUCT, that);
    }

    /**
     * Multiples this vector (X) by given {@code that} (A).
     *
     * @param that the right hand matrix for multiplication
     *
     * @return X * A
     */
    default Vector multiply(Matrix that) {
        return apply(LinearAlgebra.OO_PLACE_VECTOR_BY_MATRIX_MULTIPLICATION, that);
    }

    /**
     * Subtracts given {@code value} (v) from this vector (X).
     *
     * @param value the right hand value for subtraction
     *
     * @return X - v
     */
    default Vector subtract(double value) {
        return add(-value);
    }

    /**
     * Subtracts given {@code that} (Y) from this vector (X).
     *
     * @param that the right hand vector for subtraction
     *
     * @return X - Y
     */
    default Vector subtract(Vector that) {
        return apply(LinearAlgebra.OO_PLACE_VECTORS_SUBTRACTION, that);
    }

    /**
     * Divides this vector (X) by given {@code value} (v).
     *
     * @param value the right hand value for division
     *
     * @return X / v
     */
    default Vector divide(double value) {
        return multiply(1.0 / value);
    }

    /**
     * Calculates the inner product of this vector and given {@code that}.
     *
     * @param that the right hand vector for inner product
     *
     * @return the inner product of two vectors
     */
    default double innerProduct(Vector that) {
        return apply(LinearAlgebra.OO_PLACE_INNER_PRODUCT, that);
    }

    /**
     * Calculates the outer product of this vector and given {@code that}.
     *
     * @param that the the right hand vector for outer product
     *
     * @return the outer product of two vectors
     */
    default Matrix outerProduct(Vector that) {
        return apply(LinearAlgebra.OO_PLACE_OUTER_PRODUCT, that);
    }

    /**
     * Calculates the cosine similarity between this vector and given {@code that}.
     *
     * @param that the vector to calculated cosine similarity with
     *
     * @return the cosine similarity of the two vectors
     */
    default double cosineSimilarity(Vector that) {
        return this.innerProduct(that) / (this.euclideanNorm() * that.euclideanNorm());
    }

    /**
     * Multiplies up all elements of this vector.
     *
     * @return product of all elements of this vector
     */
    default double product() {
        return fold(Vectors.asProductAccumulator(1.0));
    }

    /**
     * Summarizes all elements of the vector
     *
     * @return sum of all elements of the vector
     */
    default double sum() {
        return fold(Vectors.asSumAccumulator(0.0));
    }

    /**
     * Returns true when vector is equal to given {@code that} vector with given
     * {@code precision}.
     *
     * @param that vector
     * @param precision given precision
     *
     * @return equals of this matrix to that
     */
    boolean equals(Vector that, double precision);


    VectorFactory factory();

    /**
     * Encodes this vector into a byte array.
     *
     * @return a byte array representing this vector
     */
    byte[] toBinary();

    /**
     * Creates a blank (an empty vector with same length) copy of this vector.
     *
     * @return blank vector
     */
    default Vector blank() {
        return blank(length());
    }

    /**
     * Creates a blank (an empty vector) copy of this vector with the given
     * {@code length}.
     *
     * @param length the length of the blank vector
     *
     * @return blank vector
     */
    default Vector blank(int length) {
        return factory().zero(length);
    }

    /**
     * Copies this vector.
     *
     * @return the copy of this vector
     */
    default Vector copy() {
        return copy(length());
    }

    /**
     * Copies this vector into the new vector with specified {@code length}.
     *
     * @param length the length of new vector
     *
     * @return the copy of this vector with new length
     */
    Vector copy(int length);

    /**
     * Converts this vector to matrix with only one row.
     *
     * @return the row matrix
     */
    Matrix toRowMatrix();

    /**
     * Converts this vector to matrix with only one column.
     *
     * @return the column matrix
     */
    Matrix toColumnMatrix();

    /**
     * Converts this vector to a diagonal matrix.
     *
     * @return a diagonal matrix
     */
    Matrix toDiagonalMatrix();

    /**
     * Converts this vector into the CSV (Comma Separated Value) string.
     *
     * @return a CSV string representing this vector
     */
    default String toCSV() {
        return toCSV(DEFAULT_FORMATTER);
    }

    /**
     * Converts this vector into the CSV (Comma Separated Value) string
     * using the given {@code formatter}.
     *
     * @return a CSV string representing this vector
     */
    String toCSV(NumberFormat formatter);

    /**
     * Converts this vector into the string in Matrix Market format.
     *
     * @return a Matrix Market string representing this vector
     */
    default String toMatrixMarket() {
        return toMatrixMarket(DEFAULT_FORMATTER);
    }

    /**
     * Converts this vector into the string in Matrix Market format
     * using the given {@code formatter};
     *
     * @param formatter the number formater
     *
     * @return a Matrix Market string representing this vector
     */
    String toMatrixMarket(NumberFormat formatter);


}
