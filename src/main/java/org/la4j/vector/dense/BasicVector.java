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
 * Contributor(s): -
 * 
 */

package org.la4j.vector.dense;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Random;

import org.la4j.Vector;
import org.la4j.Vectors;
import org.la4j.vector.DenseVector;
import org.la4j.vector.VectorFactory;

import static org.la4j.vector.VectorFactory.BASIC_VECTOR_TAG;

/**
 * A basic dense vector implementation using an array.
 * 
 * A dense data structure stores data in an underlying array. Even zero elements
 * take up memory space. If you want a data structure that will not have zero
 * elements take up memory space, try a sparse structure.
 * 
 * However, fetch/store operations on dense data structures only take O(1) time,
 * instead of the O(log n) time on sparse structures.
 * 
 * {@code BasicVector} stores the underlying data in a standard array.
 */
public class BasicVector extends DenseVector {

    private double[] self;

    public BasicVector() {
        this(0);
    }

    public BasicVector(int length) {
        this(new double[length]);
    }

    public BasicVector(double[] array) {
        super(array.length);
        this.self = array;
    }

    /**
     * Creates a constant {@link BasicVector} of the given {@code length} with
     * the given {@code value}.
     */
    public static BasicVector constant(int length, double value) {
        double[] array = new double[length];
        Arrays.fill(array, value);

        return new BasicVector(array);
    }

    /**
     * Creates an unit {@link BasicVector} of the given {@code length}.
     */
    public static BasicVector unit(int length) {
        return BasicVector.constant(length, 1.0);
    }

    /**
     * Creates a random {@link BasicVector} of the given {@code length} with
     * the given {@code Random}.
     */
    public static BasicVector random(int length, Random random) {
        double[] array = new double[length];
        for (int i = 0; i < length; i++) {
            array[i] = random.nextDouble();
        }

        return new BasicVector(array);
    }

    public VectorFactory factory() {
        return Vectors.BASIC;
    }

    public double get(int i) { return self[i]; }

    public void set(int i, double value) {
        self[i] = value;
    }

    @Override
    public void swapElements(int i, int j) {
        if (i != j) {
            double d = self[i];
            self[i] = self[j];
            self[j] = d;
        }
    }

    public BasicVector copy(int length) {
      ensureLengthIsCorrect(length);

      double[] $self = new double[length];
      System.arraycopy(self, 0, $self, 0, Math.min($self.length, self.length));

      return new BasicVector($self);
    }

    public double[] toArray() {
        double[] result = new double[length];
        System.arraycopy(self, 0, result, 0, length);
        return result;
    }

    public byte[] toBinary() {
        int size = 1 +          // 1 byte: class tag
                   4 +          // 4 bytes: length
                  (8 * length); // 8 * length bytes: values

        ByteBuffer buffer = ByteBuffer.allocate(size);

        buffer.put(BASIC_VECTOR_TAG);
        buffer.putInt(length);
        for (double value: self) {
            buffer.putDouble(value);
        }

        return buffer.array();
    }
}
