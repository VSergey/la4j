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
*                 Yuriy Drozd
*                 Maxim Samoylov
*
*/

package org.la4j.vector;

import org.junit.Assert;
import org.junit.Test;
import org.la4j.V;
import org.la4j.Vector;
import org.la4j.Vectors;
import org.la4j.Matrix;
import org.la4j.vector.functor.*;

import static org.la4j.V.*;
import static org.la4j.M.*;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public abstract class VectorTest<T extends Vector> {

    private VectorFactory<T> factory;

    public VectorTest(VectorFactory<T> factory) {
        this.factory = factory;
    }

    public T v(double... values) {
        Vector vector = V.v(values);
        return factory.convert(vector);
    }

    public T vz(int length) {
        Vector vector = V.vz(length);
        return factory.convert(vector);
    }

    @Test
    public void testAccess_10() {
        Vector a = v(0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.1, 2.0);

        Assert.assertEquals(10, a.length());
        Assert.assertEquals(0.0, a.get(0), Vectors.EPS);
        Assert.assertEquals(2.0, a.get(2), Vectors.EPS);
        Assert.assertEquals(2.0, a.get(9), Vectors.EPS);

        a.set(3, 42.0);
        a.set(8, 14.0);
        a.set(0, 3.3);

        Assert.assertEquals(42.0, a.get(3), Vectors.EPS);
        Assert.assertEquals(14.0, a.get(8), Vectors.EPS);
        Assert.assertEquals(3.3, a.get(0), Vectors.EPS);
    }

    @Test
    public void testSetAll_5() {
        Vector a = v(0.0, 0.0, 0.0, 0.0, 1.0, 4.0);
        a.setAll(11.1);

        for (double d : a) {
            Assert.assertEquals(11.1, d, Vectors.EPS);
        }
    }

    @Test
    public void testCopyOfLength_3_to_5_to_2() {
        Vector a = v(7.0, 0.0, 1.0);
        Vector b = v(7.0, 0.0, 1.0, 0.0, 0.0);
        Vector c = v(7.0, 0.0);

        a = a.copy(5);
        Assert.assertEquals(b, a);

        a = a.copy(2);
        Assert.assertEquals(c, a);
    }

    @Test
    public void testCopyOf_5_to_0_to_4() {
        Vector a = v(0.0, 1.0, 2.0, 3.0, 0.0);
        Vector b = v();
        Vector c = v(0.0, 0.0, 0.0, 0.0);

        a = a.copy(0);
        Assert.assertEquals(b, a);

        a = a.copy(4);
        Assert.assertEquals(c, a);
    }

    @Test
    public void testSlice_5_to_2_and_3() {
        Vector a = v(1.0, 2.0, 3.0, 4.0, 5.0);
        Vector b = v(2.0, 3.0);
        Vector c = v(3.0, 4.0, 5.0);

        Assert.assertEquals(b, a.slice(1, 3));
        Assert.assertEquals(c, a.slice(2, 5));
    }

    @Test
    public void testSliceLeftRight_5_to_1_and_4() {
        Vector a = v(9.0, 2.0, 0.0, 4.0, 0.0);
        Vector b = v(9.0);
        Vector c = v(2.0, 0.0, 4.0, 0.0);

        Assert.assertEquals(b, a.sliceLeft(1));
        Assert.assertEquals(c, a.sliceRight(1));
    }

    @Test
    public void testSliceLeftRight_5_to_2_and_3() {
        Vector a = v(9.0, 2.0, 0.0, 4.0, 0.0);
        Vector b = v(9.0, 2.0);
        Vector c = v(0.0, 4.0, 0.0);

        Assert.assertEquals(b, a.sliceLeft(2));
        Assert.assertEquals(c, a.sliceRight(2));
    }

    @Test
    public void testSelect_4() {
        Vector a = v(0.0, 3.0, 7.0, 0.0);
        Vector b = v(3.0, 0.0, 0.0, 7.0);
        Vector c = v(7.0, 7.0, 0.0, 0.0);

        Assert.assertEquals(b, a.select(1, 0, 3, 2));
        Assert.assertEquals(c, a.select(2, 2, 0, 3));
    }

    @Test
    public void testSelect_5() {
        Vector a = v(1.0, 6.0, 0.0, 0.0, 8.0);
        Vector b = v(1.0, 1.0, 1.0);
        Vector c = v(0.0, 0.0, 8.0, 8.0, 1.0, 0.0);

        Assert.assertEquals(b, a.select(0, 0, 0));
        Assert.assertEquals(c, a.select(2, 3, 4, 4, 0, 3));
    }

    @Test
    public void testSelect_3() {
        Vector a = v(1.0, 0.0, 0.0);
        Vector b = v(0.0, 0.0, 0.0, 0.0);
        Vector c = v(1.0);

        Assert.assertEquals(b, a.select(1, 2, 2, 1));
        Assert.assertEquals(c, a.select(0));
    }

    @Test
    public void testSwapElements_5() {
        Vector a = v(1.0, 0.0, 0.0, 0.0, 3.0);
        Vector b = v(3.0, 0.0, 0.0, 0.0, 1.0);

        a.swapElements(0, 4);
        Assert.assertEquals(b, a);
    }

    @Test
    public void testSwapElements_4() {
        Vector a = v(0.0, 1.0, 0.0, 0.0);
        Vector b = v(0.0, 0.0, 1.0, 0.0);

        a.swapElements(1, 2);
        Assert.assertEquals(b, a);
    }

    @Test
    public void testSwapElements_4_2() {
        Vector a = v(0.0, 1.0, 0.0, 2.0);
        Vector b = v(0.0, 0.0, 1.0, 2.0);

        a.swapElements(1, 2);
        Assert.assertEquals(a, b);
    }

    @Test
    public void testSwapElements_6() {
        Vector a = v(0.0, 0.0, 0.0, 0.0, 0.0, -5.0);
        Vector b = v(0.0, 0.0, 0.0, -5.0, 0.0, 0.0);

        a.swapElements(3, 5);
        Assert.assertEquals(a, b);
    }

    @Test
    public void testSwapElements_2() {
        Vector a = v(1.0, 2.0);
        Vector b = v(2.0, 1.0);

        a.swapElements(0, 1);
        Assert.assertEquals(b, a);
    }

    @Test
    public void testEuclideanNorm_3() {
        Vector a = v(1.0, 2.0, 3.0);
        Assert.assertEquals(3.74165, a.fold(Vectors.mkEuclideanNormAccumulator()), 1e-5);
        Assert.assertEquals(3.74165, a.euclideanNorm(), 1e-5);
    }

    @Test
    public void testEuclideanNorm_5() {
        Vector a = v(1.0, 0.0, 3.0, 0.0, -5.0);
        Assert.assertEquals(5.91607, a.fold(Vectors.mkEuclideanNormAccumulator()), 1e-5);
        Assert.assertEquals(5.91607, a.euclideanNorm(), 1e-5);
    }

    @Test
    public void testManhattanNorm_3() {
        Vector a = v(1.0, 2.0, 3.0);
        Assert.assertEquals(6.0, a.fold(Vectors.mkManhattanNormAccumulator()), 1e-5);
        Assert.assertEquals(6.0, a.manhattanNorm(), 1e-5);
    }

    @Test
    public void testManhattanNorm_5() {
        Vector a = v(1.0, 0.0, 3.0, 0.0, -5.0);
        Assert.assertEquals(9.0, a.fold(Vectors.mkManhattanNormAccumulator()), 1e-5);
        Assert.assertEquals(9.0, a.manhattanNorm(), 1e-5);
    }

    @Test
    public void testInfinityNorm_3() {
        Vector a = v(1.0, 2.0, 3.0);
        Assert.assertEquals(3.0, a.fold(Vectors.mkInfinityNormAccumulator()), 1e-5);
        Assert.assertEquals(3.0, a.infinityNorm(), 1e-5);
    }

    @Test
    public void testInfinityNorm_5() {
        Vector a = v(1.0, 0.0, 3.0, 0.0, -5.0);
        Assert.assertEquals(5.0, a.fold(Vectors.mkInfinityNormAccumulator()), 1e-5);
        Assert.assertEquals(5.0, a.infinityNorm(), 1e-5);
    }

    @Test
    public void testAdd_3() {
        for (Vector b : vs(0.0, 5.0, 0.0)) {
            Vector a = v(0.0, 0.0, 3.0);
            Vector c = v(7.0, 7.0, 10.0);
            Vector d = v(0.0, 5.0, 3.0);

            Assert.assertEquals(c, a.add(7.0));
            Assert.assertEquals(d, a.add(b));
            Assert.assertEquals(d, b.add(a));
        }
    }

    @Test
    public void testAdd_4() {
        for (Vector b : vs(0.0, 1.0, 0.0, 6.0)) {
            Vector a = v(1.0, 0.0, 0.0, 3.0);
            Vector c = v(1.0, 1.0, 0.0, 9.0);

            Assert.assertEquals(c, a.add(b));
            Assert.assertEquals(c, b.add(a));
        }
    }

    @Test
    public void testSubtract_3() {
        for (Vector b : vs(4.0, 0.0, 0.0)) {
            Vector a = v(0.0, 0.0, 3.0);
            Vector c = v(-7.0, -7.0, -4.0);
            Vector d = v(-4.0, 0.0, 3.0);

            Assert.assertEquals(c, a.subtract(7.0));
            Assert.assertEquals(d, a.subtract(b));
        }
    }

    @Test
    public void testSubtract_4() {
        for (Vector b : vs(4.0, -2.0, -1.0)) {
            Vector a = v(0.0, 0.0, 3.0);
            Vector c = v(3.0, 3.0, 6.0);
            Vector d = v(-4.0, 2.0, 4.0);

            Assert.assertEquals(c, a.subtract(-3.0));
            Assert.assertEquals(d, a.subtract(b));
        }
    }

    @Test
    public void testMultiply_3() {
        for (Vector b : vs(0.0, 5.0, 0.0)) {
            Vector a = v(0.0, 0.0, 1.0);
            Vector c = v(0.0, 0.0, 10.0);
            Vector d = v(0.0, 0.0, 0.0);

            Assert.assertEquals(c, a.multiply(10.0));
            Assert.assertEquals(d, a.hadamardProduct(b));
        }
    }

    @Test
    public void testHadamardProduct_3() {
        for (Vector b : vs(3.0, 0.0, 0.0)) {
            Vector a = v(1.0, 0.0, 2.0);
            Vector c = v(3.0, 0.0, 0.0);

            Assert.assertEquals(c, a.hadamardProduct(b));
        }
    }

    @Test
    public void testMultiply_2_2x4() {
        for (Matrix b : ms(a(0.0, 5.0, 0.0, 6.0),
                a(1.0, 0.0, 8.0, 0.0))) {

            Vector a = v(1.0, 2.0);
            Vector c = v(2.0, 5.0, 16.0, 6.0);

            Assert.assertEquals(c, a.multiply(b));
        }
    }

    @Test
    public void testMultiply_3_3x1() {
        for (Matrix b : ms(a(0.0),
                a(3.0),
                a(0.0))) {

            Vector a = v(0.0, 2.0, 0.0);
            Vector c = v(6.0);

            Assert.assertEquals(c, a.multiply(b));
        }
    }

    @Test
    public void testProduct_3() {
        Vector a = v(2.0, 4.0, 6.0);
        Assert.assertEquals(a.product(), 48.0, Vectors.EPS);
    }

    @Test
    public void testSum_3() {
        Vector a = v(2.0, 4.0, 6.0);
        Assert.assertEquals(a.sum(), 12.0, Vectors.EPS);
    }

    @Test
    public void testDivide_3() {
        Vector a = v(0.0, 0.0, 3.0);
        Vector b = v(0.0, 0.0, 0.3);

        Assert.assertEquals(b, a.divide(10));
    }

    @Test
    public void testCopy_5() {
        Vector a = v(0.0, 0.0, 0.0, 0.0, 1.0);
        Assert.assertEquals(a, a.copy());
    }

    @Test
    public void testBlank_5() {
        Vector a = v(0.0, 0.0, 0.0, 0.0, 1.0);
        Vector b = v(0.0, 0.0, 0.0, 0.0, 0.0);

        Assert.assertEquals(b, a.blank());
    }

    @Test
    public void testInnerProduct_1() {
        for (Vector b : vs(10.0)) {
            Vector a = v(18.0);
            Assert.assertEquals(180.0, a.innerProduct(b), Vectors.EPS);
        }
    }

    @Test
    public void testInnerProduct_3() {
        for (Vector b : vs(10.0, 0.0, 10.0)) {
            Vector a = v(1.0, 2.0, 3.0);
            Assert.assertEquals(40.0, a.innerProduct(b), Vectors.EPS);
        }
    }

    @Test
    public void testInnerProduct_4() {
        for (Vector b : vs(11.0, 13.0, 17.0, 19.0)) {
            Vector a = v(2.0, 3.0, 5.0, 7.0);
            Assert.assertEquals(279.0, a.innerProduct(b), Vectors.EPS);
        }
    }

    @Test
    public void testOuterProduct_3_4() {
        for (Vector b : vs(7.0, 11.0, 13.0, 17.0)) {
            Vector a = v(2.0, 3.0, 5.0);

            Matrix c = m(a(14.0, 22.0, 26.0, 34.0),
                    a(21.0, 33.0, 39.0, 51.0),
                    a(35.0, 55.0, 65.0, 85.0));

            Assert.assertEquals(c, a.outerProduct(b));
        }
    }

    @Test
    public void testOuterProduct_1_2() {
        for (Vector b : vs(24.0, 1.0)) {
            Vector a = v(2.0);
            Matrix c = m(a(48.0, 2.0));

            Assert.assertEquals(c, a.outerProduct(b));
        }
    }

    @Test
    public void testOuterProduct_4_2() {
        for (Vector b : vs(4.0, -10.0)) {
            Vector a = v(2.0, 0.0, -1.0, 41.0);

            Matrix c = m(a(8.0, -20.0),
                    a(0.0, 0.0),
                    a(-4.0, 10.0),
                    a(164.0, -410.0));

            Assert.assertEquals(c, a.outerProduct(b));
        }
    }

    /**
     * Tests whether two vectors contain the same elements
     *
     * @param vector1 Vector1
     * @param vector2 Vector2
     * @return True if both vectors contain the same elements
     */
    public boolean testWhetherVectorsContainSameElements(Vector vector1, Vector vector2) {
        if (vector1.length() == vector2.length()) {
            boolean[] checkList = new boolean[vector1.length()];

            for (int i = 0; i < vector1.length(); i++) {
                for (int j = 0; j < vector2.length(); j++) {
                    if (vector1.get(i) == vector2.get(j)) {
                        if (!checkList[j]) {
                            checkList[j] = true;
                            break;
                        }
                    }
                }
            }

            boolean result = true;
            for (boolean b : checkList) {
                if (!b) {
                    result = false;
                }
            }
            return result;
        } else {
            return false;
        }
    }

    @Test
    public void testTestWhetherVectorsContainSameElements() {
        Vector a = v(1.0, 1.0, 3.0, 4.0);
        Vector b = v(4.0, 1.0, 1.0, 3.0);
        Vector c = v(4.0, 2.0, 1.0, 3.0);

        Assert.assertTrue(testWhetherVectorsContainSameElements(a, b));
        Assert.assertFalse(testWhetherVectorsContainSameElements(a, c));
    }

    @Test
    public void testShuffle() {
        Vector a = v(1.0, 1.0, 3.0, 4.0);
        Vector b = a.shuffle();

        Assert.assertTrue(testWhetherVectorsContainSameElements(a, b));
    }

    @Test
    public void testMax() {
        Vector a = v(1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -5.0, 0.0, 0.0, 5.0);
        Assert.assertEquals(5.0, a.max(), Vectors.EPS);
    }

    @Test
    public void testMaxCompressed() {
        Vector a = v(0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -5.0, 0.0, 0.0);
        Assert.assertEquals(0.0, a.max(), Vectors.EPS);
    }

    @Test
    public void testMin() {
        Vector a = v(1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -5.0, 0.0, 0.0, 5.0);
        Assert.assertEquals(-5.0, a.min(), Vectors.EPS);
    }

    @Test
    public void testMinCompressed() {
        Vector a = v(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0);
        Assert.assertEquals(0.0, a.min(), Vectors.EPS);
    }

    @Test
    public void testFold_6() {
        Vector a = v(0.0, 0.0, 5.0, 0.0, 2.0, 1.0);

        VectorAccumulator sum = Vectors.asSumAccumulator(0.0);
        VectorAccumulator product = Vectors.asProductAccumulator(1.0);

        Assert.assertEquals(8.0, a.fold(sum), Vectors.EPS);
        // check whether the accumulator were flushed
        Assert.assertEquals(8.0, a.fold(sum), Vectors.EPS);

        Assert.assertEquals(0.0, a.fold(product), Vectors.EPS);
        // check whether the accumulator were flushed
        Assert.assertEquals(0.0, a.fold(product), Vectors.EPS);
    }

    @Test
    public void testIssue162_0() {
        VectorPredicate pi = (i, value) -> value == 3.14;

        Vector a = v();
        Vector b = a.copy(31);

        Assert.assertEquals(0, a.length());
        Assert.assertEquals(31, b.length());

        b.setAll(3.14);
        Assert.assertTrue(b.is(pi));

        Vector c = b.copy(42);
        c.setAll(3.14);
        Assert.assertTrue(c.is(pi));

        Vector d = c.copy(54);
        d.setAll(3.14);
        Assert.assertTrue(d.is(pi));
    }

    @Test
    public void testResize_32_to_110_to_1076_to_31() {
        VectorPredicate fortyTwo = (i, value) -> value == 42.0;

        Vector a = v();
        Vector b = a.copy(32);

        Assert.assertEquals(32, b.length());

        b.setAll(42.0);
        Assert.assertTrue(b.is(fortyTwo));

        Vector c = b.copy(110);
        c.setAll(42.0);
        Assert.assertTrue(c.is(fortyTwo));

        Vector d = c.copy(1076);
        d.setAll(42.0);
        Assert.assertTrue(d.is(fortyTwo));

        Vector e = d.copy(31);
        e.setAll(42.0);
        Assert.assertTrue(e.is(fortyTwo));
    }

    @Test
    public void testNormalize_Default() {
        Vector a = v(3.0, 0.0, -4.0);
        Vector b = a.divide(a.norm());

        Assert.assertEquals(3, b.length());
        Assert.assertEquals(0.6, b.get(0), Vectors.EPS);
        Assert.assertEquals(0.0, b.get(1), Vectors.EPS);
        Assert.assertEquals(-0.8, b.get(2), Vectors.EPS);
        // Verify b is a unit vector
        // The default normalize() uses Euclidean as the accumulator
        Assert.assertEquals(1.0, b.norm(), Vectors.EPS);
    }

    @Test
    public void testNormalize_EuclideanNormAccumulator() {
        Vector a = v(3.0, 0.0, -4.0);
        Vector b = a.divide(a.euclideanNorm());

        Assert.assertEquals(3, b.length());
        Assert.assertEquals(0.6, b.get(0), Vectors.EPS);
        Assert.assertEquals(0.0, b.get(1), Vectors.EPS);
        Assert.assertEquals(-0.8, b.get(2), Vectors.EPS);
        // Verify b is a unit vector
        Assert.assertEquals(1.0, b.euclideanNorm(), Vectors.EPS);
    }

    @Test
    public void testNormalize_ManhattanNormAccumulator() {
        Vector a = v(3.0, 0.0, -4.0);
        Vector b = a.divide(a.manhattanNorm());

        Assert.assertEquals(3, b.length());
        Assert.assertEquals(0.42857, b.get(0), 0.00001);
        Assert.assertEquals(0.0, b.get(1), Vectors.EPS);
        Assert.assertEquals(-0.57142, b.get(2), 0.00001);
        // Verify b is a unit vector
        Assert.assertEquals(1.0, b.manhattanNorm(), Vectors.EPS);
    }

    @Test
    public void testNormalize_InfinityNormAccumulator() {
        Vector a = v(3.0, 0.0, -4.0);
        Vector b = a.divide(a.infinityNorm());

        Assert.assertEquals(3, b.length());
        Assert.assertEquals(0.75, b.get(0), Vectors.EPS);
        Assert.assertEquals(0.0, b.get(1), Vectors.EPS);
        Assert.assertEquals(-1.0, b.get(2), Vectors.EPS);
        // Verify b is a unit vector
        Assert.assertEquals(1.0, b.infinityNorm(), Vectors.EPS);
    }

    @Test
    public void testEqualsWithPrecision() throws Exception {
        Vector a = v(1000.0, 1.0);
        Assert.assertTrue(a.equals(a, Vectors.EPS));

        Vector b = v().copy(1000);
        b.setAll(42.0);
        Assert.assertFalse(a.equals(b, Vectors.EPS));
        Assert.assertFalse(b.equals(a, Vectors.EPS));

        Vector e = v();
        Assert.assertTrue(e.equals(e, Vectors.EPS));

        Vector f = v(Double.MIN_VALUE, Double.MIN_VALUE);
        Vector g = vz(2);
        Assert.assertTrue(f.equals(g, Vectors.EPS));
        Assert.assertTrue(g.equals(f, Vectors.EPS));

        Vector i = v(Double.MIN_NORMAL, Double.MIN_NORMAL);
        Assert.assertTrue(i.equals(g, Vectors.EPS));
        Assert.assertTrue(i.equals(f, Vectors.EPS));
        Assert.assertTrue(g.equals(i, Vectors.EPS));
        Assert.assertTrue(f.equals(i, Vectors.EPS));
    }

    @Test
    public void testEquals() throws Exception {
        Vector a = v().copy(1000);
        a.setAll(1.0);
        Assert.assertTrue(a.equals(a));
        Assert.assertTrue(a.copy().equals(a));

        Vector d = v().copy(1000);
        Assert.assertFalse(d.equals(a));
        Assert.assertTrue(d.equals(d.copy()));

        Vector e = v();
        Assert.assertTrue(e.equals(e.copy()));

        Vector f = v(Double.MIN_VALUE);
        Vector g = v(0.0);
        Assert.assertTrue(f.equals(g));
        Assert.assertTrue(g.equals(f));

        Vector i = v(Double.MIN_NORMAL);
        Assert.assertTrue(i.equals(g));
        Assert.assertTrue(i.equals(f));
        Assert.assertTrue(g.equals(i));
        Assert.assertTrue(f.equals(i));
    }

    @Test
    public void testFromCollection_empty() {
        List<Number> values = new LinkedList<>();
        Assert.assertEquals(Vectors.BASIC.fromCollection(values), Vectors.BASIC.zero(0));
    }

    @Test
    public void testFromCollection_normal_x3() {
        List<Double> values = Arrays.asList(1.0, 2.0, 3.0);
        Vector v = Vectors.BASIC.fromCollection(values);
        Assert.assertEquals(v, Vectors.BASIC.fromArray(new double[]{1.0, 2.0, 3.0}));
    }

    @Test
    public void testFromCollection_byte() {
        List<Byte> values = Arrays.asList((byte) 1, (byte) 3, (byte) 5, (byte) 6);
        Vector v = Vectors.BASIC.fromCollection(values);
        Assert.assertEquals(v, Vectors.BASIC.fromArray(new double[]{1.0, 3.0, 5.0, 6.0}));
    }

    @Test(expected = NullPointerException.class)
    public void testFromCollection_NPE() {
        Vector v = Vectors.BASIC.fromCollection(null);
    }

    @Test
    public void testFromMap_empty() {
        Map<Integer, Double> map = new HashMap<>();
        Assert.assertEquals(Vectors.SPARSE.fromMap(map, 0), Vectors.SPARSE.zero(0));
    }

    @Test
    public void testFromMap_normal() {
        Map<Integer, Double> map = new HashMap<>();
        map.put(0, 1.0);
        map.put(3, 2.0);
        map.put(5, 1.0);
        Vector v = Vectors.SPARSE.fromArray(new double[]{1, 0, 0, 2, 0, 1, 0});
        Assert.assertEquals(v, Vectors.SPARSE.fromMap(map, 7));
    }

    @Test
    public void testFromMap_emptyMap() {
        Map<Integer, Double> map = new HashMap<>();
        Assert.assertEquals(Vectors.SPARSE.fromMap(map, 5), Vectors.SPARSE.zero(5));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFromMap_invalidMap() {
        Map<Integer, Double> map = new HashMap<>();
        map.put(0, 1.0);
        map.put(3, 2.0);
        map.put(-2, 1.0);
        Vector v = Vectors.SPARSE.fromMap(map, 5);
    }

    @Test(expected = NullPointerException.class)
    public void testFromMap_NPE() {
        Vector v = Vectors.SPARSE.fromMap(null, 4);
    }

    @Test
    public void testCosineSimilarity() {
        Vector a = v(5, 1, 0, 0, 0, 1, 10, 15);
        Vector b = v(1, 8, 0, 9, 6, 4, 2, 5);
        Vector c = v(9, 0, 2, 1, 1, 0, 8, 12);
        Vector d = v(900, 0, 200, 100, 100, 0, 800, 1200);

        // a & c are more similar to each other than b
        Assert.assertTrue(a.cosineSimilarity(b) < a.cosineSimilarity(c));
        Assert.assertTrue(c.cosineSimilarity(b) < c.cosineSimilarity(a));

        Assert.assertEquals(1.0, c.cosineSimilarity(d), 0.00005);
        Assert.assertEquals(1.0, d.cosineSimilarity(c), 0.00005);
    }

    @Test
    public void testToCSV() {
        Vector a = v(-5, 1.1, 0, 0, 0, 1.12345, 100000, 15.15);
        Vector b = v(1, 8, 0, 9, 6, 4, 2, 5);
        Vector c = v(900, 0.0, -200, 100.123, -100, 0, 800, -1200);

        Assert.assertEquals("-5.000, 1.100, 0.000, 0.000, 0.000, 1.123, 100000.000, 15.150", a.toCSV());
        Assert.assertEquals("1.000, 8.000, 0.000, 9.000, 6.000, 4.000, 2.000, 5.000", b.toCSV());
        Assert.assertEquals("900.000, 0.000, -200.000, 100.123, -100.000, 0.000, 800.000, -1200.000", c.toCSV());

        Assert.assertEquals("-5, 1.1, 0, 0, 0, 1.12345, 100000, 15.15", a.toCSV(new DecimalFormat("0.#####")));
        Assert.assertEquals("1., 8., 0., 9., 6., 4., 2., 5.", b.toCSV(new DecimalFormat("0.")));
        Assert.assertEquals("900, 0, -200, 100.1, -100, 0, 800, -1200", c.toCSV(new DecimalFormat("0.#")));
    }

    @Test
    public void testMutation() {
        Vector a = v(900, 0.0, -200, 100.123, -100, 0, 800, -1200);

        Assert.assertEquals(a, Vectors.SPARSE.convert(a));
        Assert.assertEquals(a, Vectors.DENSE.convert(a));

        final double sum[] = new double[1];
        a.shuffle().each((i, d) -> sum[0]+=d);
        Assert.assertEquals(300.123, sum[0], 0.0001);

        Vector b = a.transform((i, d) -> d+1);
        a.update((i, d) -> d+1);
        Assert.assertEquals(a, b);
    }

    @Test
    public void testSerialization() {
        Vector a = v(900, 0.0, -200, 100.123, -100, 0, 800, -1200);
        byte[] bytes = a.toBinary();

        Vector b = VectorFactory.fromBinaryArray(bytes);
        Assert.assertEquals(a, b);
    }
}