package org.la4j;

import org.la4j.vector.VectorFactory;

import java.util.Arrays;

public final class V {

    public static Vector v(double... values) {
        return Vectors.DENSE.fromArray(values);
    }

    public static Iterable<Vector> vs(double... values) {
        Vector vector = v(values);
        return Arrays.asList(
                Vectors.BASIC.convert(vector),
                Vectors.COMPRESSED.convert(vector)
        );
    }

    public static Vector vz(int length) {
        return VectorFactory.anyZero(length);
    }
}
