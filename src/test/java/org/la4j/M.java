package org.la4j;

import org.la4j.matrix.MatrixFactory;

import java.util.Arrays;

public class M {

    public static double[] a(double... values) {
        return values;
    }

    public static Matrix m(double[]... values) {
        return Matrices.BASIC_2D.from2DArray(values);
    }

    public static Iterable<Matrix> ms(double[]... values) {
        Matrix matrix = m(values);
        return Arrays.asList(
                Matrices.CCS.convert(matrix),
                Matrices.CRS.convert(matrix),
                Matrices.BASIC_1D.convert(matrix),
                Matrices.BASIC_2D.convert(matrix)
        );
    }

    public static Matrix mz(int rows, int columns) {
        return MatrixFactory.zeroMatrix(rows, columns);
    }
}
