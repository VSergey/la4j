package org.la4j.vector;

import org.la4j.*;
import org.la4j.Vector;
import org.la4j.iterator.VectorIterator;

import java.text.NumberFormat;
import java.util.*;

public abstract class AbstractVector implements Vector {

    private static final String DEFAULT_DELIMITER = " ";

    /**
     * Length of this vector.
     */
    protected int length;

    /**
     * Creates a vector of given {@code length}.
     *
     * @param length the length of the vector
     */
    protected AbstractVector(int length) {
        ensureLengthIsCorrect(length);
        this.length = length;
    }

    public int length() {
        return length;
    }

    @Override
    public Vector slice(int from, int until) {
        if (until - from < 0) {
            fail("Wrong slice range: [" + from + ".." + until + "].");
        }

        Vector result = blank(until - from);

        for (int i = from; i < until; i++) {
            result.set(i - from, get(i));
        }

        return result;
    }

    @Override
    public Vector select(int ... indices) {
        int newLength = indices.length;

        if (newLength == 0) {
            fail("No elements selected.");
        }

        Vector result = blank(newLength);

        for (int i = 0; i < newLength; i++) {
            result.set(i, get(indices[i]));
        }

        return result;
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
    public boolean equals(Vector that, double precision) {
        if (this == that) {
            return true;
        }

        if (this.length != that.length()) {
            return false;
        }

        boolean result = true;

        for (int i = 0; result && i < length; i++) {
            double a = get(i);
            double b = that.get(i);
            double diff = Math.abs(a - b);
            result = (a == b) ||
                    (diff < precision || diff / Math.max(Math.abs(a), Math.abs(b)) < precision);
        }

        return result;
    }

    /**
     * Converts this vector into the string representation.
     *
     * @param formatter the number formatter
     *
     * @return the vector converted to a string
     */
    public String mkString(NumberFormat formatter) {
        return mkString(formatter, DEFAULT_DELIMITER);
    }

    /**
     * Converts this vector into the string representation.
     *
     * @param formatter the number formatter
     * @param delimiter the element's delimiter
     *
     * @return the vector converted to a string
     */
    public String mkString(NumberFormat formatter, String delimiter) {
        StringBuilder sb = new StringBuilder();
        VectorIterator it = iterator();

        while (it.hasNext()) {
            double x = it.next();
            int i = it.index();
            sb.append(formatter.format(x))
                    .append((i < length - 1 ? delimiter : ""));
        }

        return sb.toString();
    }

    /**
     * Converts this vector into a string representation.
     *
     * @return a string representation of this vector
     */
    @Override
    public String toString() {
        return mkString(DEFAULT_FORMATTER, DEFAULT_DELIMITER);
    }

    /**
     * Checks where this vector is equal to the given object {@code o}.
     */
    @Override
    public boolean equals(Object o) {
        return o != null && (o instanceof Vector) && equals((Vector) o, Vectors.EPS);
    }

    /**
     * Calculates the hash-code of this vector.
     */
    @Override
    public int hashCode() {
        VectorIterator it = iterator();
        int result = 17;

        while (it.hasNext()) {
            long value = it.next().longValue();
            result = 37 * result + (int) (value ^ (value >>> 32));
        }

        return result;
    }

    /**
     * Returns a vector iterator.
     *
     * @return a vector iterator.
     */
    @Override
    public VectorIterator iterator() {
        return new VectorIterator(length) {
            private int i = -1;

            @Override
            public int index() {
                return i;
            }

            @Override
            public double get() {
                return AbstractVector.this.get(i);
            }

            @Override
            public void set(double value) {
                AbstractVector.this.set(i, value);
            }

            @Override
            public boolean hasNext() {
                return i + 1 < length;
            }

            @Override
            public Double next() {
                if(!hasNext()) {
                    throw new NoSuchElementException();
                }
                i++;
                return get();
            }
        };
    }


    public String toCSV(NumberFormat formatter) {
        return mkString(formatter, ", ");
    }

    protected void ensureLengthIsCorrect(int length) {
        if (length < 0) {
            fail("Wrong vector length: " + length);
        }
        if (length == Integer.MAX_VALUE) {
            fail("Wrong vector length: use 'Integer.MAX_VALUE - 1' instead.");
        }
    }

    protected void fail(String message) {
        throw new IllegalArgumentException(message);
    }

}
