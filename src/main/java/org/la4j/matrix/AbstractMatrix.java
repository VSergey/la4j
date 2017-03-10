package org.la4j.matrix;

import org.la4j.*;
import org.la4j.Vector;
import org.la4j.decomposition.MatrixDecompositor;
import org.la4j.inversion.MatrixInverter;
import org.la4j.iterator.*;
import org.la4j.linear.LinearSystemSolver;
import org.la4j.matrix.functor.*;
import org.la4j.vector.functor.*;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;

abstract class AbstractMatrix implements Matrix {
    private static final String DEFAULT_ROWS_DELIMITER = "\n";
    private static final String DEFAULT_COLUMNS_DELIMITER = " ";
    private static final NumberFormat DEFAULT_FORMATTER = new DecimalFormat("0.000");
    private static final String[] INDENTS = { // 9 predefined indents for alignment
            " ",
            "  ",
            "   ",
            "    ",
            "     ",
            "      ",
            "       ",
            "        ",
            "         ",
            "          "
    };

    protected int rows;
    protected int columns;

    /**
     * Creates a matrix of given shape {@code rows} x {@code columns};
     */
    AbstractMatrix(int rows, int columns) {
        ensureDimensionsAreCorrect(rows, columns);
        this.rows = rows;
        this.columns = columns;
    }

    public int rows() {
        return rows;
    }

    public int columns() {
        return columns;
    }

    public void setAll(double value) {
        MatrixIterator it = iterator();

        while (it.hasNext()) {
            it.next();
            it.set(value);
        }
    }

    public void setRow(int i, double value) {
        VectorIterator it = iteratorOfRow(i);

        while (it.hasNext()) {
            it.next();
            it.set(value);
        }
    }

    public void setColumn(int j, double value) {
        VectorIterator it = iteratorOfColumn(j);

        while (it.hasNext()) {
            it.next();
            it.set(value);
        }
    }

    public void swapRows(int i, int j) {
        if (i != j) {
            Vector ii = getRow(i);
            Vector jj = getRow(j);

            setRow(i, jj);
            setRow(j, ii);
        }
    }

    public void swapColumns(int i, int j) {
        if (i != j) {
            Vector ii = getColumn(i);
            Vector jj = getColumn(j);

            setColumn(i, jj);
            setColumn(j, ii);
        }
    }

    public Matrix blankOfShape(int rows, int columns) {
        return factory().zero(rows, columns);
    }

    public Matrix transpose() {
        Matrix result = blankOfShape(columns(), rows());
        MatrixIterator it = result.iterator();

        while (it.hasNext()) {
            it.next();
            int i = it.rowIndex();
            int j = it.columnIndex();
            it.set(get(j, i));
        }
        return result;
    }

    public Matrix rotate() {
        Matrix result = blankOfShape(columns(), rows());
        MatrixIterator it = result.iterator();

        while (it.hasNext()) {
            it.next();
            int i = it.rowIndex();
            int j = it.columnIndex();
            it.set(get(rows() - 1 - j, i));
        }
        return result;
    }

    public Matrix power(int n) {
        if (n < 0) {
            fail("The exponent should be positive: " + n + ".");
        }

        Matrix result = blankOfShape(rows(), rows());
        Matrix that = this;

        for (int i = 0; i < rows(); i++) {
            result.set(i, i, 1.0);
        }
        while (n > 0) {
            if (n % 2 == 1) {
                result = result.multiply(that);
            }

            n /= 2;
            that = that.multiply(that);
        }
        return result;
    }

    public Matrix multiply(double value) {
        Matrix result = blank();
        MatrixIterator it = iterator();

        while (it.hasNext()) {
            double x = it.next();
            int i = it.rowIndex();
            int j = it.columnIndex();
            result.set(i, j, x * value);
        }
        return result;
    }

    public Vector multiply(Vector that) {
        return apply(LinearAlgebra.OO_PLACE_MATRIX_BY_VECTOR_MULTIPLICATION, that);
    }

    public Matrix multiply(Matrix that) {
        return apply(LinearAlgebra.OO_PLACE_MATRICES_MULTIPLICATION, that);
    }

    public Matrix multiplyByItsTranspose() {
        return apply(LinearAlgebra.OO_PLACE_MATRIX_BY_ITS_TRANSPOSE_MULTIPLICATION);
    }

    public Matrix subtract(double value) {
        return add(-value);
    }

    public Matrix subtract(Matrix that) {
        return apply(LinearAlgebra.OO_PLACE_MATRICES_SUBTRACTION, that);
    }

    public Matrix add(double value) {
        MatrixIterator it = iterator();
        Matrix result = blank();

        while (it.hasNext()) {
            double x = it.next();
            int i = it.rowIndex();
            int j = it.columnIndex();
            result.set(i, j, x + value);
        }
        return result;
    }

    public Matrix add(Matrix that) {
        return apply(LinearAlgebra.OO_PLACE_MATRIX_ADDITION, that);
    }

    public Matrix insert(Matrix that) {
        return insert(that, 0, 0, 0, 0, that.rows(), that.columns());
    }

    public Matrix insert(Matrix that, int rows, int columns) {
        return insert(that, 0, 0, 0, 0, rows, columns);
    }

    public Matrix insert(Matrix that, int destRow, int destColumn, int rows, int columns) {
        return insert(that, 0, 0, destRow, destColumn, rows, columns);
    }

    public Matrix insert(Matrix that, int srcRow, int srcColumn, int destRow, int destColumn, int rows, int columns) {
        if (rows < 0 || columns < 0) {
            fail("Cannot have negative rows or columns: " + rows + "x" + columns);
        }

        if (destRow < 0 || destColumn < 0) {
            fail("Cannot have negative destination position: " + destRow + ", " + destColumn);
        }

        if (destRow > rows() || destColumn > columns()) {
            fail("Destination position out of bounds: " + destRow + ", " + destColumn);
        }

        if (srcRow < 0 || srcColumn < 0) {
            fail("Cannot have negative source position: " + destRow + ", " + destColumn);
        }

        if (srcRow > that.rows() || srcColumn > that.columns()) {
            fail("Destination position out of bounds: " + srcRow + ", " + srcColumn);
        }

        if (destRow + rows > rows() || destColumn + columns > columns()) {
            fail("Out of bounds: Cannot add " + rows + " rows and " + columns + " cols at "
                    + destRow + ", " + destColumn + " in a " + rows() + "x" + columns() + " matrix.");
        }

        if (srcRow + rows > that.rows() || srcColumn + columns > that.columns()) {
            fail("Out of bounds: Cannot get " + rows + " rows and " + columns + " cols at "
                    + srcRow + ", " + srcColumn + " from a " + that.rows() + "x" + that.columns() + " matrix.");
        }

        Matrix result = copy();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result.set(i + destRow, j + destColumn, that.get(i + srcRow, j + srcColumn));
            }
        }
        return result;
    }

    public Matrix divide(double value) {
        return multiply(1.0 / value);
    }

    public Matrix kroneckerProduct(Matrix that) {
        return apply(LinearAlgebra.OO_PLACE_KRONECKER_PRODUCT, that);
    }

    public double trace() {
        double result = 0.0;

        for (int i = 0; i < rows(); i++) {
            result += get(i, i);
        }
        return result;
    }

    public double diagonalProduct() {
        BigDecimal result = BigDecimal.ONE;

        for (int i = 0; i < rows; i++) {
            result = result.multiply(BigDecimal.valueOf(get(i, i)));
        }
        return result.setScale(Matrices.ROUND_FACTOR, RoundingMode.CEILING).doubleValue();
    }

    public double norm() {
        return euclideanNorm();
    }

    public double euclideanNorm() {
        return fold(Matrices.mkEuclideanNormAccumulator());
    }

    public double manhattanNorm() {
        return fold(Matrices.mkManhattanNormAccumulator());
    }

    public double infinityNorm() {
        return fold(Matrices.mkInfinityNormAccumulator());
    }

    public double product() {
        return fold(Matrices.asProductAccumulator(1.0));
    }

    public double sum() {
        return fold(Matrices.asSumAccumulator(0.0));
    }

    public Matrix hadamardProduct(Matrix that) {
        return apply(LinearAlgebra.OO_PLACE_MATRIX_HADAMARD_PRODUCT, that);
    }

    public double determinant() {
        if (rows != columns) {
            throw new IllegalStateException("Can not compute determinant of non-square matrix.");
        }
        if (rows == 0) {
            return 0.0;
        } else if (rows == 1) {
            return get(0, 0);
        } else if (rows == 2) {
            return get(0, 0) * get(1, 1) -
                    get(0, 1) * get(1, 0);
        } else if (rows == 3) {
            return get(0, 0) * get(1, 1) * get(2, 2) +
                    get(0, 1) * get(1, 2) * get(2, 0) +
                    get(0, 2) * get(1, 0) * get(2, 1) -
                    get(0, 2) * get(1, 1) * get(2, 0) -
                    get(0, 1) * get(1, 0) * get(2, 2) -
                    get(0, 0) * get(1, 2) * get(2, 1);
        }

        MatrixDecompositor decompositor = withDecompositor(LinearAlgebra.LU);
        Matrix[] lup = decompositor.decompose();
        // TODO: Why Java doesn't support pattern matching?
        Matrix u = lup[1];
        Matrix p = lup[2];

        double result = u.diagonalProduct();

        // TODO: we can do that in O(n log n)
        //       just google: "counting inversions divide and conqueror"
        int[] permutations = new int[p.rows()];
        for (int i = 0; i < p.rows(); i++) {
            for (int j = 0; j < p.columns(); j++) {
                if (p.get(i, j) > 0.0) {
                    permutations[i] = j;
                    break;
                }
            }
        }
        int sign = 1;
        for (int i = 0; i < permutations.length; i++) {
            for (int j = i + 1; j < permutations.length; j++) {
                if (permutations[j] < permutations[i]) {
                    sign *= -1;
                }
            }
        }
        return sign * result;
    }

    public int rank() {
        if (rows == 0 || columns == 0) {
            return 0;
        }
        // TODO:
        // handle small (1x1, 1xn, nx1, 2x2, 2xn, nx2, 3x3, 3xn, nx3)
        // matrices without SVD

        MatrixDecompositor decompositor = withDecompositor(LinearAlgebra.SVD);
        Matrix[] usv = decompositor.decompose();
        // TODO: Where is my pattern matching?
        Matrix s = usv[1];
        double tolerance = Math.max(rows, columns) * s.get(0, 0) * Matrices.EPS;

        int result = 0;
        for (int i = 0; i < s.rows(); i++) {
            if (s.get(i, i) > tolerance) {
                result++;
            }
        }

        return result;
    }

    public void setRow(int i, Vector row) {
        if (columns != row.length()) {
            fail("Wrong vector length: " + row.length() + ". Should be: " + columns + ".");
        }
        for (int j = 0; j < row.length(); j++) {
            set(i, j, row.get(j));
        }
    }

    public void setColumn(int j, Vector column) {
        if (rows != column.length()) {
            fail("Wrong vector length: " + column.length() + ". Should be: " + rows + ".");
        }
        for (int i = 0; i < column.length(); i++) {
            set(i, j, column.get(i));
        }
    }

    public Matrix insertRow(int i, Vector row) {
        if (i >= rows || i < 0) {
            throw new IndexOutOfBoundsException("Illegal row number, must be 0.." + (rows - 1));
        }

        Matrix result = blankOfShape(rows + 1, columns);

        for (int ii = 0; ii < i; ii++) {
            result.setRow(ii, getRow(ii));
        }

        result.setRow(i, row);

        for (int ii = i; ii < rows; ii++) {
            result.setRow(ii + 1, getRow(ii));
        }
        return result;
    }

    public Matrix insertColumn(int j, Vector column) {
        if (j >= columns || j < 0) {
            throw new IndexOutOfBoundsException("Illegal column number, must be 0.." + (columns - 1));
        }

        Matrix result = blankOfShape(rows, columns + 1);

        for (int jj = 0; jj < j; jj++) {
            result.setColumn(jj, getColumn(jj));
        }

        result.setColumn(j, column);

        for (int jj = j; jj < columns; jj++) {
            result.setColumn(jj + 1, getColumn(jj));
        }
        return result;
    }

    public Matrix removeRow(int i) {
        if (i >= rows || i < 0) {
            throw new IndexOutOfBoundsException("Illegal row number, must be 0.." + (rows - 1));
        }

        Matrix result = blankOfShape(rows - 1, columns);

        for (int ii = 0; ii < i; ii++) {
            result.setRow(ii, getRow(ii));
        }

        for (int ii = i + 1; ii < rows; ii++) {
            result.setRow(ii - 1, getRow(ii));
        }
        return result;
    }

    public Matrix removeColumn(int j) {
        if (j >= columns || j < 0) {
            throw new IndexOutOfBoundsException("Illegal column number, must be 0.." + (columns - 1));
        }

        Matrix result = blankOfShape(rows, columns - 1);

        for (int jj = 0; jj < j; jj++) {
            result.setColumn(jj, getColumn(jj));
        }

        for (int jj = j + 1; jj < columns; jj++) {
            result.setColumn(jj - 1, getColumn(jj));
        }
        return result;
    }

    public Matrix removeFirstRow() {
        return removeRow(0);
    }

    public Matrix removeFirstColumn() {
        return removeColumn(0);
    }

    public Matrix removeLastRow() {
        return removeRow(rows - 1);
    }

    public Matrix removeLastColumn() {
        return removeColumn(columns - 1);
    }

    public Matrix blank() {
        return blankOfShape(rows, columns);
    }

    public Matrix blankOfRows(int rows) {
        return blankOfShape(rows, columns);
    }

    public Matrix blankOfColumns(int columns) {
        return blankOfShape(rows, columns);
    }

    public Matrix copy() {
        return copyOfShape(rows, columns);
    }

    public Matrix copyOfRows(int rows) {
        return copyOfShape(rows, columns);
    }

    public Matrix copyOfColumns(int columns) {
        return copyOfShape(rows, columns);
    }

    public Matrix shuffle() {
        Matrix result = copy();

        // Conduct Fisher-Yates shuffle
        Random random = new Random();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                int ii = random.nextInt(rows - i) + i;
                int jj = random.nextInt(columns - j) + j;

                double a = result.get(ii, jj);
                result.set(ii, jj, result.get(i, j));
                result.set(i, j, a);
            }
        }
        return result;
    }

    public Matrix slice(int fromRow, int fromColumn, int untilRow, int untilColumn) {
        ensureIndexArgumentsAreInBounds(fromRow, fromColumn);
        ensureIndexArgumentsAreInBounds(untilRow - 1, untilColumn - 1);

        if (untilRow - fromRow < 0 || untilColumn - fromColumn < 0) {
            fail("Wrong slice range: [" + fromRow + ".." + untilRow + "][" + fromColumn + ".." + untilColumn + "].");
        }

        Matrix result = blankOfShape(untilRow - fromRow, untilColumn - fromColumn);

        for (int i = fromRow; i < untilRow; i++) {
            for (int j = fromColumn; j < untilColumn; j++) {
                result.set(i - fromRow, j - fromColumn, get(i, j));
            }
        }
        return result;
    }

    public Matrix sliceTopLeft(int untilRow, int untilColumn) {
        return slice(0, 0, untilRow, untilColumn);
    }

    public Matrix sliceBottomRight(int fromRow, int fromColumn) {
        return slice(fromRow, fromColumn, rows, columns);
    }

    public Matrix select(int[] rowIndices, int[] columnIndices) {
        int m = rowIndices.length;
        int n = columnIndices.length;

        if (m == 0 || n == 0) {
            fail("No rows or columns selected.");
        }

        Matrix result = blankOfShape(m, n);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result.set(i, j, get(rowIndices[i], columnIndices[j]));
            }
        }
        return result;
    }

    public void each(MatrixProcedure procedure) {
        MatrixIterator it = iterator();

        while (it.hasNext()) {
            double x = it.next();
            int i = it.rowIndex();
            int j = it.columnIndex();
            procedure.apply(i, j, x);
        }
    }

    public void eachInRow(int i, VectorProcedure procedure) {
        VectorIterator it = iteratorOfRow(i);

        while (it.hasNext()) {
            double x = it.next();
            int j = it.index();
            procedure.apply(j, x);
        }
    }

    public void eachInColumn(int j, VectorProcedure procedure) {
        VectorIterator it = iteratorOfColumn(j);

        while (it.hasNext()) {
            double x = it.next();
            int i = it.index();
            procedure.apply(i, x);
        }
    }

    public double max() {
        return fold(Matrices.mkMaxAccumulator());
    }

    public double min() {
        return fold(Matrices.mkMinAccumulator());
    }

    public double maxInRow(int i) {
        return foldRow(i, Vectors.mkMaxAccumulator());
    }

    public double minInRow(int i) {
        return foldRow(i, Vectors.mkMinAccumulator());
    }

    public double maxInColumn(int j) {
        return foldColumn(j, Vectors.mkMaxAccumulator());
    }

    public double minInColumn(int j) {
        return foldColumn(j, Vectors.mkMinAccumulator());
    }

    public Matrix transform(MatrixFunction function) {
        Matrix result = blank();
        MatrixIterator it = iterator();

        while (it.hasNext()) {
            double x  = it.next();
            int i = it.rowIndex();
            int j = it.columnIndex();
            result.set(i, j, function.evaluate(i, j, x));
        }
        return result;
    }

    public Matrix transformRow(int i, VectorFunction function) {
        Matrix result = copy();
        VectorIterator it = result.iteratorOfRow(i);

        while (it.hasNext()) {
            double x = it.next();
            int j = it.index();
            it.set(function.evaluate(j, x));
        }
        return result;
    }

    public Matrix transformColumn(int j, VectorFunction function) {
        Matrix result = copy();
        VectorIterator it = result.iteratorOfColumn(j);

        while (it.hasNext()) {
            double x = it.next();
            int i = it.index();
            it.set(function.evaluate(i, x));
        }
        return result;
    }

    public void update(MatrixFunction function) {
        MatrixIterator it = iterator();

        while (it.hasNext()) {
            double x = it.next();
            int i = it.rowIndex();
            int j = it.columnIndex();
            it.set(function.evaluate(i, j, x));
        }
    }

    public void updateAt(int i, int j, MatrixFunction function) {
        set(i, j, function.evaluate(i, j, get(i, j)));
    }

   public void updateRow(int i, VectorFunction function) {
        VectorIterator it = iteratorOfRow(i);

        while (it.hasNext()) {
            double x = it.next();
            int j = it.index();
            it.set(function.evaluate(j, x));
        }
    }

    public void updateColumn(int j, VectorFunction function) {
        VectorIterator it = iteratorOfColumn(j);

        while (it.hasNext()) {
            double x = it.next();
            int i = it.index();
            it.set(function.evaluate(i, x));
        }
    }

    public double fold(MatrixAccumulator accumulator) {
        each(Matrices.asAccumulatorProcedure(accumulator));
        return accumulator.accumulate();
    }

   public double foldRow(int i, VectorAccumulator accumulator) {
        eachInRow(i, Vectors.asAccumulatorProcedure(accumulator));
        return accumulator.accumulate();
    }

    public double[] foldRows(VectorAccumulator accumulator) {
        double[] result = new double[rows];

        for (int i = 0; i < rows; i++) {
            result[i] = foldRow(i, accumulator);
        }
        return result;
    }

    public double foldColumn(int j, VectorAccumulator accumulator) {
        eachInColumn(j, Vectors.asAccumulatorProcedure(accumulator));
        return accumulator.accumulate();
    }

    public double[] foldColumns(VectorAccumulator accumulator) {
        double[] result = new double[columns];

        for (int i = 0; i < columns; i++) {
            result[i] = foldColumn(i, accumulator);
        }
        return result;
    }

    public boolean is(MatrixPredicate predicate) {
        MatrixIterator it = iterator();
        boolean result = predicate.test(rows, columns);

        while (it.hasNext() && result) {
            double x = it.next();
            int i = it.rowIndex();
            int j = it.columnIndex();
            result = predicate.test(i, j, x);
        }
        return result;
    }

    public boolean is(AdvancedMatrixPredicate predicate) {
        return predicate.test(this);
    }

    public boolean non(MatrixPredicate predicate) {
        return !is(predicate);
    }

    public boolean non(AdvancedMatrixPredicate predicate) {
        return !is(predicate);
    }

    public Vector toRowVector() {
        return getRow(0);
    }

    public Vector toColumnVector() {
        return getColumn(0);
    }

    public LinearSystemSolver withSolver(LinearAlgebra.SolverFactory factory) {
        return factory.create(this);
    }

    public MatrixInverter withInverter(LinearAlgebra.InverterFactory factory) {
        return factory.create(this);
    }

    public MatrixDecompositor withDecompositor(LinearAlgebra.DecompositorFactory factory) {
        return factory.create(this);
    }

    public boolean equals(Matrix matrix, double precision) {
        if (rows != matrix.rows() || columns != matrix.columns()) {
            return false;
        }

        boolean result = true;

        for (int i = 0; result && i < rows; i++) {
            for (int j = 0; result && j < columns; j++) {
                double a = get(i, j);
                double b = matrix.get(i, j);
                double diff = Math.abs(a - b);

                result = (a == b) || (diff < precision || diff / Math.max(Math.abs(a), Math.abs(b)) < precision);
            }
        }

        return result;
    }

    public String mkString(NumberFormat formatter) {
        return mkString(formatter, DEFAULT_ROWS_DELIMITER, DEFAULT_COLUMNS_DELIMITER);
    }

    public String mkString(String rowsDelimiter, String columnsDelimiter) {
        return mkString(DEFAULT_FORMATTER, rowsDelimiter, columnsDelimiter);
    }

    public String mkString(NumberFormat formatter, String rowsDelimiter, String columnsDelimiter) {
        // TODO: rewrite using iterators
        int[] formats = new int[columns];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                double value = get(i, j);
                String output = formatter.format(value);
                int size = output.length();
                formats[j] = size > formats[j] ? size : formats[j];
            }
        }

        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                String output = formatter.format(get(i, j));
                int outputLength = output.length();

                if (outputLength < formats[j]) {
                    int align = formats[j] - outputLength;
                    if (align > INDENTS.length - 1) {
                        indent(sb, align);
                    } else {
                        sb.append(INDENTS[align - 1]);
                    }
                }

                sb.append(output)
                        .append(j < columns - 1 ? columnsDelimiter : "");
            }
            sb.append(rowsDelimiter);
        }

        return sb.toString();
    }

    @Override
    public String toString() {
        return mkString(DEFAULT_FORMATTER, DEFAULT_ROWS_DELIMITER, DEFAULT_COLUMNS_DELIMITER);
    }

    @Override
    public MatrixIterator iterator() {
        return rowMajorIterator();
    }

    public RowMajorMatrixIterator rowMajorIterator() {
        return new RowMajorMatrixIterator(rows, columns) {
            private long limit = (long) rows * columns;
            private int i = - 1;

            public int rowIndex() {
                return i / columns;
            }

            public int columnIndex() {
                return i - rowIndex() * columns;
            }

            public double get() {
                return AbstractMatrix.this.get(rowIndex(), columnIndex());
            }

            public void set(double value) {
                AbstractMatrix.this.set(rowIndex(), columnIndex(), value);
            }

            public boolean hasNext() {
                return i + 1 < limit;
            }

            public Double next() {
                if(!hasNext()) {
                    throw new NoSuchElementException();
                }
                i++;
                return get();
            }
        };
    }

   public ColumnMajorMatrixIterator columnMajorIterator() {
        return new ColumnMajorMatrixIterator(rows, columns) {
            private long limit = (long) rows * columns;
            private int i = -1;

            public int rowIndex() {
                return i - columnIndex() * rows;
            }

            public int columnIndex() {
                return i / rows;
            }

            public double get() {
                return AbstractMatrix.this.get(rowIndex(), columnIndex());
            }

            public void set(double value) {
                AbstractMatrix.this.set(rowIndex(), columnIndex(), value);
            }

            public boolean hasNext() {
                return i + 1 < limit;
            }

            public Double next() {
                if(!hasNext()) {
                    throw new NoSuchElementException();
                }
                i++;
                return get();
            }
        };
    }

    public VectorIterator iteratorOfRow(final int i) {
        return new VectorIterator(columns) {
            private int j = -1;

            public int index() {
                return j;
            }

            public double get() {
                return AbstractMatrix.this.get(i, j);
            }

            public void set(double value) {
                AbstractMatrix.this.set(i, j, value);
            }

            public boolean hasNext() {
                return j + 1 < columns;
            }

            public Double next() {
                if(!hasNext()) {
                    throw new NoSuchElementException();
                }
                j++;
                return get();
            }
        };
    }

    /**
     * Returns a vector iterator of the given column {code j}.
     *
     * @return a vector iterator
     */
    public VectorIterator iteratorOfColumn(final int j) {
        return new VectorIterator(rows) {
            private int i = -1;
            public int index() {
                return i;
            }

            public double get() {
                return AbstractMatrix.this.get(i, j);
            }

            public void set(double value) {
                AbstractMatrix.this.set(i, j, value);
            }

            public boolean hasNext() {
                return i + 1 < rows;
            }

            public Double next() {
                if(!hasNext()) {
                    throw new NoSuchElementException();
                }
                i++;
                return get();
            }
        };
    }

    @Override
    public int hashCode() {
        MatrixIterator it = iterator();
        int result = 17;

        while (it.hasNext()) {
            long value = it.next().longValue();
            result = 37 * result + (int) (value ^ (value >>> 32));
        }

        return result;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o ) return true;
        if (o == null || !(o instanceof Matrix)) {
            return false;
        }
        return equals((Matrix) o, Matrices.EPS);
    }

    public String toCSV() {
        return toCSV(DEFAULT_FORMATTER);
    }

    public String toMatrixMarket() {
        return toMatrixMarket(DEFAULT_FORMATTER);
    }

    public String toCSV(NumberFormat formatter) {
        return mkString(formatter, "\n", ", ");
    }

    protected void ensureDimensionsAreCorrect(int rows, int columns) {
        if (rows < 0 || columns < 0) {
            fail("Wrong matrix dimensions: " + rows + "x" + columns);
        }
        if (rows == Integer.MAX_VALUE || columns == Integer.MAX_VALUE) {
            fail("Wrong matrix dimensions: use 'Integer.MAX_VALUE - 1' instead.");
        }
    }

    protected void ensureIndexArgumentsAreInBounds(int i, int j) {
        if (i < 0 || i >= rows) {
            fail(String.format("Bad row argument %d; out of bounds", i));
        }
        if (j < 0 || j >= columns) {
            fail(String.format("Bad column argument %d; out of bounds", j));
        }
    }

    protected void ensureIndexesAreInBounds(int i, int j) {
        if (i < 0 || i >= rows) {
            throw new IndexOutOfBoundsException("Row '" + i + "' is invalid.");
        }
        if (j < 0 || j >= columns) {
            throw new IndexOutOfBoundsException("Column '" + j + "' is invalid.");
        }
    }

    protected void fail(String message) {
        throw new IllegalArgumentException(message);
    }

    private void indent(StringBuilder sb, int howMany) {
        while (howMany > 0) {
            sb.append(" ");
            howMany--;
        }
    }
}
