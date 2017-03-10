package org.la4j.operation.ooplace;

import org.la4j.Matrices;
import org.la4j.Matrix;
import org.la4j.matrix.DenseMatrix;
import org.la4j.matrix.ColumnMajorSparseMatrix;
import org.la4j.matrix.RowMajorSparseMatrix;
import org.la4j.matrix.sparse.CRSMatrix;
import org.la4j.operation.MatrixOperation;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class OoPlaceMatrixByItsTransposeMultiplication implements MatrixOperation<Matrix> {

    public Matrix apply(DenseMatrix a) {
        Matrix result = a.blankOfShape(a.rows(), a.rows());

        for (int i = 0; i < a.rows(); i++) {
            for (int j = 0; j < a.rows(); j++) {
                double acc = 0.0;
                for (int k = 0; k < a.columns(); k++) {
                    acc += a.get(i, k) * a.get(j, k);
                }
                result.set(i, j, acc);
            }
        }
        return result;
    }

    public Matrix apply(RowMajorSparseMatrix a) {
        Matrix result = a.blankOfShape(a.rows(), a.rows());
        List<Integer> nzRows = new ArrayList<Integer>();
        Iterator<Integer> it = a.iteratorOfNonZeroRows();

        while (it.hasNext()) {
            nzRows.add(it.next());
        }
        for (int i: nzRows) {
            for (int j: nzRows) {
                result.set(i, j, a.nonZeroIteratorOfRow(i)
                                  .innerProduct(a.nonZeroIteratorOfRow(j)));
            }
        }
        return result;
    }

    public Matrix apply(ColumnMajorSparseMatrix a) {
        // TODO: Implement its own algorithm
        CRSMatrix matrix = Matrices.SPARSE_ROW_MAJOR.convert(a);
        return apply(matrix);
    }
}
