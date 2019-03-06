package Util;

import org.apache.commons.math3.linear.RealMatrix;

import java.util.List;

public class MyMatrixUtils {
    public static void printRealMatrix(RealMatrix m)
    {
        for (int i = 0; i < m.getRowDimension(); i++) {
            double[] row = m.getRow(i);
            for (double val : row) {
                System.out.print(val + ", ");
            }
            System.out.println();
        }
    }

    public static void printRealMatrices(List<RealMatrix> list)
    {
        for (int i = 0; i < list.size(); i++) {
            System.out.println(i + ":");
            printRealMatrix(list.get(i));
        }
    }

    public static RealMatrix sigmoid(RealMatrix m)
    {
        RealMatrix ret = m.copy();
        for (int row = 0; row < ret.getRowDimension(); row++) {
            for (int column = 0; column < ret.getColumnDimension(); column++) {
                ret.setEntry(row, column, sigmoid(ret.getEntry(row, column)));
            }
        }
        return ret;
    }

    public static double sigmoid(double x)
    {
        return 1 / (1 + Math.exp(-x));
    }

}
