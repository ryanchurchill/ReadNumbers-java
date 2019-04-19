package Util;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.List;

public class MyMathUtils {
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

    public static RealVector sigmoid(RealVector v)
    {
        for (int i = 0; i < v.getDimension(); i++) {
            v.setEntry(i, sigmoid(v.getEntry(i)));
        }
        return v;
    }

    public static RealVector sigmoidPrime(RealVector v)
    {
        for (int i = 0; i < v.getDimension(); i ++) {
            v.setEntry(i, sigmoidPrime(v.getEntry(i)));
        }
        return v;
    }

    public static RealVector zeroes(int count)
    {
        double[] values = new double[count];
        return MatrixUtils.createRealVector(values);
    }

    /**
     * TODO: probably slow
     * This isn't exactly accurate - it doesn't care what shape the vectors are in
     * @param v1
     * @param v2
     * @return
     */
    public static RealMatrix multiplyVectors(RealVector v1, RealVector v2)
    {
        double[][] vals = new double[v1.getDimension()][v2.getDimension()];
        for (int i = 0; i < v1.getDimension(); i++) {
            for (int j = 0; j < v2.getDimension(); j++) {
                vals[i][j] = v1.getEntry(i) * v2.getEntry(j);
            }
        }

        return MatrixUtils.createRealMatrix(vals);
    }

    /**
     * TODO: are rows and columns correct?
     * @param rows
     * @param columns
     * @return
     */
    public static RealMatrix zeroes(int rows, int columns)
    {
        double[][] values = new double[rows][columns];
        return MatrixUtils.createRealMatrix(values);
    }

    /**
     * sigmoid: 2,327 milliseconds for 10 million, or 4,297,379 / sec
     * @param x
     * @return
     */
    public static double sigmoid(double x)
    {
        return 1 / (1 + Math.exp(-x));
    }

    public static double sigmoidPrime(double x)
    {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    public static RealVector[] addRealVectorArrays(RealVector[] vs1, RealVector[] vs2)
    {
        RealVector[] ret = new RealVector[vs1.length];

        for (int i = 0; i < vs1.length; i++) {
            ret[i] = vs1[i].add(vs2[i]);
        }

        return ret;
    }


    /*
    TESTING
     */

    public static void main(String[] args) {

    }

    public static void testAddVectorArrays()
    {

    }

    public static void testMultiplyVectors()
    {
        double[] v1Data = new double[2];
        v1Data[0] = 1;
        v1Data[1] = 2;
        double[] v2Data = new double[3];
        v2Data[0] = 3;
        v2Data[1] = 4;
        v2Data[2] = 5;

        RealMatrix m = multiplyVectors(MatrixUtils.createRealVector(v1Data), MatrixUtils.createRealVector(v2Data));
        System.out.println(m);
    }
}
