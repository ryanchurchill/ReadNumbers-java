package Util;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
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

    public static RealMatrix[] addRealMatrixArrays(RealMatrix[] ms1, RealMatrix[] ms2)
    {
        RealMatrix[] ret = new RealMatrix[ms1.length];

        for (int i = 0; i < ms1.length; i++) {
            ret[i] = ms1[i].add(ms2[i]);
        }

        return ret;
    }

    public static void applyGradientDescentToBiases(RealVector[] biases, RealVector[] biasNablas, double learningRate, double batchSize)
    {
        for (int layerIndex = 0; layerIndex < biases.length; layerIndex++) {
            RealVector biasesAtLayer = biases[layerIndex];
            for (int biasIndex = 0; biasIndex < biasesAtLayer.getDimension(); biasIndex++) {
                double bias = biasesAtLayer.getEntry(biasIndex);
                double newBias =  bias - (learningRate / batchSize * biasNablas[layerIndex].getEntry(biasIndex));
                biasesAtLayer.setEntry(biasIndex, newBias);
            }
        }
    }

    public static void applyGradientDescentToWeights(RealMatrix[] weights, RealMatrix[] weightNablas, double learningRate, double batchSize)
    {
        for (int layerIndex = 0; layerIndex < weightNablas.length; layerIndex++) {
            RealMatrix weightsAtLayer = weights[layerIndex];
            for (int i = 0; i < weightsAtLayer.getRowDimension(); i++) {
                for (int j = 0; j < weightsAtLayer.getColumnDimension(); j++) {
                    double weight = weightsAtLayer.getEntry(i, j);
                    double newWeight = weight - (learningRate / batchSize * weightNablas[layerIndex].getEntry(i, j));
                    weightsAtLayer.setEntry(i, j, newWeight);
                }
            }
        }
    }

    public static List<Double> rvToList(RealVector rv)
    {
        List<Double> ret = new ArrayList<Double>();
        double[] arr = rv.toArray();
        for (int i=0; i < arr.length; i++) {
            ret.add(arr[i]);
        }
        return ret;
    }

    public static RealVector listToRv(List<Double> list)
    {
        double[] arr = new double[list.size()];
        for (int i=0; i<list.size(); i++) {
            arr[i] = list.get(i);
        }
        return MatrixUtils.createRealVector(arr);
    }

    /*
    TESTING
     */

    public static void main(String[] args) {
        testAddVectorArrays();
    }

    public static void testAddVectorArrays()
    {
        RealVector[] vs1 = new RealVector[2];
        RealVector[] vs2 = new RealVector[2];
//        RealMatrix[] ms1 = new RealMatrix[2];
//        RealMatrix[] ms2 = new RealMatrix[2];

        double[] vals = new double[2];
        vals[0] = 3;
        vals[1] = 4;
        vs1[0] = MatrixUtils.createRealVector(vals);
        vals[0] = 5;
        vals[1] = 6;
        vs1[1] = MatrixUtils.createRealVector(vals);

        vals[0] = 1;
        vals[1] = 4;
        vs2[0] = MatrixUtils.createRealVector(vals);
        vals[0] = 2;
        vals[1] = 3;
        vs2[1] = MatrixUtils.createRealVector(vals);

        RealVector[] response = addRealVectorArrays(vs1, vs2);
        System.out.println(response);
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
