package Mnist;

import Exceptions.ValidationException;
import Network.Learning.TrainingExample;
import Network.NetworkWithArrays;
import Network.NetworkWithObjects;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

import java.util.*;

public class ImageToNetwork {
    public static void feedImageToNetwork(NetworkWithObjects n, Image i) throws Exception
    {
        n.feedForward(i.getPixelsForNetwork());
    }

    public static void feedImageToNetworkWithArrays(NetworkWithArrays n, Image i) throws Exception
    {
        RealVector rv1 = n.feedForward(i.getPixelsForArrayNetwork());
//        RealVector rv2 = na.feedForwardOld(i.getPixelsForArrayNetwork());
//        System.out.println("blah");
    }

    /**
     * Returns the index of the desiredOutput layer with the value closest to 1
     * @param n
     * @return
     */
    public static int determineResultFromNetwork(NetworkWithObjects n)
    {
        double closestValue = Double.MIN_VALUE;
        int closestValueIndex = -1;
        double[] outputValues = n.getOutputValues();
        for (int i = 0; i < outputValues.length; i++) {
            if (Math.abs(outputValues[i] - 1) < Math.abs(closestValue - 1)) {
                closestValue = outputValues[i];
                closestValueIndex = i;
            }
        }
        return closestValueIndex;
    }

    public static int determineResultFromVector(RealVector output)
    {
        double closestValue = Double.MIN_VALUE;
        int closestValueIndex = -1;
        double[] outputValues = output.toArray();
        for (int i = 0; i < outputValues.length; i++) {
            if (Math.abs(outputValues[i] - 1) < Math.abs(closestValue - 1)) {
                closestValue = outputValues[i];
                closestValueIndex = i;
            }
        }
        return closestValueIndex;
    }

    /**
     * Doesn't do anything
     * @param n
     * @param miniBatch
     * @throws ValidationException
     */
    public static void trainNetworkOnImageBatchArray(NetworkWithArrays n, List<Image> miniBatch) throws ValidationException
    {
        List<TrainingExample> tes = new ArrayList<>();
        for (Image i : miniBatch) {
            tes.add(new TrainingExample(i.getPixelsForArrayNetwork(), MatrixUtils.createRealVector(determineExpectedOutputValuesForDigit(i.getActualDigit()))));
        }
        n.trainWithMiniBatch(tes);
    }

    public static void trainNetworkOnImageBatch(NetworkWithObjects n, List<Image> miniBatch) throws ValidationException
    {
        List<TrainingExample> tes = new ArrayList<>();
        for (Image i : miniBatch) {
            tes.add(new TrainingExample(i.getPixelsForArrayNetwork(), MatrixUtils.createRealVector(determineExpectedOutputValuesForDigit(i.getActualDigit()))));
        }
        n.trainWithMiniBatch(tes);
    }

    /**
     * Input: 0 Output: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     * Input: 5 Output: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
     * @param digit
     * @return
     * @throws ValidationException
     */
    public static double[] determineExpectedOutputValuesForDigit(int digit) throws ValidationException
    {
        if (digit < 0 || digit > 9) {
            throw new ValidationException("Digit not valid");
        }

        double[] ret = new double[10];
        for (int i=0; i<10; i++) {
            if (i == digit) {
                ret[i] = 1;
            } else {
                ret[i] = 0;
            }
        }

        return ret;
    }
}
