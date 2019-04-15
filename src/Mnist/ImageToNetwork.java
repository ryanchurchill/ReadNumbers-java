package Mnist;

import Exceptions.ValidationException;
import Network.NetworkWithObjects;

import java.util.*;

public class ImageToNetwork {
    public static void feedImageToNetwork(NetworkWithObjects n, Image i, boolean calcError) throws Exception
    {
        n.feedForward(i.getPixelsForNetwork());
        if (calcError) {
            n.calculateErrors(i.getPixelsForNetwork(), determineExpectedOutputValuesForDigit(i.getActualDigit()));
        }
    }

    /**
     * Returns the index of the output layer with the value closest to 1
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

    /**
     * Input: 0 Output: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     * Input: 5 Output: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
     * @param digit
     * @return
     * @throws ValidationException
     */
    public static List<Double> determineExpectedOutputValuesForDigit(int digit) throws ValidationException
    {
        if (digit < 0 || digit > 9) {
            throw new ValidationException("Digit not valid");
        }

        ArrayList<Double> ret = new ArrayList<>();
        for (int i=0; i<10; i++) {
            if (i == digit) {
                ret.add((double) 1);
            } else {
                ret.add((double) 0);
            }
        }

        return ret;
    }
}
