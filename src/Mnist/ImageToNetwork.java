package Mnist;

import Network.NetworkWithObjects;

public class ImageToNetwork {
    public static void feedImageToNetwork(NetworkWithObjects n, Image i) throws Exception
    {
        n.feedForward(i.getPixelsForNetwork());
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
}
