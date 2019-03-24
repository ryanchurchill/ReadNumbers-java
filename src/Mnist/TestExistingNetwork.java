package Mnist;

import Network.NetworkObjects.*;
import Network.NetworkUtils.*;
import Network.*;

import java.util.*;

public class TestExistingNetwork {

    static final String networkFilePath = "E:\\dev\\ai-az\\ReadNumbers-Java\\data\\wb_ryan_python.txt";

    public static void main(String[] args) throws Exception {
        List<Image> trainingImages = ReadMnist.getTrainingImages();
        LoadNetworkFromFileNumpy loader = new LoadNetworkFromFileNumpy(networkFilePath);
        NetworkWithObjects n = loader.load();

        for (int i = 0; i < 20; i++) {
            testImage(n, trainingImages.get(i));
        }
    }

    public static void testImage(NetworkWithObjects n, Image i) throws Exception
    {
        feedImageToNetwork(n, i);
        System.out.println("Actual value: " + i.getActualDigit());
        System.out.println("Network value: " + determineResultFromNetwork(n));
    }

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
