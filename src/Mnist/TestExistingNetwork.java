package Mnist;

import Network.Learning.TrainingExample;
import Network.NetworkUtils.*;
import Network.*;

import java.util.*;

public class TestExistingNetwork {

    static final String digitNetworkPath = "E:\\dev\\ai-az\\ReadNumbers-Java\\data\\wb_ryan_python.txt";
    static final String simpleNetworkPath = "E:\\dev\\ai-az\\ReadNumbers-Java\\data\\sandbox_ryan_python.txt";

    public static void main(String[] args) throws Exception
    {
        testSimpleNetwork();
    }

    public static void testSimpleNetwork() throws Exception
    {
        LoadNetworkFromFileNumpy loader = new LoadNetworkFromFileNumpy(simpleNetworkPath);
        NetworkWithObjects simpleNetwork = loader.load();

        // TE1: 1, 2 -> 0, 1
        List<Double> input1 = new ArrayList<>();
        input1.add(1.0);
        input1.add(2.0);
        List<Double> desiredOutput1 = new ArrayList<>();
        desiredOutput1.add(0.0);
        desiredOutput1.add(1.0);
        TrainingExample te1 = new TrainingExample(input1, desiredOutput1);

        // TE2: 2, 1 -> 1, 0
        List<Double> input2 = new ArrayList<>();
        input2.add(2.0);
        input2.add(1.0);
        List<Double> desiredOutput2 = new ArrayList<>();
        desiredOutput2.add(1.0);
        desiredOutput2.add(0.0);
        TrainingExample te2 = new TrainingExample(input2, desiredOutput2);

        System.out.println("Initial network:");
        System.out.println(simpleNetwork);

//        simpleNetwork.feedForward(input2);
//        simpleNetwork.calculateErrors(desiredOutput2);
//        simpleNetwork.updateWeightsAndBiasesAfterSingleTrainingExample();

        List<TrainingExample> miniBatch = new ArrayList<>();
        miniBatch.add(te1);
        miniBatch.add(te2);
        simpleNetwork.trainWithMiniBatch(miniBatch);

        System.out.println("Network after one mini-batch:");
        System.out.println(simpleNetwork);
    }

    public static void testDigits(String[] args) throws Exception {
        List<Image> trainingImages = ReadMnist.getTrainingImages();
        LoadNetworkFromFileNumpy loader = new LoadNetworkFromFileNumpy(digitNetworkPath);
        NetworkWithObjects oldNetwork = loader.load();

//        List<Integer> sizes = new ArrayList<>();
//        sizes.add(784); sizes.add(30); sizes.add(10);
//        NetworkWithObjects newNetwork = NetworkWithObjects.initializeNetworkRandom(sizes);

        System.out.println("Old Network");
        for (int i = 0; i < 2; i++) {
            testImageWithErrorDebugging(oldNetwork, trainingImages.get(i+30));
        }
//        System.out.println("New Network");
//        for (int i = 0; i < 20; i++) {
//            testImage(newNetwork, trainingImages.get(i));
//        }
    }

    public static void testImage(NetworkWithObjects n, Image i) throws Exception
    {
        ImageToNetwork.feedImageToNetwork(n, i, false);
        int actualValue = i.getActualDigit();
        int networkValue = ImageToNetwork.determineResultFromNetwork(n);
        System.out.println("Actual value: " + actualValue);
        System.out.println("Network value: " + networkValue);
        if (actualValue != networkValue) {
            System.out.println("Mismatch!");
            System.out.println(i);
            System.out.println("End mismatch");
        }
    }

    public static void testImageWithErrorDebugging(NetworkWithObjects n, Image i) throws Exception
    {
        ImageToNetwork.feedImageToNetwork(n, i, true);
        int actualValue = i.getActualDigit();
        int networkValue = ImageToNetwork.determineResultFromNetwork(n);
        System.out.println("Actual value: " + actualValue);
        System.out.println("Network value: " + networkValue);
        System.out.println(n);
    }
}
