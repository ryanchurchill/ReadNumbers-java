package Mnist;

import Network.Learning.TrainingExample;
import Network.NetworkUtils.*;
import Network.*;
import com.google.common.collect.Lists;

import java.util.*;

public class TestNetworks {

    static final String digitNetworkPath = "E:\\dev\\ai-az\\ReadNumbers-Java\\data\\wb_ryan_python.txt";
    static final String simpleNetworkPath = "E:\\dev\\ai-az\\ReadNumbers-Java\\data\\sandbox_ryan_python.txt";
    static final String oneExamplePath = "E:\\dev\\ai-az\\ReadNumbers-Java\\data\\python_after_one_example.txt";

    public static void main(String[] args) throws Exception
    {
        performanceTestingObjects();
        performanceTestingArrays();
    }

    public static void performanceTestingObjects() throws Exception
    {
        System.out.println("Testing NetworkWithObjects...");

        ArrayList<Integer> sizes = new ArrayList<>();
        sizes.add(784); sizes.add(30); sizes.add(10);
        NetworkWithObjects n = NetworkWithObjects.initializeNetworkRandom(sizes);

        List<Image> allTrainingImages = ReadMnist.getTrainingImages();

//        for (int i = 0; i < 10; i++) {
//            Image img = allTrainingImages.get(i);
//            long duration = timeSingleFeedForward(n, img);
//            System.out.println(duration);
//        }

        int count = 1000;
        long duration = timeBatchFeedForwardObjects(n, allTrainingImages.subList(0, count));
        double rate = ((double) count / (double)duration) * 1000.0;
        System.out.println("Avg rate: " + rate + " / sec");
    }

    public static void performanceTestingArrays() throws Exception
    {
        System.out.println("Testing network with arrays...");

        ArrayList<Integer> sizes = new ArrayList<>();
        sizes.add(784); sizes.add(30); sizes.add(10);
        NetworkWithArrays n = new NetworkWithArrays(sizes);

        List<Image> allTrainingImages = ReadMnist.getTrainingImages();

        int count = 1000;
        long duration = timeBatchFeedForwardArrays(n, allTrainingImages.subList(0, count));
        double rate = ((double) count / (double)duration) * 1000.0;
        System.out.println("Avg rate: " + rate + " / sec");
    }

    public static long timeSingleFeedForward(NetworkWithObjects n, Image i) throws Exception
    {
        long startTime = System.currentTimeMillis();
        ImageToNetwork.feedImageToNetwork(n, i);
        long endTime = System.currentTimeMillis();
        return (endTime - startTime);
    }

    public static long timeBatchFeedForwardObjects(NetworkWithObjects n, List<Image> images) throws Exception
    {
        long startTime = System.currentTimeMillis();
        for (Image img : images) {
            ImageToNetwork.feedImageToNetwork(n, img);
        }
        long endTime = System.currentTimeMillis();
        return (endTime - startTime);
    }

    public static long timeBatchFeedForwardArrays(NetworkWithArrays n, List<Image> images) throws Exception
    {
        long startTime = System.currentTimeMillis();
        for (Image img : images) {
            ImageToNetwork.feedImageToNetworkWithArrays(n, img);
        }
        long endTime = System.currentTimeMillis();
        return (endTime - startTime);
    }

    public static void testLightlyTrainedNetwork() throws Exception
    {
        LoadNetworkFromFileNumpy loader = new LoadNetworkFromFileNumpy(oneExamplePath);
        NetworkWithObjects n = loader.load();
        System.out.println(n);

        List<Image> allTrainingImages = ReadMnist.getTrainingImages();
        List<List<Image>> imageBatches = Lists.partition(allTrainingImages, 1);
        ImageToNetwork.trainNetworkOnImageBatch(n, imageBatches.get(0));

        System.out.println(n);
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
        ImageToNetwork.feedImageToNetwork(n, i);
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
        ImageToNetwork.feedImageToNetwork(n, i);
        int actualValue = i.getActualDigit();
        int networkValue = ImageToNetwork.determineResultFromNetwork(n);
        System.out.println("Actual value: " + actualValue);
        System.out.println("Network value: " + networkValue);
        System.out.println(n);
    }
}
