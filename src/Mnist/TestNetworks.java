package Mnist;

import Network.NetworkUtils.*;
import Network.*;
import Util.MyMathUtils;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
//import com.google.common.collect.Lists;

import java.text.DecimalFormat;
import java.util.*;

public class TestNetworks {

    static final String digitNetworkPath = "E:\\dev\\ai-az\\ReadNumbers-Java\\data\\wb_ryan_python.txt";
    static final String simpleNetworkPath = "E:\\dev\\ai-az\\ReadNumbers-Java\\data\\sandbox_ryan_python.txt";
    static final String oneExamplePath = "E:\\dev\\ai-az\\ReadNumbers-Java\\data\\python_after_one_example.txt";

//    static final String digitNetworkPath = "/Users/rchurchill/udemy/NN-Book/ReadNumbers-java/data/wb_ryan_python.txt";
//    static final String simpleNetworkPath = "/Users/rchurchill/udemy/NN-Book/ReadNumbers-java/data/sandbox_ryan_python.txt";
//    static final String oneExamplePath = "/Users/rchurchill/udemy/NN-Book/ReadNumbers-java/data/python_after_one_example.txt";

    public static void main(String[] args) throws Exception
    {
//        sigmoidPerfomanceTest();
//        for (int i = 0; i < 10; i++) {
//            addingArraysPerfTest();
//            Thread.sleep(200);
//        }
//        for (int i = 0; i < 10; i++) {
//            addingListsPerfTest();
//            Thread.sleep(200);
//        }

        for (int i = 0; i < 10; i++) {
            performanceTestingFFObjects();
            performanceTestingArrays();
        }
//        matrixPerformanceTest();



    }

    /*
    single dot product of 1 million in 4 MS
    250 million / sec

    1,000 dot products of 1,000 in 5 MS
    200,000 / sec
     */
    public static void matrixPerformanceTest()
    {
        int count = 1000;
        int size = 1000;

        RealMatrix matrix1 = initializeGaussianMatrix(size, 1);
        RealVector v1 = matrix1.getColumnVector(0);
        RealMatrix matrix2 = initializeGaussianMatrix(1, size);
        RealVector v2 = matrix2.getRowVector(0);

        long startTime = System.currentTimeMillis();

        for (int i = 0; i < count; i++) {
            v1.dotProduct(v2);
        }

        long endTime = System.currentTimeMillis();
        long duration = (endTime - startTime);
        double rate = ((double) count / (double)duration) * 1000.0;
        DecimalFormat formatter = new DecimalFormat("#,###");
        System.out.format("Took %s milliseconds", formatter.format(duration));
        System.out.println();
        System.out.format("Rate: %s / sec", formatter.format(rate));
    }

    private static RealMatrix initializeGaussianMatrix(int rows, int columns)
    {
        // createRealMatrix takes an array that's the transpose of what i'd expect
        // the first dimension is y, the second dimension is x

        Random r = new Random();
        double[][] nums = new double[rows][columns];
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < columns; x++) {
                nums[y][x] = r.nextGaussian();
            }
        }

        return MatrixUtils.createRealMatrix(nums);
    }

    public static void sigmoidPerfomanceTest()
    {
        int count = 10000000;

        long startTime = System.currentTimeMillis();

        // sigmoid: 2,327 milliseconds or 4,297,379 / sec
        for (int i = 1; i <= count; i++) {
            MyMathUtils.sigmoid(i);
        }

        long endTime = System.currentTimeMillis();
        long duration = (endTime - startTime);
        double rate = ((double) count / (double)duration) * 1000.0;
        DecimalFormat formatter = new DecimalFormat("#,###");
        System.out.format("Took %s milliseconds", formatter.format(duration));
        System.out.println();
        System.out.format("Rate: %s / sec", formatter.format(rate));
    }

    /*
    Took 13 milliseconds for 100,000
    Rate: 7,692,308 / sec
     */
    public static void addingListsPerfTest()
    {
        System.out.println("Adding lists...");
        int count = 100000;
        List<Double> list1 = generateRandomList(count);
        List<Double> list2 = generateRandomList(count);

        long startTime = System.currentTimeMillis();

        addLists(list1, list2);

        long endTime = System.currentTimeMillis();
        long duration = (endTime - startTime);
        double rate = ((double) count / (double)duration) * 1000.0;
        DecimalFormat formatter = new DecimalFormat("#,###");
        System.out.format("Took %s milliseconds", formatter.format(duration));
        System.out.println();
        System.out.format("Rate: %s / sec", formatter.format(rate));
        System.out.println();
    }

    /*
    Took 1-2 milliseconds for 100,000
     */
    public static void addingArraysPerfTest()
    {
        System.out.println("Adding arrays...");
        int count = 100000;
        double[] arr1 = generateRandomArray(count);
        double[] arr2 = generateRandomArray(count);

        long startTime = System.currentTimeMillis();

        addArrays2(arr1, arr2);

        long endTime = System.currentTimeMillis();
        long duration = (endTime - startTime);
        double rate = ((double) count / (double)duration) * 1000.0;
        DecimalFormat formatter = new DecimalFormat("#,###");
        System.out.format("Took %s milliseconds", formatter.format(duration));
        System.out.println();
        System.out.format("Rate: %s / sec", formatter.format(rate));
        System.out.println();
    }

    private static List<Double> generateRandomList(int size)
    {
        Random r = new Random();
        List<Double> ret = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            ret.add(r.nextGaussian());
        }
        return ret;
    }

    private static List<Double> addLists(List<Double> list1, List<Double> list2)
    {
        List<Double> ret = new ArrayList<>();
        for (int i = 0; i < list1.size(); i++) {
            ret.add(list1.get(i) + list2.get(1));
        }
        return ret;
    }

    private static double[] generateRandomArray(int size)
    {
        Random r = new Random();
        double[] ret = new double[size];
        for (int i = 0; i < size; i++) {
            ret[i] = r.nextGaussian();
        }
        return ret;
    }

    private static double[] addArrays(double[] arr1, double[] arr2)
    {
        double[] ret = new double[arr1.length];
        for (int i = 0; i < arr1.length; i++) {
            ret[i] = arr1[i] + arr2[i];
        }
        return ret;
    }

    private static List<Double> addArrays2(double[] arr1, double[] arr2)
    {
        List<Double> ret = new ArrayList<>();
        for (int i = 0; i < arr1.length; i++) {
            ret.add(arr1[i] + arr2[i]);
        }
        return ret;
    }

    public static void performanceTestingFFObjects() throws Exception
    {
        System.out.println("Testing NetworkWithObjects...");

        ArrayList<Integer> sizes = new ArrayList<>();
        sizes.add(784); sizes.add(30); sizes.add(10);
        NetworkWithObjects n = NetworkWithObjects.initializeNetworkRandom(sizes);

        List<Image> allTrainingImages = ReadMnist.getTrainingImages();

//        for (int i = 0; i < 10; i++) {
//            Image img = allTrainingImages.get(i);
//            long duration = timeSingleFeedForward(na, img);
//            System.out.println(duration);
//        }

        /*
        Duration: 985 milliseconds
        Avg rate: 10,152.284263959391 / sec
         */

        int count = allTrainingImages.size();
        long duration = timeBatchFeedForwardObjects(n, allTrainingImages.subList(0, count));
        double rate = ((double) count / (double)duration) * 1000.0;
        DecimalFormat formatter = new DecimalFormat("#,###");
        System.out.format("Duration: %s milliseconds", formatter.format(duration));
        System.out.println();
        System.out.format("Avg rate: %s / sec", formatter.format(rate));
        System.out.println();
//        System.out.format("initializeTimer: %s", formatter.format(Globals.initializeTimer.getTimeInMs()));
//        System.out.println();
//        System.out.format("ffTimer: %s", formatter.format(Globals.ffTimer.getTimeInMs()));
//        System.out.println();
//        System.out.format("layerTimer: %s", formatter.format(Globals.layerTimer.getTimeInMs()));
//        System.out.println();
//        System.out.format("nodeTimer: %s", formatter.format(Globals.nodeTimer.getTimeInMs()));
//        System.out.println();
//        System.out.format("synapseTimer: %s", formatter.format(Globals.synapseTimer.getTimeInMs()));
//        System.out.println();
    }

    /*
    12k/sec
     */
    public static void performanceTestingArrays() throws Exception
    {
        System.out.println("Testing network with arrays...");

        ArrayList<Integer> sizes = new ArrayList<>();
        sizes.add(784); sizes.add(30); sizes.add(10);
        NetworkWithArrays n = new NetworkWithArrays(sizes);

        List<Image> allTrainingImages = ReadMnist.getTrainingImages();

        int count = allTrainingImages.size();
        long duration = timeBatchFeedForwardArrays(n, allTrainingImages.subList(0, count));
        DecimalFormat formatter = new DecimalFormat("#,###");
        double rate = ((double) count / (double)duration) * 1000.0;
        System.out.format("Duration: %s milliseconds", formatter.format(duration));
        System.out.println();
        System.out.format("Avg rate: %s / sec", formatter.format(rate));
        System.out.println();
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

//    public static void testLightlyTrainedNetwork() throws Exception
//    {
//        LoadNetworkFromFile loader = new LoadNetworkFromFile(oneExamplePath);
//        NetworkWithObjects na = loader.load();
//        System.out.println(na);
//
//        List<Image> allTrainingImages = ReadMnist.getTrainingImages();
//        List<List<Image>> imageBatches = ListUtils.partition(allTrainingImages, 1);
//        ImageToNetwork.trainNetworkOnImageBatchArray(na, imageBatches.get(0));
//
//        System.out.println(na);
//    }

    public static void testSimpleNetwork() throws Exception
    {
//        LoadNetworkFromFile loader = new LoadNetworkFromFile(simpleNetworkPath);
//        NetworkWithObjects simpleNetwork = loader.load();
//
//        // TE1: 1, 2 -> 0, 1
//        List<Double> input1 = new ArrayList<>();
//        input1.add(1.0);
//        input1.add(2.0);
//        List<Double> desiredOutput1 = new ArrayList<>();
//        desiredOutput1.add(0.0);
//        desiredOutput1.add(1.0);
//        TrainingExample te1 = new TrainingExample(input1, desiredOutput1);
//
//        // TE2: 2, 1 -> 1, 0
//        List<Double> input2 = new ArrayList<>();
//        input2.add(2.0);
//        input2.add(1.0);
//        List<Double> desiredOutput2 = new ArrayList<>();
//        desiredOutput2.add(1.0);
//        desiredOutput2.add(0.0);
//        TrainingExample te2 = new TrainingExample(input2, desiredOutput2);
//
//        System.out.println("Initial network:");
//        System.out.println(simpleNetwork);
//
////        simpleNetwork.feedForwardRecursive(input2);
////        simpleNetwork.calculateErrors(desiredOutput2);
////        simpleNetwork.updateWeightsAndBiasesAfterSingleTrainingExample();
//
//        List<TrainingExample> miniBatch = new ArrayList<>();
//        miniBatch.add(te1);
//        miniBatch.add(te2);
//        simpleNetwork.trainWithMiniBatch(miniBatch);
//
//        System.out.println("Network after one mini-batch:");
//        System.out.println(simpleNetwork);
    }

    public static void testDigits(String[] args) throws Exception {
        List<Image> trainingImages = ReadMnist.getTrainingImages();
        LoadNetworkFromFile loader = new LoadNetworkFromFile(digitNetworkPath);
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
