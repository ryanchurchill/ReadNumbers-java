package Mnist;

import Network.NetworkWithArrays;
import Network.NetworkWithObjects;
//import com.google.common.collect.Lists;
import org.apache.commons.collections4.*;
import org.apache.commons.math3.linear.RealVector;

import java.text.DecimalFormat;
import java.util.*;

/*
Performance 4/15/19
- Object Oriented (no arrays)
- Iterative feed-forward
- 18 seconds for full epoch (of full set)
- Uses 10% of CPU
- Python project takes 6 seconds for full epoch, uses 50% of CPU
- Python seems to be 10x faster at feed_forward
 */

public class TrainWithMnist {
    NetworkWithArrays na;
    NetworkWithObjects no;

    public static void main(String[] args) throws Exception
    {
        TrainWithMnist twn = new TrainWithMnist();
//        twn.trainWithArrays();
//        twn.trainWithObjects(3);
        twn.trainWithBoth();
    }

    public void trainWithBoth() throws Exception
    {
//        System.out.println("Training With Objects");
//        trainWithObjects(1);

        ArrayList<Integer> sizes = new ArrayList<>();
        sizes.add(784); sizes.add(30); sizes.add(10);

        // need to build both networks at the same time so training of one doesn't affect the other
        no = NetworkWithObjects.initializeNetworkRandom(sizes);
        na = new NetworkWithArrays(no);

        System.out.println("Loading images...");
        List<Image> allTrainingImages = ReadMnist.getTrainingImages();
        List<Image> testImages = allTrainingImages.subList(0, 1000);

        System.out.println("Testing...");
        System.out.println("Epoch -1");
        outputBatchTestObjects(testImages);

//        System.out.println("Training with Arrays");
//        trainWithArrays(1);


        outputBatchTestArray(testImages);
    }

    public void trainWithArrays(int epochs) throws Exception
    {
        // "params"
        int miniBatchSize = 10;
//        int learningRate = 3; // hard-coded in nn

        ArrayList<Integer> sizes = new ArrayList<>();
        sizes.add(784); sizes.add(30); sizes.add(10);
//        na = new NetworkWithArrays(sizes);
        na = new NetworkWithArrays(no);

        List<Image> allTrainingImages = ReadMnist.getTrainingImages();
        List<List<Image>> imageBatches = ListUtils.partition(allTrainingImages, miniBatchSize);
        // TEMP to speed things up
//        imageBatches = imageBatches.subList(0, 1000);

        List<Image> testImages = allTrainingImages.subList(0, 10000);

        System.out.println("Epoch -1");
        outputBatchTestArray(testImages);

//        for (int epochCounter = 0; epochCounter < epochs; epochCounter++) {
//            // TODO: randomize training image order
//            System.out.println("Starting Epoch " + epochCounter);
//            long startTime = System.currentTimeMillis();
//            int batchCounter = 0;
//            for (List<Image> miniBatch : imageBatches) {
//                ImageToNetwork.trainNetworkOnImageBatchArray(na, miniBatch);
////                printWithTimestamp("Completed batch " + batchCounter);
//                batchCounter++;
//            }
//            long endTime = System.currentTimeMillis();
//            long duration = endTime - startTime;
////            System.out.println("Completed Epoch " + epochCounter + " in " + duration + " ms");
//            DecimalFormat formatter = new DecimalFormat("#,###");
//            System.out.format("Completed Epoch %d in %s milliseconds", epochCounter, formatter.format(duration));
//            System.out.println();
//
//            outputBatchTestArray(testImages);
//        }
    }

    public void trainWithObjects(int epochs) throws Exception
    {
        // "params"
        int miniBatchSize = 10;
//        int learningRate = 3; // hard-coded in nn

        ArrayList<Integer> sizes = new ArrayList<>();
        sizes.add(784); sizes.add(30); sizes.add(10);
        no = NetworkWithObjects.initializeNetworkRandom(sizes);

        List<Image> allTrainingImages = ReadMnist.getTrainingImages();
        List<List<Image>> imageBatches = ListUtils.partition(allTrainingImages, miniBatchSize);
        // TEMP to speed things up
//        imageBatches = imageBatches.subList(0, 1000);

        List<Image> testImages = allTrainingImages.subList(0, 10000);

        System.out.println("Epoch -1");
        outputBatchTestObjects(testImages);

        for (int epochCounter = 0; epochCounter < epochs; epochCounter++) {
            // TODO: randomize training image order
            System.out.println("Starting Epoch " + epochCounter);
            long startTime = System.currentTimeMillis();
            int batchCounter = 0;
            for (List<Image> miniBatch : imageBatches) {
                ImageToNetwork.trainNetworkOnImageBatch(no, miniBatch);
//                printWithTimestamp("Completed batch " + batchCounter);
                batchCounter++;
            }
            long endTime = System.currentTimeMillis();
            long duration = endTime - startTime;
//            System.out.println("Completed Epoch " + epochCounter + " in " + duration + " ms");
            DecimalFormat formatter = new DecimalFormat("#,###");
            System.out.format("Completed Epoch %d in %s milliseconds", epochCounter, formatter.format(duration));
            System.out.println();

            outputBatchTestObjects(testImages);
        }
    }

    private void printWithTimestamp(String s)
    {
        System.out.println(Calendar.getInstance().getTime() + ": " + s);
    }

    private void outputBatchTestArray(List<Image> images) throws Exception
    {
        int numCorrect = 0;
        for (Image i : images) {
            if (testImageArray(i)) {
                numCorrect++;
            }
        }
        System.out.println(numCorrect + " / " + images.size());
    }

    private void outputBatchTestObjects(List<Image> images) throws Exception
    {
        int numCorrect = 0;
        for (Image i : images) {
            if (testImageObjects(i)) {
                numCorrect++;
            }
        }
        System.out.println(numCorrect + " / " + images.size());
    }

    private boolean testImageArray(Image i) throws Exception
    {
        RealVector output = na.feedForward(i.getPixelsForArrayNetwork());
        int actualValue = i.getActualDigit();
        int networkValue = ImageToNetwork.determineResultFromVector(output);
        return (actualValue == networkValue);
    }

    private boolean testImageObjects(Image i) throws Exception
    {
        no.feedForward(i.getPixelsForNetwork());
        int actualValue = i.getActualDigit();
        int networkValue = ImageToNetwork.determineResultFromNetwork(no);
        return (actualValue == networkValue);
    }
}
