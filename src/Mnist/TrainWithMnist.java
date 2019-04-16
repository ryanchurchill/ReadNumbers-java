package Mnist;

import Network.NetworkWithObjects;
import com.google.common.collect.Lists;

import java.text.DecimalFormat;
import java.util.*;

public class TrainWithMnist {
    NetworkWithObjects n;

    public static void main(String[] args) throws Exception
    {
        TrainWithMnist twn = new TrainWithMnist();
        twn.train();
    }

    public void train() throws Exception
    {
        // "params"
        int miniBatchSize = 10;
        int epochs = 10;
//        int learningRate = 3; // hard-coded in nn

        ArrayList<Integer> sizes = new ArrayList<>();
        sizes.add(784); sizes.add(30); sizes.add(10);
        n = NetworkWithObjects.initializeNetworkRandom(sizes);

        List<Image> allTrainingImages = ReadMnist.getTrainingImages();
        List<List<Image>> imageBatches = Lists.partition(allTrainingImages, miniBatchSize);
        // TEMP to speed things up
//        imageBatches = imageBatches.subList(0, 1000);

        List<Image> testImages = allTrainingImages.subList(0, 10000);

        System.out.println("Epoch -1");
        outputBatchTest(testImages);

        for (int epochCounter = 0; epochCounter < epochs; epochCounter++) {
            // TODO: randomize training image order
            System.out.println("Starting Epoch " + epochCounter);
            long startTime = System.currentTimeMillis();
            int batchCounter = 0;
            for (List<Image> miniBatch : imageBatches) {
                ImageToNetwork.trainNetworkOnImageBatch(n, miniBatch);
//                printWithTimestamp("Completed batch " + batchCounter);
                batchCounter++;
            }
            long endTime = System.currentTimeMillis();
            long duration = endTime - startTime;
//            System.out.println("Completed Epoch " + epochCounter + " in " + duration + " ms");
            DecimalFormat formatter = new DecimalFormat("#,###");
            System.out.format("Completed Epoch %d in %s milliseconds", epochCounter, formatter.format(duration));
            System.out.println();


            outputBatchTest(testImages);
        }
    }

    private void printWithTimestamp(String s)
    {
        System.out.println(Calendar.getInstance().getTime() + ": " + s);
    }

    private void outputBatchTest(List<Image> images) throws Exception
    {
        int numCorrect = 0;
        for (Image i : images) {
            if (testImage(i)) {
                numCorrect++;
            }
        }
        System.out.println(numCorrect + " / " + images.size());
    }

    private boolean testImage(Image i) throws Exception
    {
        ImageToNetwork.feedImageToNetwork(n, i);
        int actualValue = i.getActualDigit();
        int networkValue = ImageToNetwork.determineResultFromNetwork(n);
        return (actualValue == networkValue);
    }
}
