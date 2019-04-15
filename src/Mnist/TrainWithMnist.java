package Mnist;

import Network.NetworkWithObjects;
import com.google.common.collect.Lists;

import java.util.*;

public class TrainWithMnist {
    NetworkWithObjects n;

    public void train() throws Exception
    {
        // "params"
        int miniBatchSize = 10;
        int epochs = 1;
//        int learningRate = 3; // hard-coded in nn

        ArrayList<Integer> sizes = new ArrayList<>();
        sizes.add(784); sizes.add(30); sizes.add(10);
        n = NetworkWithObjects.initializeNetworkRandom(sizes);

        List<Image> allTrainingImages = ReadMnist.getTrainingImages();
        List<List<Image>> imageBatches = Lists.partition(allTrainingImages, miniBatchSize);

        List<Image> testImages = allTrainingImages.subList(0, 1000);

        for (int epochCounter = 0; epochCounter < epochs; epochCounter++) {
            System.out.println("Starting Epoch " + epochCounter);
            int batchCounter = 0;
            for (List<Image> miniBatch : imageBatches) {
                ImageToNetwork.trainNetworkOnImageBatch(n, miniBatch);
                printWithTimestamp("Completed batch " + batchCounter++);
            }
            printWithTimestamp("Completed Epoch " + epochCounter);
            epochCounter++;

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

    public static void main(String[] args) throws Exception
    {
        TrainWithMnist twn = new TrainWithMnist();
        twn.train();
    }
}
