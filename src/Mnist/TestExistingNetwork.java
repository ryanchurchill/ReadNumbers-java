package Mnist;

import Network.NetworkUtils.*;
import Network.*;

import java.util.*;

public class TestExistingNetwork {

    static final String networkFilePath = "E:\\dev\\ai-az\\ReadNumbers-Java\\data\\wb_ryan_python.txt";

    public static void main(String[] args) throws Exception {
        List<Image> trainingImages = ReadMnist.getTrainingImages();
        LoadNetworkFromFileNumpy loader = new LoadNetworkFromFileNumpy(networkFilePath);
        NetworkWithObjects oldNetwork = loader.load();

        List<Integer> sizes = new ArrayList<>();
        sizes.add(784); sizes.add(30); sizes.add(10);
        NetworkWithObjects newNetwork = NetworkWithObjects.initializeNetworkRandom(sizes);

        System.out.println("Old Network");
        for (int i = 0; i < 20; i++) {
            testImage(oldNetwork, trainingImages.get(i+30));
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
}
