package Mnist;

import Network.NetworkUtils.*;
import Network.*;

import java.util.*;

public class TestExistingNetwork {

    static final String networkFilePath = "E:\\dev\\ai-az\\ReadNumbers-Java\\data\\wb_ryan_python.txt";

    public static void main(String[] args) throws Exception {
        List<Image> trainingImages = ReadMnist.getTrainingImages();
        Image firstImage = ReadMnist.getTrainingImages().get(0);
        LoadNetworkFromFileNumpy loader = new LoadNetworkFromFileNumpy(networkFilePath);
        NetworkWithObjects n = loader.load();
    }

    public static void feedImageToNetwork(NetworkWithObjects n, Image i)
    {

    }
}
