package Network;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

import Exceptions.ValidationException;
import Network.Learning.TrainingExample;
import Network.NetworkObjects.*;
import Util.MyMathUtils;

/**
 * General rule: we initialize the network from left to right.
 * When we feed forward the network, code flow is recursive from right to left (leading to propogation from left to right).
 * TODO: don't know yet how back prop will work
 */
public class NetworkWithObjects {
    List<Layer> layers;

    public double learningRate = 3;

    /*
    CONSTRUCTORS AND FACTORIES
     */

    private NetworkWithObjects()
    {
        layers = new ArrayList<>();
    }

    private NetworkWithObjects(List<Layer> _layers) { layers = _layers; }

    /**
     * Initializes Neural Network with random gaussian distribution of weights and biases
     * @param sizes ordered list of size of each layer
     * @return
     */
    public static NetworkWithObjects initializeNetworkRandom(List<Integer> sizes) throws ValidationException
    {
        NetworkWithObjects network = new NetworkWithObjects();
        Layer layer = null;
        for (int i = 0; i < sizes.size(); i++) {
            layer = Layer.initializeLayerRandom(sizes.get(i), i, layer);
            network.layers.add(layer);
        }

        return network;
    }

    /**
     *
     * @param sizes number of neurons in each layer
     * @param biases biases[0] is the list of biases for layer 1 (second layer)
     * @param weights weights[0][0] is the weights from all neurons in layer 0 to the first neuron in layer 1
     * @return
     */
    public static NetworkWithObjects initializeFromData(
            List<Integer> sizes, List<List<Double>> biases, List<List<List<Double>>> weights) throws ValidationException
    {
        // TODO: more validations

        ArrayList<Layer> layers = new ArrayList<>();
        Layer inputLayer = Layer.initializeInputLayer(sizes.get(0));
        layers.add(inputLayer);
        Layer priorLayer = inputLayer;

        for (int layerIndex = 1; layerIndex < sizes.size(); layerIndex++) {
            Layer nextLayer = Layer.buildLayerFromData(
                    layerIndex, priorLayer, biases.get(layerIndex - 1), weights.get(layerIndex - 1));
            layers.add(nextLayer);
            priorLayer = nextLayer;
        }

        return new NetworkWithObjects(layers);
    }

    /*
    OTHER
     */

    public Layer getInputLayer()
    {
        return layers.get(0);
    }

    public Layer getOutputLayer()
    {
        return layers.get(layers.size() - 1);
    }

    public double[] getOutputValues()
    {
        Layer outputLayer = getOutputLayer();
        double[] ret = new double[outputLayer.nodeCount()];
        for (int i = 0; i < outputLayer.nodeCount(); i++) {
            ret[i] = outputLayer.nodes.get(i).currentValue;
        }
        return ret;
    }

    public int getLayerCount()
    {
        return layers.size();
    }

    /**
     * Deprecated! Iterative is much faster!
     * @param input
     * @throws ValidationException
     */
    public void feedForwardRecursive(List<Double> input) throws ValidationException
    {
        if (layers.isEmpty()) {
            throw new ValidationException("cannot feed forward empty network");
        }

        // skip clearing first layer since it will be initialized with new data
        for (int i = 1; i < getLayerCount(); i++) {
            Layer l = layers.get(i);
            for (Node n : l.nodes) {
                n.currentValue = null;
            }
        }

        // initialize neurons in first layer with input
        getInputLayer().initializeWithInputData(input);

        // call recursive method on desiredOutput layer
        getOutputLayer().feedForward();
    }

    public void feedForwardIterative(List<Double> input) throws ValidationException {
        if (layers.isEmpty()) {
            throw new ValidationException("cannot feed forward empty network");
        }

        // initialize neurons in first layer with input
        getInputLayer().initializeWithInputData(input);

        for (int i = 1; i < getLayerCount(); i++) {
            Layer l = layers.get(i);
            for (Node n : l.nodes) {
                double val = 0;
                for (Synapse s : n.synapsesFromPriorLayer) {
                    val += s.nodeInPriorLayer.currentValue * s.weight;
                }
                n.weightedInput = val;
                n.currentValue = MyMathUtils.sigmoid(val);
            }
        }
    }

    public void feedForwardParallel2(List<Double> input) throws ValidationException {
        if (layers.isEmpty()) {
            throw new ValidationException("cannot feed forward empty network");
        }

        // initialize neurons in first layer with input
        getInputLayer().initializeWithInputData(input);

        for (int i = 1; i < getLayerCount(); i++) {
            Layer l = layers.get(i);
            l.nodes.parallelStream().forEach(n -> {
                double val = 0;
                for (Synapse s : n.synapsesFromPriorLayer) {
                    val += s.nodeInPriorLayer.currentValue * s.weight;
                }
                n.weightedInput = val;
                n.currentValue = MyMathUtils.sigmoid(val);
            });
        }
    }

    /**
     * Slow and bad!
     * @param input
     * @throws ValidationException
     */
    public void feedForwardParallel(List<Double> input) throws ValidationException {
        if (layers.isEmpty()) {
            throw new ValidationException("cannot feed forward empty network");
        }

        // initialize neurons in first layer with input
        getInputLayer().initializeWithInputData(input);

        for (int i = 1; i < getLayerCount(); i++) {
            Layer l = layers.get(i);
            ExecutorService service = Executors.newFixedThreadPool(50);
            for (Node n : l.nodes) {
                service.execute(new FeedForwardNodeTask(n));
            }
            // this will get blocked until all task finish
            service.shutdown();
            try {
                service.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    private static class FeedForwardNodeTask implements Runnable {
        Node n;

        protected FeedForwardNodeTask(Node _n)
        {
            n = _n;
        }

        public void run()
        {
            double val = 0;
            for (Synapse s : n.synapsesFromPriorLayer) {
                val += s.nodeInPriorLayer.currentValue * s.weight;
            }
            n.weightedInput = val;
            n.currentValue = MyMathUtils.sigmoid(val);
        }
    }

    private void clearAllValues()
    {
        for (Layer l : layers) {
            for (Node n : l.nodes) {
                n.currentValue = null;
            }
        }
    }

    /**
     * Update the network so every node has a NodeLearningProperties object with its error
     * Must be called after feedForwardRecursive
     * @param desiredOutput
     */
    public void calculateErrors(List<Double> desiredOutput) throws ValidationException
    {
        // calculate the error at the desiredOutput layer
        Layer outputLayer = getOutputLayer();
        int nodeIndex = 0;
        for (Node n : outputLayer.nodes) {
            n.setErrorForOutputNode(desiredOutput.get(nodeIndex));
            nodeIndex++;
        }

        // propogate error back to the remaining layers
        for (int layerIndex = getLayerCount() - 2; layerIndex >= 0; layerIndex--) {
            Layer currentLayer = layers.get(layerIndex);
            for (Node n : currentLayer.nodes) {
                n.setErrorForNonOutputNode();
            }
        }
    }

    /**
     * Deprecated! Use trainWithMiniBatch
     * Must be called after feedForwardRecursive and calculateErrors
     * @throws ValidationException
     */
    public void updateWeightsAndBiasesAfterSingleTrainingExample() throws ValidationException
    {
        for (Layer l : layers)
        {
            for (Node n : l.nodes) {
                // nableBias == error
                if (!(n.bias == null)) {
                    n.bias = (n.bias - (learningRate * n.error));
                }
                for (Synapse s : n.synapsesToNextLayer) {
                    double nablaWeight = s.nodeInPriorLayer.currentValue * s.nodeInNextLayer.error;
                    s.weight = (s.weight - (learningRate * nablaWeight));
                }
            }
        }
    }

    public void trainWithMiniBatch(List<TrainingExample> miniBatch) throws ValidationException
    {
//        resetAllNablas();
        int counter = 0;
        for (TrainingExample te : miniBatch) {
//            feedForwardRecursive(te.input);
            feedForwardIterative(te.input);
            calculateErrors(te.desiredOutput);
            updateAllNablas();
//            System.out.println("Processed example " + counter++);
        }

        updateWeightsAndBiasesAfterProcessingMiniBatch(miniBatch.size());
    }

    /**
     * We go left to right, but order we iterate doesn't matter
     * Shouldn't need to be called!
     */
    private void resetAllNablas()
    {
        for (Layer l : layers)
        {
            for (Node n : l.nodes) {
                n.biasNablaForMiniBatch = 0;
                for (Synapse s : n.synapsesToNextLayer) {
                    s.weightNablaForMiniBatch = 0;
                }
            }
        }
    }

    /**
     * We go left to right, but order we iterate doesn't matter
     */
    private void updateAllNablas()
    {
        for (Layer l : layers)
        {
            for (Node n : l.nodes) {
                n.updateBiasNablaForMiniBatch();
                for (Synapse s : n.synapsesToNextLayer) {
                    s.updateWeightNablaForMiniBatch();
                }
            }
        }
    }

    private void updateWeightsAndBiasesAfterProcessingMiniBatch(int miniBatchSize)
    {
        for (Layer l : layers)
        {
            for (Node n : l.nodes) {
                if (l.layerNum > 0) {
                    n.updateBiasFromNabla(learningRate, miniBatchSize);
                }
                for (Synapse s : n.synapsesToNextLayer) {
                    s.updateWeightFromNabla(learningRate, miniBatchSize);
                }
            }
        }
    }

    /**
     * synapsesFromPriorLayer are set as we build up the network. Once that's done, this method can be called
     * to populate synapsesToNextLayer
     */
//    public void setSynapsesToNextLayer()
//    {
//        Layer outputLayer = getOutputLayer();
//    }

    /*
    TEST
     */

    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        for (Layer layer : layers) {
            sb.append(layer.toString() + System.lineSeparator());
        }
        return sb.toString();
    }

    public static void main(String[] args) throws ValidationException
    {
        ArrayList<Integer> sizes = new ArrayList<>();
        sizes.add(2); sizes.add(3); sizes.add(1);
//        sizes.add(2); sizes.add(1);
        NetworkWithObjects n = NetworkWithObjects.initializeNetworkRandom(sizes);
        System.out.println(n);

        ArrayList<Double> input = new ArrayList<>();
        input.add(.3); input.add(.7);
        n.feedForwardRecursive(input);
        System.out.println(n);

    }
}
