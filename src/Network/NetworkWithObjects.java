package Network;

import java.util.ArrayList;
import java.util.List;

import Exceptions.ValidationException;
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

    public void feedForward(List<Double> input) throws ValidationException
    {
        if (layers.isEmpty()) {
            throw new ValidationException("cannot feed forward empty network");
        }

        // initialize neurons in first layer with input
        getInputLayer().initializeWithInputData(input);

        // call recursive method on output layer
        getOutputLayer().feedForward();
    }

    /**
     * Update the network so every node has a NodeLearningProperties object with its error
     * Must be called after feedForward
     * @param desiredOutput
     */
    public void calculateErrors(List<Double> desiredOutput) throws ValidationException
    {
        // calculate the error at the output layer
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
     * Must be called after feedForward and calculateErrors
     * TODO: make the flow more cohesive and work with mini batches
     * @throws ValidationException
     */
    public void updateWeightsAndBiases() throws ValidationException
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
        n.feedForward(input);
        System.out.println(n);

    }
}
