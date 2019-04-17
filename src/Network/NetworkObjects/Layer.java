package Network.NetworkObjects;

import Exceptions.ValidationException;

import java.util.*;

public class Layer {
    public int layerNum;
    public List<Node> nodeList;
    public Node[] nodeArray;

    /*
    CONSTRUCTORS AND FACTORIES
     */

    private Layer(int _layerNum) {
        layerNum = _layerNum;
        nodeList = new ArrayList<>();
    }

    public static Layer initializeLayerRandom(int nodeCount, int layerNum, Layer priorLayer) throws ValidationException {
        if (layerNum > 0 && priorLayer == null) {
            throw new ValidationException("Cannot initialize layer with num > 0 without priorLayer");
        }
        if (nodeCount <= 0) {
            throw new ValidationException("layer size must be < 0");
        }

        Layer layer = new Layer(layerNum);

        for (int i = 0; i < nodeCount; i++) {
            Node n = Node.initializeNodeRandom(priorLayer);
            layer.nodeList.add(n);
        }
        return layer;
    }

    public static Layer initializeInputLayer(int nodeCount) throws ValidationException
    {
        if (nodeCount <= 0) {
            throw new ValidationException("node count must be < 0");
        }

        Layer layer = new Layer(0);

        for (int i = 0; i < nodeCount; i++) {
            Node n = new Node();
            layer.nodeList.add(n);
        }
        return layer;
    }

    /**
     *
     * @param layerNum
     * @param priorLayer
     * @param biasesForThisLayer
     * @param weightsFromPriorLayer [0]: weights from all neurons in the previous layer to the first neuron in this layer
     * @return
     * @throws ValidationException
     */
    public static Layer buildLayerFromData(
            int layerNum,
            Layer priorLayer,
            List<Double> biasesForThisLayer,
            List<List<Double>> weightsFromPriorLayer
    ) throws ValidationException {
        if (priorLayer == null || biasesForThisLayer == null || weightsFromPriorLayer == null) {
            throw new ValidationException("Cannot initialize non-input layer without all data");
        }
        if (biasesForThisLayer.size() != weightsFromPriorLayer.size()) {
            throw new ValidationException("Weights and Biases must have same count");
        }
        int nodeCount = biasesForThisLayer.size();
        if (nodeCount <= 0) {
            throw new ValidationException("layer size must be < 0");
        }
        if (weightsFromPriorLayer.get(0).size() != priorLayer.nodeCount()) { // TODO: verify all weights lists, not just first one
            throw new ValidationException("Each list in weightsFromPriorLayer must match nodeCount of prior layer");
        }

        Layer layer = new Layer(layerNum);

        for (int nodeIndexThisLayer = 0; nodeIndexThisLayer < nodeCount; nodeIndexThisLayer++)
        {
            double bias = biasesForThisLayer.get(nodeIndexThisLayer);
            List<Synapse> synapsesFromPriorLayer = new ArrayList<>();
            Node n = new Node(bias);
            for (int nodeIndexPriorLayer = 0; nodeIndexPriorLayer < priorLayer.nodeCount(); nodeIndexPriorLayer++) {
                double weight = weightsFromPriorLayer.get(nodeIndexThisLayer).get(nodeIndexPriorLayer);
                new Synapse(weight, priorLayer.nodeList.get(nodeIndexPriorLayer), n);
            }
            layer.nodeList.add(n);

        }

        return layer;
    }

    /*
    OTHER
     */

    public void getReadyToProcess()
    {
        nodeArray = new Node[nodeList.size()];
        nodeArray = nodeList.toArray(nodeArray);
        nodeList.clear(); // to sanity check we aren't still using the list

        for (Node n : nodeArray) {
            n.getReadyToProcess();
        }
    }

    /**
     * First layer only - start of feeding data through the network
     * @param input
     */
    public void setNodeValuesWithInputData(List<Double> input) throws ValidationException
    {
        // validate
        if (!isInputLayer()) {
            throw new ValidationException("layerNum must be 0 to initialize with input data");
        }
        if (input.size() != nodeArray.length) {
            throw new ValidationException("input size does not match layer size");
        }

        // initialize
        for (int i = 0; i < input.size(); i++) {
            nodeArray[i].currentValue = input.get(i);
        }
    }

    public boolean isInputLayer()
    {
        return (layerNum == 0);
    }

    /**
     * @Deprecated
     * @throws ValidationException
     */
    public void feedForward() throws ValidationException
    {
        for (Node node : nodeArray) {
            node.feedForward();
        }
    }

    public int nodeCount()
    {
        // temporary hack to get around list/array discrepancy
        return Math.max(nodeList.size(), nodeArray.length);
    }

    /*
    TEST
     */

    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("Layer: " + layerNum + " Count: " + nodeCount() + System.lineSeparator());
        for (Node n : nodeList) {
            sb.append(n.toString() + System.lineSeparator());
        }

        return sb.toString();
    }
}
