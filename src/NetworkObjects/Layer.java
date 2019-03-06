package NetworkObjects;

import Exceptions.ValidationException;

import java.util.*;

public class Layer {
    public int layerNum;
    public List<Node> nodes;

    /*
    CONSTRUCTORS AND FACTORIES
     */

    private Layer(int _layerNum)
    {
        layerNum = _layerNum;
        nodes = new ArrayList<>();
    }

    public static Layer initializeLayerRandom(int nodeCount, int layerNum, Layer priorLayer) throws ValidationException
    {
        if (layerNum > 0 && priorLayer == null) {
            throw new ValidationException("Cannot initialize layer with num > 0 without priorLayer");
        }
        if (nodeCount <= 0) {
            throw new ValidationException("layer size must be < 0");
        }

        Layer layer = new Layer(layerNum);

        for (int i = 0; i < nodeCount; i++) {
            Node n = Node.initializeNodeRandom(priorLayer);
            layer.nodes.add(n);
        }
        return layer;
    }

    /*
    OTHER
     */

    /**
     * First layer only
     * @param input
     */
    public void initializeWithInputData(List<Double> input) throws ValidationException
    {
        // validate
        if (!isInputLayer()) {
            throw new ValidationException("layerNum must be 0 to initialize with input data");
        }
        if (input.size() != nodes.size()) {
            throw new ValidationException("input size does not match layer size");
        }

        // initialize
        for (int i = 0; i < input.size(); i++) {
            nodes.get(i).currentValue = input.get(i);
        }
    }

    public boolean isInputLayer()
    {
        return (layerNum == 0);
    }

    public void feedForward()
    {
        for (Node node : nodes) {
            node.feedForward();
        }
    }

    /*
    TEST
     */

    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("Layer: " + layerNum + System.lineSeparator());
        for (Node n : nodes) {
            sb.append(n.toString() + System.lineSeparator());
        }

        return sb.toString();
    }
}
