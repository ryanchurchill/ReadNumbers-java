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
