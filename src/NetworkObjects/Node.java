package NetworkObjects;

import java.util.*;

public class Node {
    public Double bias;
    public double currentValue;
    public List<Synapse> synapsesToPriorLayer;

    /*
    CONSTRUCTORS AND FACTORIES
     */

    private Node()
    {
        synapsesToPriorLayer = new ArrayList<>();
    }

    public static Node initializeNodeRandom(Layer priorLayer)
    {
        Node n = new Node();

        // if we have a prior layer, we build a synapse from each node in that layer to this node
        // if no prior layer no need for bias either
        if (priorLayer != null) {
            Random r = new Random();
            n.bias = r.nextGaussian();
            for (Node node : priorLayer.nodes) {
                n.synapsesToPriorLayer.add(Synapse.initializeSynapseRandom(node));
            }
        }

        return n;
    }

    /*
    OTHER
     */

    public String toString()
    {
        StringBuilder sb = new StringBuilder();

        if (synapsesToPriorLayer.isEmpty()) {
            return "Input Node";
        }
        else {
            sb.append("bias: " + bias);
            sb.append(" synapes to prior layer: ");
            for (Synapse synapse : synapsesToPriorLayer) {
                sb.append(synapse.toString() + ", ");
            }
        }
        return sb.substring(0, sb.length() - 2).toString();
    }
}
