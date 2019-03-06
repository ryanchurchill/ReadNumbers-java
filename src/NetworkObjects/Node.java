package NetworkObjects;

import java.util.*;

import Util.MyMatrixUtils;

public class Node {
    public Double bias;
    public Double currentValue;
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

    /**
     * sets currentValue and returns it based on the nodes in the prior layer
     * @return
     */
    public double feedForward()
    {
        if (!synapsesToPriorLayer.isEmpty()) {
            double newVal = 0;

            // sum weighted values from all nodes in prior layer
            for (Synapse synapse : synapsesToPriorLayer) {
                newVal += synapse.nodeInPriorLayer.feedForward() * synapse.weight;
            }
            // add bias
            newVal += bias;
            // signmoid
            newVal = MyMatrixUtils.sigmoid(newVal);

            currentValue = newVal;
        }

        return currentValue;
    }

    /*
    TEST
     */

    public String toString()
    {
        StringBuilder sb = new StringBuilder();

        if (synapsesToPriorLayer.isEmpty()) {
            return "Input Node value: " + currentValue;
        }
        else {
            sb.append("value: " + currentValue);
            sb.append(" bias: " + bias);
            sb.append(" synapses to prior layer: ");
            for (Synapse synapse : synapsesToPriorLayer) {
                sb.append(synapse.toString() + ", ");
            }
        }
        return sb.substring(0, sb.length() - 2);
    }
}
