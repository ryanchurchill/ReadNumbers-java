package Network.NetworkObjects;

import java.util.*;

import Util.MyMatrixUtils;

public class Node {
    public Double bias;
    public Double currentValue;
    public List<Synapse> synapsesFromPriorLayer;

    /*
    CONSTRUCTORS AND FACTORIES
     */

    public Node()
    {
        synapsesFromPriorLayer = new ArrayList<>();
    }

    public Node(double _bias, List<Synapse> _synapsesToPriorLayer)
    {
        bias = _bias;
        synapsesFromPriorLayer = _synapsesToPriorLayer;
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
                n.synapsesFromPriorLayer.add(Synapse.initializeSynapseRandom(node));
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
        if (!synapsesFromPriorLayer.isEmpty()) {
            double newVal = 0;

            // sum weighted values from all nodes in prior layer
            for (Synapse synapse : synapsesFromPriorLayer) {
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

        if (synapsesFromPriorLayer.isEmpty()) {
            return "Input Node value: " + currentValue;
        }
        else {
            sb.append("value: " + currentValue);
            sb.append(" bias: " + bias);
            sb.append(" synapses to prior layer: ");
            for (Synapse synapse : synapsesFromPriorLayer) {
                sb.append(synapse.toString() + ", ");
            }
        }
        return sb.substring(0, sb.length() - 2);
    }
}
