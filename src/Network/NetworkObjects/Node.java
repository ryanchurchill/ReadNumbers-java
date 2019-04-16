package Network.NetworkObjects;

import java.util.*;

import Exceptions.ValidationException;
import Util.MyMathUtils;

public class Node {
    // structural / permanent
    public List<Synapse> synapsesFromPriorLayer;
    public List<Synapse> synapsesToNextLayer;

    // changes slowly, and only when learning
    public Double bias;

    // changes with each data point
    public Double currentValue;
    public double error;
    public double weightedInput; // AKA Z
    public double biasNablaForMiniBatch = 0;

    /*
    CONSTRUCTORS AND FACTORIES
     */

    public Node()
    {
        synapsesFromPriorLayer = new ArrayList<>();
        synapsesToNextLayer = new ArrayList<>();
    }

    public Node(double _bias)
    {
        this();
        bias = _bias;
    }

    public static Node initializeNodeRandom(Layer priorLayer)
    {
        Node n = new Node();

        // if we have a prior layer, we build a synapse from each node in that layer to this node
        // if no prior layer no need for bias either
        if (priorLayer != null) {
            Random r = new Random();
            n.bias = r.nextGaussian();
            for (Node priorLayerNode : priorLayer.nodes) {
                Synapse.initializeSynapseRandom(priorLayerNode, n);
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
    public double feedForward() throws ValidationException
    {
        if (currentValue != null) {
            return currentValue;
        }

        if (!synapsesFromPriorLayer.isEmpty()) {
            double newVal = 0;

            // sum weighted values from all nodes in prior layer
            for (Synapse synapse : synapsesFromPriorLayer) {
                newVal += synapse.nodeInPriorLayer.feedForward() * synapse.weight;
            }
            // add bias
            newVal += bias;

            // store Z since we'll need it later
            weightedInput = newVal;

            // signmoid
            newVal = MyMathUtils.sigmoid(newVal);

            currentValue = newVal;
        }

        return currentValue;
    }

    public void setErrorForOutputNode(double expectedValue) throws ValidationException
    {
        if (!synapsesToNextLayer.isEmpty()) {
            throw new ValidationException("method must be called on desiredOutput node!");
        }
        // for desiredOutput layer, use BP1: (a - y) * (sigmoidPrime(z))
        error = (currentValue - expectedValue)
                * MyMathUtils.sigmoidPrime(weightedInput);
    }

    public void setErrorForNonOutputNode() throws ValidationException
    {
        if (synapsesToNextLayer.isEmpty()) {
            throw new ValidationException("method must not be called on desiredOutput node!");
        }

        // for other layers, use BP2 to feed error back from next layer
        double newError = 0;

        // sum weighted errors from all nodes in next layer
        for (Synapse synapse : synapsesToNextLayer) {
            newError += synapse.nodeInNextLayer.error * synapse.weight;
        }
        // multiply by sigmoid prime of this node's Z
        newError *= MyMathUtils.sigmoidPrime(weightedInput);

        error = newError;
    }

    public void updateBiasNablaForMiniBatch()
    {
        biasNablaForMiniBatch += error;
    }

    /**
     * TODO: refactor calculation with updateWeightFromNabla
     * @param learningRate
     * @param miniBatchSize
     */
    public void updateBiasFromNabla(double learningRate, int miniBatchSize)
    {
        bias = bias - (learningRate / miniBatchSize * biasNablaForMiniBatch);
        biasNablaForMiniBatch = 0;
    }

    /*
    TEST
     */

    public String toString()
    {
        StringBuilder sb = new StringBuilder();

        sb.append("value: " + currentValue);
        sb.append(" z : " + weightedInput);
        sb.append(" error: " + error);
        if (!synapsesFromPriorLayer.isEmpty()) {

            sb.append(" bias: " + bias);
            sb.append(" synapses to prior layer: ");
            for (Synapse synapse : synapsesFromPriorLayer) {
                sb.append(synapse.toString() + ", ");
            }
        }
        sb.append(" synapses to next layer: ");
        for (Synapse synapse : synapsesToNextLayer) {
            sb.append(synapse.toString() + ", ");
        }
        return sb.substring(0, sb.length() - 2);
    }
}
