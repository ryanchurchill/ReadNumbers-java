package Network.NetworkObjects;

import java.util.*;

import Exceptions.ValidationException;
import Util.MyMathUtils;

public class Node {
    // structural / permanent
    public List<Synapse> synapsesFromPriorLayerList;
    public List<Synapse> synapsesToNextLayerList;
    // for performance
    public Synapse[] synapsesFromPriorLayerArray;
    public Synapse[] synapsesToNextLayerArray;

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
        synapsesFromPriorLayerList = new ArrayList<>();
        synapsesToNextLayerList = new ArrayList<>();
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
            for (Node priorLayerNode : priorLayer.nodeList) {
                Synapse.initializeSynapseRandom(priorLayerNode, n);
            }
        }

        return n;
    }

    /*
    OTHER
     */

    public void getReadyToProcess()
    {
        synapsesToNextLayerArray = new Synapse[synapsesToNextLayerList.size()];
        synapsesToNextLayerArray = synapsesToNextLayerList.toArray(synapsesToNextLayerArray);
        synapsesToNextLayerList.clear(); // sanity check that we stop using the list

        synapsesFromPriorLayerArray = new Synapse[synapsesFromPriorLayerList.size()];
        synapsesFromPriorLayerArray = synapsesFromPriorLayerList.toArray(synapsesFromPriorLayerArray);
        synapsesFromPriorLayerList.clear(); // sanity check that we stop using the list
    }

    /**
     * Deprecated
     * sets currentValue and returns it based on the nodeList in the prior layer
     * @return
     */
    public double feedForward() throws ValidationException
    {
        if (currentValue == null && !(synapsesFromPriorLayerArray.length == 0)) {
            double newVal = 0;

            // sum weighted values from all nodeList in prior layer
            for (Synapse synapse : synapsesFromPriorLayerArray) {
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
        if (!(synapsesToNextLayerArray.length == 0)) {
            throw new ValidationException("method must be called on desiredOutput node!");
        }
        // for desiredOutput layer, use BP1: (a - y) * (sigmoidPrime(z))
        error = (currentValue - expectedValue)
                * MyMathUtils.sigmoidPrime(weightedInput);
    }

    public void setErrorForNonOutputNode() throws ValidationException
    {
        if (synapsesToNextLayerArray.length == 0) {
            throw new ValidationException("method must not be called on desiredOutput node!");
        }

        // for other layers, use BP2 to feed error back from next layer
        double newError = 0;

        // sum weighted errors from all nodeList in next layer
        for (Synapse synapse : synapsesToNextLayerArray) {
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
        if (!synapsesFromPriorLayerList.isEmpty()) {

            sb.append(" bias: " + bias);
            sb.append(" synapses to prior layer: ");
            for (Synapse synapse : synapsesFromPriorLayerList) {
                sb.append(synapse.toString() + ", ");
            }
        }
        sb.append(" synapses to next layer: ");
        for (Synapse synapse : synapsesToNextLayerList) {
            sb.append(synapse.toString() + ", ");
        }
        return sb.substring(0, sb.length() - 2);
    }
}
