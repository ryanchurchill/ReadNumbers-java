package Network.NetworkObjects;

import java.util.Random;

public class Synapse {
    // structural / permanent
    public Node nodeInPriorLayer;
    public Node nodeInNextLayer;

    // changes slowly as network learns
    public double weight;

    // changes per training example
    public double weightNablaForMiniBatch = 0;

    /*
    CONSTRUCTORS AND FACTORIES
     */

    public Synapse(double _weight, Node _nodeInPriorLayer, Node _nodeInNextLayer)
    {
        weight = _weight;
        nodeInPriorLayer = _nodeInPriorLayer;
        nodeInNextLayer = _nodeInNextLayer;
        nodeInPriorLayer.synapsesToNextLayer.add(this);
        nodeInNextLayer.synapsesFromPriorLayer.add(this);
    }

    public static Synapse initializeSynapseRandom(Node _nodeInPriorLayer, Node _nodeInNextLayer)
    {
        Random r = new Random();
        double weight = r.nextGaussian();
        Synapse s = new Synapse(weight, _nodeInPriorLayer, _nodeInNextLayer);
        return s;
    }

    /*
    OTHER
     */

    public String toString()
    {
        return "weight: " + weight;
    }

    public void updateWeightNablaForMiniBatch()
    {
        double nabla = nodeInPriorLayer.currentValue * nodeInNextLayer.error;
        weightNablaForMiniBatch += nabla;
    }

    /**
     * TODO: refactor calculation with updateBiasFromNabla
     * @param learningRate
     * @param miniBatchSize
     */
    public void updateWeightFromNabla(double learningRate, int miniBatchSize)
    {
        weight = weight - (learningRate / miniBatchSize * weightNablaForMiniBatch);
        weightNablaForMiniBatch = 0;
    }
}
