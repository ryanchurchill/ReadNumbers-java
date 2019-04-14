package Network.NetworkObjects;

import java.util.Random;

public class Synapse {
    // structural / permanent
    public Node nodeInPriorLayer;
    public Node nodeInNextLayer;

    // changes slowly as network learns
    public double weight;

    /*
    CONSTRUCTORS AND FACTORIES
     */

    public Synapse(double _weight, Node _nodeInPriorLayer, Node _nodeInNextLayer)
    {
        weight = _weight;
        nodeInPriorLayer = _nodeInPriorLayer;
        nodeInNextLayer = _nodeInNextLayer;
        nodeInPriorLayer.synapsesToNextLayer.add(this);
        nodeInNextLayer.synapsesFromPriorLayer.add(this); // TODO: this is duplicate..
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

//    public void setSynapsesToNextLayer()
//    {
//        nodeInPriorLayer.synapsesToNextLayer.add(this);
//    }
}
