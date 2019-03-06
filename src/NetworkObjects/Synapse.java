package NetworkObjects;

import java.util.Random;

public class Synapse {
    public double weight;
    public Node nodeInPriorLayer;

    /*
    CONSTRUCTORS AND FACTORIES
     */

    private Synapse(double _weight, Node _nodeInPriorLayer)
    {
        weight = _weight;
        nodeInPriorLayer = _nodeInPriorLayer;
    }

    public static Synapse initializeSynapseRandom(Node _nodeInPriorLayer)
    {
        Random r = new Random();
        double weight = r.nextGaussian();
        return new Synapse(weight, _nodeInPriorLayer);
    }

    /*
    OTHER
     */

    public String toString()
    {
        return "weight: " + weight;
    }
}
