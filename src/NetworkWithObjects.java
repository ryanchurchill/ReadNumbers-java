import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import Exceptions.ValidationException;
import NetworkObjects.*;

/**
 * General rule: we initialize the network from left to right.
 * When we feed forward the network, code flow is recursive from right to left (leading to propogation from left to right).
 * TODO: don't know yet how back prop will work
 */
public class NetworkWithObjects {
    List<Layer> layers;

    /*
    CONSTRUCTORS AND FACTORIES
     */

    private NetworkWithObjects()
    {
        layers = new ArrayList<>();
    }

    /**
     * Initializes Neural Network with random gaussian distribution of weights and biases
     * @param sizes ordered list of size of each layer
     * @return
     */
    public static NetworkWithObjects initializeNetworkRandom(List<Integer> sizes) throws ValidationException
    {
        NetworkWithObjects network = new NetworkWithObjects();
        Layer layer = null;
        for (int i = 0; i < sizes.size(); i++) {
            layer = Layer.initializeLayerRandom(sizes.get(i), i, layer);
            network.layers.add(layer);
        }

        return network;
    }

    /*
    OTHER
     */

    public void feedForward(List<Double> input) throws ValidationException
    {
        if (layers.isEmpty()) {
            throw new ValidationException("cannot feed forward empty network");
        }

        // initialize neurons in first layer with input
        getInputLayer().initializeWithInputData(input);

        // call recursive method on output layer
        getOutputLayer().feedForward();
    }

    public Layer getInputLayer()
    {
        return layers.get(0);
    }

    public Layer getOutputLayer()
    {
        return layers.get(layers.size() - 1);
    }

    /*
    TEST
     */

    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        for (Layer layer : layers) {
            sb.append(layer.toString() + System.lineSeparator());
        }
        return sb.toString();
    }

    public static void main(String[] args) throws ValidationException
    {
        ArrayList<Integer> sizes = new ArrayList<>();
        sizes.add(2); sizes.add(3); sizes.add(1);
//        sizes.add(2); sizes.add(1);
        NetworkWithObjects n = NetworkWithObjects.initializeNetworkRandom(sizes);
        System.out.println(n);

        ArrayList<Double> input = new ArrayList<>();
        input.add(.3); input.add(.7);
        n.feedForward(input);
        System.out.println(n);

    }
}
