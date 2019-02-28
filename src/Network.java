import org.apache.commons.math3.linear.RealMatrix;

import java.util.List;

public class Network {
    // size of number of neurons at each layer
    // 0 -> input layer
    // sizes.length - 1 -> output layer
    List<Integer> sizes;

    // matrix at index 0 are the biases at layer 1
    List<RealMatrix> biases;

    // matrix at index 0 is the weights from layer 0 to layer 1
    List<RealMatrix> weights;
}
