import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Network {
    // TODO: consider making this more object oriented (e.g. having Neuron and/or NeuronLayer objects, instead of
    // relying on indexes.)

    // size of number of neurons at each layer
    // 0 -> input layer
    // sizes.length - 1 -> output layer
    List<Integer> sizes;

    // matrix at index 0 are the biases at layer 1
    List<RealMatrix> biases;

    // matrix at index 0 is the weights from layer 0 to layer 1
    List<RealMatrix> weights;

    boolean debugging = true;

    public Network(List<Integer> _sizes)
    {
        sizes = _sizes;
        initializeBiases();
        initializeWeights();

        if (debugging) {
            System.out.println("Biases:");
            MyMatrixUtils.printRealMatrices(biases);
            System.out.println("Weights:");
            MyMatrixUtils.printRealMatrices(weights);
        }
    }

    public int getNumLayers()
    {
        return sizes.size();
    }

    private RealMatrix initializeGaussianMatrix(int rows, int columns)
    {
        // createRealMatrix takes an array that's the transpose of what i'd expect
        // the first dimension is y, the second dimension is x

        Random r = new Random();
        double[][] nums = new double[rows][columns];
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < columns; x++) {
                nums[y][x] = r.nextGaussian();
            }
        }

        return MatrixUtils.createRealMatrix(nums);
    }

    private void initializeBiases()
    {
        biases = new ArrayList<>();
        for (int neuronLayer=1; neuronLayer<sizes.size(); neuronLayer++) {
            // row count is size of neuron layer
            // column count is 1
            biases.add(initializeGaussianMatrix(sizes.get(neuronLayer), 1));
        }
    }

    private void initializeWeights()
    {
        weights = new ArrayList<>();
        for (int neuronLayer = 0; neuronLayer < sizes.size() - 1; neuronLayer++) {
            // row count number is size of next neuron layer
            // column count is size of current neuron layer
            int rowCount = sizes.get(neuronLayer+1);
            int columnCount = sizes.get(neuronLayer);
            weights.add(initializeGaussianMatrix(rowCount, columnCount));
        }
    }

    /**
     *
     * @param input an (n, 1) matrix of the values going to the input layer
     * @return an (m, 1) matrix of the values coming out of the output layer
     */
    private RealVector feedForward(RealVector input)
    {
        for (int startingLayer = 0; startingLayer < getNumLayers() - 1; startingLayer ++) {
            RealMatrix biasesAtNextLayer = biases.get(startingLayer);
            RealMatrix weightsBetweenLayers = weights.get(startingLayer);

            int outputSize = sizes.get(startingLayer+1);
            double[] outputArr = new double[outputSize];
            // index of neuron at the next layer
            for (int neuronIndex = 0; neuronIndex < outputSize; neuronIndex ++) {
                double val = input.dotProduct(weightsBetweenLayers.getRowVector(neuronIndex));
                val += biasesAtNextLayer.getEntry(neuronIndex, 0);
                val = MyMatrixUtils.sigmoid(val);
                outputArr[neuronIndex] = val;
            }

            input = MatrixUtils.createRealVector(outputArr);
            if (debugging) {
                System.out.println("Output at layer " + Integer.toString(startingLayer + 1) + ":");
                System.out.println(input);
            }
        }

        return input;
    }

    public static void main(String[] args) {
        ArrayList<Integer> sizes = new ArrayList<Integer>();
        sizes.add(2); sizes.add(3); sizes.add(1);
        Network n = new Network(sizes);

        double[] input = new double[2];
        input[0] = 1;
        input[1] = 1;
        RealVector inputVector = MatrixUtils.createRealVector(input);
        RealVector outputVector = n.feedForward(inputVector);
        System.out.println("Result:");
        System.out.println(outputVector);
    }
}
