package Network;

import Exceptions.ValidationException;
import Network.Learning.TrainingExample;
import Util.MyMathUtils;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Deprecated
 */
public class NetworkWithArrays {
    // TODO: consider making this more object oriented (e.g. having Neuron and/or NeuronLayer objects, instead of
    // relying on indexes.)

    // size of number of neurons at each layer
    // 0 -> input layer
    // sizes.length - 1 -> desiredOutput layer
    List<Integer> sizes;

    // TODO: lists -> arrays

    // vector at index 0 are the biases at layer 1
    List<RealVector> biases;

    // matrix at index 0 is the weights from layer 0 to layer 1
    List<RealMatrix> weights;

    boolean debugging = false;

    public NetworkWithArrays(List<Integer> _sizes)
    {
        sizes = _sizes;
        initializeBiases();
        initializeWeights();

//        if (debugging) {
//            System.out.println("Biases:");
//            MyMathUtils.printRealMatrices(biases);
//            System.out.println("Weights:");
//            MyMathUtils.printRealMatrices(weights);
//        }
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

    private RealVector initializeGaussianVector(int count)
    {
        Random r = new Random();
        double[] nums = new double[count];
        for (int i = 0; i < count; i++) {
            nums[i] = r.nextGaussian();
        }

        return MatrixUtils.createRealVector(nums);
    }

    private void initializeBiases()
    {
        biases = new ArrayList<>();
        for (int neuronLayer=1; neuronLayer<sizes.size(); neuronLayer++) {
            // row count is size of neuron layer
            // column count is 1
            biases.add(initializeGaussianVector(sizes.get(neuronLayer)));
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
     * @return an (m, 1) matrix of the values coming out of the desiredOutput layer
     */
    public RealVector feedForward(RealVector input)
    {
        for (int startingLayer = 0; startingLayer < getNumLayers() - 1; startingLayer ++) {
            RealVector biasesAtNextLayer = biases.get(startingLayer);
            RealMatrix weightsBetweenLayers = weights.get(startingLayer);
            input = MyMathUtils.sigmoid(weightsBetweenLayers.operate(input).add(biasesAtNextLayer));
        }

        return input;
    }

    public void trainWithMiniBatch(List<TrainingExample> miniBatch) throws ValidationException
    {
        // zero out lists of matrices for deltas in weights and biases
        RealVector[] biasNablas = new RealVector[biases.size()];
        for (int i=0; i < biases.size(); i++) {
            biasNablas[i] = MyMathUtils.zeroes(biases.get(0).getDimension());
        }

        RealMatrix[] weightNablas = new RealMatrix[weights.size()];
        for (int i=0; i < weights.size(); i++) {
            weightNablas[i] = MyMathUtils.zeroes(weights.get(0).getRowDimension(), weights.get(0).getColumnDimension());
        }

        // for each mini-batch, get values to alter nablas by
        for (TrainingExample te : miniBatch) {
            RealVector[] biasNablasDelta = new RealVector[biases.size()];
            RealMatrix[] weightNablasDelta = new RealMatrix[weights.size()];
            backpropSingleExample(te, biasNablasDelta, weightNablasDelta);


        }

        // now edit weights and biases based on nablas
    }

    /**
     *
     * @param te
     * @param biasNablasOut how much to add to each bias
     * @param weightNablasOut how much to add to each weight
     */
    public void backpropSingleExample(TrainingExample te, RealVector[] biasNablasOut, RealMatrix[] weightNablasOut)
    {
        // TODO: codify in TE
        RealVector input = MatrixUtils.createRealVector(te.input);
        RealVector desiredOutput = MatrixUtils.createRealVector(te.desiredOutput);

        /*
         1. feed-forward to get Zs and Activations at each node
          */

        // activations begins at first layer
        RealVector[] activations = new RealVector[weights.size() + 1];
        activations[0] = input;
        // zs begins at second layer (index 0 is second layer)
        RealVector[] zs = new RealVector[weights.size()];

        // though i is 0, we are starting at second layer, based on activations from prior layer
        for (int i = 0; i < weights.size(); i++) {
            RealVector biasesAtThisLayer = biases.get(i);
            RealMatrix weightsToThisLayer = weights.get(i);
            RealVector z = weightsToThisLayer.operate(activations[i]).add(biasesAtThisLayer);
            // reminder: activations begins at first layer, zs begins at second layer
            zs[i] = z;
            activations[i+1] = MyMathUtils.sigmoid(z);
        }

        /*
        2. calculate the error vector at the output layer
        use BP1: (a - y) * (sigmoidPrime(z))
         */
        RealVector delta = activations[weights.size()].subtract(desiredOutput).ebeMultiply(MyMathUtils.sigmoidPrime(zs[weights.size() - 1]));

        // error vector can set last layer of nablas
        biasNablasOut[biases.size() - 1] = delta;
        weightNablasOut[weights.size() - 1] = MyMathUtils.multiplyVectors(delta, activations[weights.size()]);

        /*
        3. backpropagate error and set nablas at prior layers
         */
        for (int i = weights.size() - 2; i >= 0; i--) {
            delta = weights.get(i).transpose().operate(delta).ebeMultiply(MyMathUtils.sigmoidPrime(zs[i]));
            biasNablasOut[i] = delta;
            weightNablasOut[i] = MyMathUtils.multiplyVectors(delta, activations[i+1]);
        }
    }

    public static void main(String[] args) {
        ArrayList<Integer> sizes = new ArrayList<Integer>();
        sizes.add(2); sizes.add(3); sizes.add(1);
        NetworkWithArrays n = new NetworkWithArrays(sizes);

        double[] input = new double[2];
        input[0] = .3;
        input[1] = .7;
        RealVector inputVector = MatrixUtils.createRealVector(input);
        RealVector outputVector = n.feedForward(inputVector);
        System.out.println("Result:");
        System.out.println(outputVector);
    }
}
