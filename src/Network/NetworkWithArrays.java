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

    int learningRate = 3;

    // vector at index 0 are the biases at layer 1
    RealVector[] biases;

    // matrix at index 0 is the weights from layer 0 to layer 1
    RealMatrix[] weights;

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

    /**
     * Converts from NetworkWithObjects
     * @param no
     */
    public NetworkWithArrays(NetworkWithObjects no)
    {
        sizes = no.getSizes();
        biases = no.getBiasVectors();
        weights = no.getWeightMatrices();
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
        biases = new RealVector[sizes.size() - 1];
        for (int neuronLayer=1; neuronLayer<sizes.size(); neuronLayer++) {
            // row count is size of neuron layer
            // column count is 1
            biases[neuronLayer - 1] = initializeGaussianVector(sizes.get(neuronLayer));
        }
    }

    private void initializeWeights()
    {
        weights = new RealMatrix[sizes.size()-1];
        for (int neuronLayer = 0; neuronLayer < sizes.size() - 1; neuronLayer++) {
            // row count number is size of next neuron layer
            // column count is size of current neuron layer
            int rowCount = sizes.get(neuronLayer+1);
            int columnCount = sizes.get(neuronLayer);
            weights[neuronLayer] = initializeGaussianMatrix(rowCount, columnCount);
        }
    }

    /**
     * TODO: remove duplicate logic between here and SGD
     * @param input an (n, 1) matrix of the values going to the input layer
     * @return an (m, 1) matrix of the values coming out of the desiredOutput layer
     */
    public RealVector feedForward(RealVector input)
    {
        for (int startingLayer = 0; startingLayer < getNumLayers() - 1; startingLayer ++) {
            RealVector biasesAtNextLayer = biases[startingLayer];
            RealMatrix weightsBetweenLayers = weights[startingLayer];
            input = MyMathUtils.sigmoid(
                    weightsBetweenLayers.operate(input)
                    .add(biasesAtNextLayer)
            );
        }

        return input;
    }

    public void trainWithMiniBatch(List<TrainingExample> miniBatch) throws ValidationException
    {
        // zero out lists of matrices for deltas in weights and biases
        RealVector[] biasNablas = new RealVector[biases.length];
        for (int i=0; i < biases.length; i++) {
            biasNablas[i] = MyMathUtils.zeroes(biases[i].getDimension());
        }

        RealMatrix[] weightNablas = new RealMatrix[weights.length];
        for (int i=0; i < weights.length; i++) {
            weightNablas[i] = MyMathUtils.zeroes(weights[i].getRowDimension(), weights[i].getColumnDimension());
        }

        // for each mini-batch, get values to alter nablas by
        for (TrainingExample te : miniBatch) {
            RealVector[] biasNablasDelta = new RealVector[biases.length];
            RealMatrix[] weightNablasDelta = new RealMatrix[weights.length];
            backpropSingleExample(te, biasNablasDelta, weightNablasDelta);
            biasNablas = MyMathUtils.addRealVectorArrays(biasNablas, biasNablasDelta);
            weightNablas = MyMathUtils.addRealMatrixArrays(weightNablas, weightNablasDelta);
        }

        // now edit weights and biases based on nablas
        MyMathUtils.applyGradientDescentToBiases(biases, biasNablas, learningRate, miniBatch.size());
        MyMathUtils.applyGradientDescentToWeights(weights, weightNablas, learningRate, miniBatch.size());
    }

    /**
     *
     * @param te
     * @param biasNablasOut how much to add to each bias
     * @param weightNablasOut how much to add to each weight
     */
    public void backpropSingleExample(TrainingExample te, RealVector[] biasNablasOut, RealMatrix[] weightNablasOut)
    {
        /*
         1. feed-forward to get Zs and Activations at each node
          */

        // activations begins at first layer
        RealVector[] activations = new RealVector[weights.length + 1];
        activations[0] = te.input;
        // zs begins at second layer (index 0 is second layer)
        RealVector[] zs = new RealVector[weights.length];

        // though i is 0, we are starting at second layer, based on activations from prior layer
        for (int i = 0; i < weights.length; i++) {
            RealVector biasesAtThisLayer = biases[i];
            RealMatrix weightsToThisLayer = weights[i];
            RealVector z = weightsToThisLayer.operate(activations[i]).add(biasesAtThisLayer);
            // reminder: activations begins at first layer, zs begins at second layer
            zs[i] = z;
            activations[i+1] = MyMathUtils.sigmoid(z);
        }

        /*
        2. calculate the error vector at the output layer
        use BP1: (a - y) * (sigmoidPrime(z))
         */
        RealVector delta = activations[weights.length].subtract(te.desiredOutput).ebeMultiply(MyMathUtils.sigmoidPrime(zs[weights.length - 1]));

        // error vector can set last layer of nablas
        biasNablasOut[biases.length - 1] = delta;
        weightNablasOut[weights.length - 1] = MyMathUtils.multiplyVectors(delta, activations[weights.length-1]);

        /*
        3. backpropagate error and set nablas at prior layers
         */
        for (int i = weights.length - 2; i >= 0; i--) {
            delta = weights[i+1].transpose().operate(delta).ebeMultiply(MyMathUtils.sigmoidPrime(zs[i]));
            biasNablasOut[i] = delta;
            weightNablasOut[i] = MyMathUtils.multiplyVectors(delta, activations[i]);
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
