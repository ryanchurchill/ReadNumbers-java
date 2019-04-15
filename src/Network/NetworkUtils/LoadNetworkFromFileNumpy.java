package Network.NetworkUtils;

import Exceptions.ValidationException;
import Network.NetworkWithObjects;

import java.io.*;
import java.util.*;

/*
This weird format is a hybrid of numpy printing and my own hacks. The idea is to generate a NN based on the desiredOutput
I'm getting from the working python program.

For format example, see data/wb_ryan_python.txt

# ignored
sizes
[2 3 1]
biases
[[ 0.63351642]
 [-1.22191447]
 [-1.04184972]]
[[-1.20921027]]
weights
[[-1.00689467e+00 -2.08261026e-01]     <-- weights from all neurons in the first layer to the first neuron in the second layer
 [-8.24438470e-01  4.51818679e-01]
 [ 1.25037762e+00 -1.49095980e+00]]
[[ 1.98546993e+00 -9.09564584e-01 -1.97794334e+00]]
 */

public class LoadNetworkFromFileNumpy {
    enum States{
        START, SIZES, BIASES, WEIGHTS
    };
    States state;

    String filePath;

//    NetworkWithObjects network;

    List<Integer> sizes;

    List<List<Double>> biases; // biases[0] is the list of biases for layer 1 (second layer)

    List<List<List<Double>>> weights; // weights[0][0] is the weights from all neurons in layer 0 to the first neuron in layer 1

    public LoadNetworkFromFileNumpy(String _filePath)
    {
        state = States.START;
        filePath = _filePath;

        biases = new ArrayList<>();
        weights = new ArrayList<>();
    }

    public NetworkWithObjects load() throws Exception
    {
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        String line;
        while ((line = br.readLine()) != null) {
            processLine(line);
        }

        // TODO: validate weights and biases

        return NetworkWithObjects.initializeFromData(sizes, biases, weights);
    }

    private void processLine(String line) throws ValidationException
    {
        line = line.trim();

        if (line.substring(0, 1).equals("#")) {
            return;
        }

        switch (state) {
            case START:
                if (line.equals("sizes")) {
                    state = States.SIZES;
                } else {
                    throw new ValidationException("Must start with 'sizes'");
                }
                break;
            case SIZES:
                sizes = doublesToInts(parseDoublesFromArrayString(line));
                state = States.BIASES;
                break;
            case BIASES:
                if (line.equals("biases")) {
                    break;
                } else if (line.equals("weights")) {
                    // verify that size of biases matches number of layers - 1
                    if (biases.size() != sizes.size() - 1) {
                        throw new ValidationException("Switching to weights with incorrect number of bias lists: " + biases.size());
                    }

                    state = States.WEIGHTS;
                    break;
                }

                String biasesArrayString;
                // if string starts with [[, it's the beginning of the list of biases for a layer
                if (line.substring(0, 2).equals("[[")) {
                    biases.add(new ArrayList<>());
                    // same line can start end end with double-brackets
                    if (line.substring(line.length() - 2).equals("]]")) {
                        biasesArrayString = line.substring(1, line.length() - 1);
                    } else {
                        biasesArrayString = line.substring(1);
                    }
                } else if (line.substring(line.length() - 2).equals("]]")) {
                    biasesArrayString = line.substring(0, line.length() - 1);
                } else {
                    biasesArrayString = line;
                }
                biases.get(biases.size() - 1).add(parseDoublesFromArrayString(biasesArrayString).get(0));

                break;
            case WEIGHTS:
                if (line.equals("weights")) {
                    break;
                }
                String weightsArrayString;
                // if string starts with [[, it's the beginning of the list of weights for a layer
                if (line.substring(0, 2).equals("[[")) {
                    weights.add(new ArrayList<>());
                    // same line can start end end with double-brackets
                    if (line.substring(line.length() - 2).equals("]]")) {
                        weightsArrayString = line.substring(1, line.length() - 1);
                    } else {
                        weightsArrayString = line.substring(1);
                    }
                } else if (line.substring(line.length() - 2).equals("]]")) {
                    weightsArrayString = line.substring(0, line.length() - 1);
                } else {
                    weightsArrayString = line;
                }
                weights.get(weights.size() - 1).add(parseDoublesFromArrayString(weightsArrayString));

                break;
        }
    }

    /**
     * expected string format: [-0.11027705 -8.24438470e-01  4.51818679e-01  4.39181097e-02]
     * (it's based on the python default numpy desiredOutput)
     * @param string
     * @return
     * @throws ValidationException
     */
    private static List<Double> parseDoublesFromArrayString(String string) throws ValidationException
    {
        string = string.trim();

        // validate starts and ends with []
        if (string.charAt(0) != '[') {
            throw new ValidationException("Expected arrayString to start with [");
        }
        if (string.charAt(string.length() - 1) != ']') {
            throw new ValidationException("Expected arrayString to end with ]");
        }

        List<Double> ret = new ArrayList<>();
        String[] numStrings = string.substring(1, string.length() - 1).split(" ");
        for (String numString : numStrings) {
            if (!numString.isEmpty()) {
                ret.add(Double.parseDouble(numString));
            }
        }

        return ret;
    }

    private static List<Integer> doublesToInts(List<Double> doubles)
    {
        List<Integer> ret = new ArrayList<>();
        for (Double d : doubles) {
            ret.add((int) Math.round(d));
        }
        return ret;
    }

    public static void main(String[] args) throws Exception
    {
//        testParsing();
//        String filePath = "E:\\dev\\ai-az\\ReadNumbers-Java\\data\\train-labels-idx1-ubyte";
        String filePath = "E:\\dev\\ai-az\\ReadNumbers-Java\\data\\wb_ryan_python.txt";
        LoadNetworkFromFileNumpy loader = new LoadNetworkFromFileNumpy(filePath);
        NetworkWithObjects network = loader.load();
        System.out.println(network);
    }

    public static void testParsing() throws ValidationException
    {
//        String s1 = " [-1.22191447]";
//        List<Double> l1 = LoadNetworkFromFileNumpy.parseDoublesFromArrayString(s1);

        String s2 = "[-1.00689467e+00 -2.08261026e-01  6.84330268e-01 -1.15259642e+00 -6.46968670e-01  1.35865468e-01 -2.50043490e-02 -9.35763213e-01 -1.70571390e+00 -3.73283397e-01  6.98315451e-01  2.67480062e-01 -7.18635889e-01 -9.08699229e-02 -9.26464303e-01  2.21197194e-03  4.58770587e-01  3.17664394e-01  9.86490329e-01 -7.09063928e-01  5.47658899e-01 -1.75108876e+00 -5.09563912e-01  4.72453120e-01  2.39380166e-01  9.76627093e-01 -2.48481330e-01  1.24093659e+00  4.51650863e-01 -6.60219861e-01 -2.19727534e+00  1.76028555e+00 -7.29449746e-01 -3.45640009e-01  1.98310580e-01 -1.07208688e+00  5.04627740e-01 -1.33524854e+00  3.06525355e-01  2.02924074e-01 -9.83023995e-01  3.97846631e-01 -5.08164419e-01 -9.00001463e-02  6.72765787e-01 -6.34935444e-01 -2.00133561e+00 -7.96221860e-01  6.94697499e-01  5.42048951e-01  8.70415856e-01  6.05718382e-01  7.80092816e-01  3.71615645e-01 -5.56678890e-01 -1.54919969e+00 -1.53537483e+00 -5.69468334e-02  5.68365195e-01 -3.45466967e-01  7.21479037e-01  9.65022591e-01  4.87363925e-01 -1.26524861e-03  3.82140371e-01  1.19109618e-01  3.63602318e-01  5.66166721e-01  1.18366009e+00 -5.11828617e-01 -4.10761075e-01 -2.73090869e-01  1.00261626e+00  2.05283515e+00 -6.35672599e-01 -5.46897016e-01 -7.25419150e-02  7.58499940e-01  4.52561592e-01  4.68090659e-01 -1.04367815e+00 -7.75221285e-01  7.22464448e-01  3.65721891e-01 -1.53172858e+00  1.53216854e+00 -5.00330944e-01  6.14299432e-01  1.12323222e+00  4.95643103e-01  5.90510124e-01  2.54407059e-01  3.15612962e-02  8.41144221e-01  3.16423932e-01 -2.50303046e-01  1.30155939e-02 -3.09871941e-02  3.02658152e-01 -1.32367076e-01  2.67914553e-01 -6.68791670e-01  9.21434592e-02 -1.50822296e+00  1.18746465e-01 -1.69952843e+00  7.24672888e-01  1.18642675e-02 -3.19249973e-02 -5.35229897e-01  1.01363766e-02  7.02083809e-01 -1.44697535e+00 -2.06853483e+00 -1.65706519e-01  2.58103149e-01  6.55663291e-01 -5.28817864e-01 -1.14981238e+00  1.77972878e-01  1.25470735e+00 -6.51299369e-02 -6.95696248e-01  1.87746917e-01 -9.16709873e-01 -1.20743802e-01 -2.50528203e+00  8.89745267e-01 -5.57433976e-01 -3.07845637e-01 -5.25971889e-01  5.50952162e-02  3.37967909e-01 -2.02968318e+00  1.23787304e+00 -2.61201326e-01 -1.82064040e+00 -7.73780656e-01 -2.40701531e-01 -1.20849395e+00  3.72763642e-01 -5.46220107e-01 -1.96633602e+00 -8.37521165e-02  9.18594026e-01 -1.56511494e+00  5.91748159e-02 -4.27659201e-01 -2.03812472e-01 -9.14431327e-01 -1.06343927e+00 -3.05468810e-01  1.12295927e-01 -8.76332215e-01 -8.52779640e-01 -1.67227740e+00 -9.26005815e-01 -1.44351198e+00 -7.37015186e-02 -8.38708243e-01  2.93777993e-01 -4.78505098e-01  2.55418772e-01 -9.66663099e-01  1.04917230e+00  8.99019329e-01 -1.22284195e+00 -1.11054792e+00  6.93091180e-01  1.07621719e+00 -6.85314004e-01 -1.27399918e+00  1.25391222e+00 -2.42534284e-01 -2.75838547e-01  5.99093925e-01 -3.39691566e-01  1.64240373e+00 -8.12062324e-01  8.59950498e-02 -2.20563862e-01  2.07338152e+00  4.31724414e-01  6.66144522e-01  9.77321307e-01  2.02282176e-01  3.65110067e-01  3.32629069e-01  1.01871686e+00  1.35745259e+00 -6.03010054e-01  7.51137132e-01  1.28426524e+00  1.30837374e-01 -1.12360586e-01  1.12629848e+00 -9.52865584e-02 -1.26508019e+00  3.25259294e-01  1.66266884e+00  5.33161758e-01 -3.74051409e-01 -1.28116187e+00 -1.51890110e+00  8.19826477e-02  4.98447176e-01  2.23524659e-01  1.15973905e+00  1.20280528e-01  8.53900662e-01  1.54339402e+00  3.49001208e-02  1.00331109e-01  1.77785787e+00  1.64308875e+00  6.61271050e-01 -3.29601562e-01  1.88062993e+00 -6.05403943e-01 -1.44328268e-01 -8.85607086e-02  3.28265862e+00 -1.24990638e+00 -4.27526818e-01  2.37468518e+00 -3.01233890e-01  5.78584799e-01 -5.89308753e-01 -2.52690309e-01  7.84415145e-01  2.12177392e+00  2.37722180e+00 -3.04411239e-02  1.14986938e+00  1.09646996e+00  9.80267945e-01  2.50372576e-01  1.80430667e+00  1.98513885e+00  2.48120176e+00  1.59723785e+00  8.68144978e-01  2.37303547e-01  1.07236861e+00  1.43453660e+00 -7.99063262e-01 -5.49207460e-01 -1.71946309e-01  9.94327912e-01  4.28187267e-01 -8.09789572e-01  1.73047039e+00  1.62117726e-01 -7.99184105e-03  7.64407960e-01  1.17638258e+00  8.15869731e-01  1.50015048e+00 -6.28557664e-01  1.40591797e+00  1.63278533e+00  1.64983105e+00  3.44395394e-01  1.21857112e+00 -3.67256311e-01  2.40681909e+00  4.00322532e-01  2.13129963e-01  1.73803741e+00  1.66273492e+00  1.93814437e-01  1.96479597e+00 -1.08164458e+00 -1.78322303e+00 -1.60513494e+00  9.70337469e-01  7.30049339e-01 -1.42784182e+00 -1.02874796e+00  1.99467640e+00 -5.56364455e-01  2.52974046e+00  8.65715295e-01  5.24554135e-01  1.27364306e+00  9.76973866e-01 -2.83788306e-01 -7.18611280e-01 -5.56968251e-01  8.04343132e-01  1.91697208e+00 -3.82793413e-02  5.38872108e-01  1.72501894e-01  9.17917599e-01 -4.89285497e-01  1.07524998e+00 -1.86513726e+00 -9.15873210e-01  2.62492805e-01  8.33901737e-01  1.77473685e+00 -3.98484514e-01 -2.62564516e-01  1.89599001e+00  8.09015310e-01 -1.85753808e+00 -1.09953981e+00 -1.75636668e+00  1.05246800e+00 -1.21177952e+00 -5.41291119e-01 -2.22119220e-01  8.79443004e-01  6.41650839e-01  7.47913396e-01  1.55919890e+00 -1.68538448e-01 -3.68679073e-01  2.82208489e+00  4.19016068e-02  1.15619502e+00 -1.59971762e-01 -8.85236867e-01 -6.53833666e-01  1.21296601e+00  1.47894891e+00 -9.50971076e-01 -1.29578434e-01 -2.92627586e-01  9.13067966e-01 -6.45474945e-01 -8.01847675e-01 -7.10510297e-01  1.96167479e-01 -4.19827967e-01  3.93192373e-01  4.16071510e-01  3.27989148e-01  4.85073357e-01 -3.16980178e-01  1.69867797e+00  1.30825431e+00  2.17718811e+00  1.17941382e-01  1.00704545e+00 -2.27198551e-01  1.03519723e+00 -2.74776211e+00  8.40211586e-02 -6.13736149e-01  9.26978548e-01 -1.67280481e+00 -8.97031995e-01 -4.92744707e-01  8.28538418e-01  5.22981618e-01  1.68906317e-01 -3.59528543e-01  1.57217498e+00  8.79370908e-01  1.31851670e+00 -1.02468412e+00  3.87876705e-01  5.70191741e-01 -5.72798363e-01  3.20934665e-01 -7.36861476e-01  1.48725578e+00 -1.94547007e+00 -9.17847177e-02 -9.55561072e-01 -7.65526827e-01 -7.01875307e-01 -2.74897594e+00 -8.11030976e-01 -5.12876795e-01 -1.28137610e+00 -5.76069695e-01 -7.43780855e-01 -1.01199318e+00 -2.81232279e-01 -2.29910141e-01  1.05068261e+00  1.67637828e-01 -9.88344432e-01  1.86352499e+00 -6.95817953e-01  5.06473916e-01  3.96870862e-01  2.71009522e+00 -8.32211501e-01 -1.51226471e+00 -6.52553424e-01  9.68110444e-01  2.04287410e+00  1.45155358e+00 -1.43771879e+00  2.03751672e+00  2.53688846e-01  1.76470230e+00 -3.15499200e-01  7.79977083e-01 -1.24469807e+00  9.82048169e-01 -2.03659584e+00 -2.63126454e+00 -1.72750631e-01  6.46168253e-01 -1.92711419e+00  6.89877899e-01 -3.30976591e-01 -6.77860597e-01 -1.88957537e-01  3.66339536e-02 -4.75865864e-01 -7.37304571e-01  6.32301400e-01 -8.39154069e-01  1.65001758e-01  1.54043652e+00 -4.94724443e-03  8.62727471e-01  7.38671067e-01  7.74375492e-01  6.12617755e-01 -1.15255652e+00 -7.02553314e-01 -7.73653599e-01 -8.96145415e-01 -6.01539191e-02 -8.01456444e-01  3.89171408e-01 -1.80128458e+00 -1.82372901e-01 -1.51318443e+00 -8.83506802e-01 -3.01028004e-01 -6.37536478e-01  4.67086320e-01  7.22280856e-02  1.44395346e+00 -2.77396961e-01  2.36139875e-01  7.80450500e-01 -8.11207552e-01  2.47633870e-01 -8.55995161e-02  2.01905796e-01 -7.45068614e-01  8.92037315e-01 -3.49009628e-01  1.27673052e+00 -1.40621691e+00  6.00661043e-01 -1.08446503e+00 -7.34591805e-01 -3.05032806e-02  1.48631164e+00 -1.28864439e-01 -6.33668541e-01 -2.00436633e-01 -1.65348952e+00 -8.34812290e-01 -2.14122392e-02  2.21162316e-01  6.32829234e-01 -9.95268448e-01 -4.70440441e-02 -9.80577176e-01 -2.12755203e+00 -1.45627056e+00  2.25663079e+00 -5.07108510e-01  5.95104413e-01 -2.15378060e-01 -1.80899785e+00 -1.06399656e+00 -8.85690688e-01 -8.73609530e-01  3.82802152e-02  1.06626909e+00 -7.61391495e-01 -1.54572285e-01  4.22905874e-01 -5.10582965e-01 -1.82223148e-01 -4.44941443e-01 -2.52575600e-01 -1.59206826e+00 -1.46600596e+00  1.36398813e+00 -8.63077246e-01 -1.25241630e+00 -4.67217933e-02 -1.21479843e+00 -1.24222028e+00  9.56444326e-01 -8.95465158e-01 -8.17453108e-01 -3.53369295e-01 -1.38952074e+00 -9.59136065e-01  6.30309293e-01  5.90655406e-01 -2.22671771e+00  1.89260086e-01  1.04284543e+00 -2.44506511e-01 -3.52679969e-01 -1.08201982e-01  5.34813965e-01  4.68070258e-01  9.26907338e-01 -9.46396054e-01 -3.90417350e-01 -1.69125855e+00 -2.27810606e-01 -1.30618064e-01 -1.80454687e+00 -2.22548829e+00 -5.10237318e-01 -2.05035153e-01  9.72217289e-01 -1.05412657e+00  7.40850277e-01 -1.29432308e+00 -1.46957583e-01 -1.14204196e+00 -1.97436757e+00  1.25567092e+00 -1.40464833e+00  2.27211876e+00 -6.22500537e-01  3.38936519e-01 -1.80936095e+00  5.17615933e-01  1.42755268e+00 -4.68414407e-02  1.81791322e+00 -1.12776053e+00  5.34654587e-01  1.39104737e+00 -8.27946968e-01 -1.33400016e-02 -1.22418336e+00  1.09359348e-01 -9.20744844e-02  4.04893511e-01 -1.37539196e+00 -1.28967310e+00 -1.26526996e+00 -5.66358103e-01  2.16860428e-03 -2.00715480e+00 -7.08253424e-01 -9.08618129e-01 -2.52025294e-01 -1.31251223e+00 -3.16954359e-02 -3.49760111e-01  7.21849498e-01  2.44693027e+00  8.04061030e-01 -2.91609979e-01  7.35882097e-01 -1.03879860e+00 -1.26479374e+00 -1.08908904e+00  1.70575870e-01  4.64472187e-02 -1.26549732e+00  9.62856738e-01  1.11704393e+00  1.06955645e+00 -1.34717140e+00  4.69479364e-01 -2.79247143e-01 -5.08723480e-01  2.74816049e-01 -1.35566102e-01 -1.28128807e+00  3.46090119e-01 -8.15223388e-01  1.46829808e+00 -4.31474199e-01 -3.74122617e-03  5.67904083e-01  7.66306970e-02 -9.35013347e-01 -1.08978679e+00  9.48796699e-01  7.41310566e-01  3.63156586e-01 -4.69325830e-01  3.24180134e-01  4.19517562e-01 -1.65111029e-01 -4.80733359e-01  7.50058201e-01 -1.05783025e+00 -3.84643986e-01 -3.79156000e-01 -2.72509063e-01 -4.84464462e-01 -5.31216255e-01 -1.59135779e+00 -1.61903078e+00 -4.06926340e-01 -1.31252324e+00  3.52792323e-01 -7.35427095e-01 -2.45148691e-01 -6.14035500e-01  1.00537613e+00 -7.73624932e-01  1.08965836e+00 -1.52600725e+00 -8.55900733e-01  1.41292631e-01 -9.02031257e-01  1.41761571e+00  7.08926500e-02  9.64304629e-01  1.25228969e+00  2.19653692e-01  1.64386053e+00 -4.49805348e-02  1.74040180e+00  7.66297976e-02  8.29722205e-01  1.36748150e+00 -1.21982326e+00  3.98158554e-01  2.88956586e-01 -4.50223728e-01  9.87053668e-01  1.20247216e+00 -2.48747156e-01 -5.18358203e-01  6.44738113e-01 -1.43963196e+00  1.18478894e-01 -1.00971895e+00  2.10213378e+00 -4.84082033e-01  1.14313559e+00  5.45021611e-01  3.56004013e-01 -8.03721587e-02  7.11003861e-01 -1.26440031e-01 -9.51688385e-01  1.23454041e+00 -5.11065577e-01 -1.32412269e+00  1.25969824e+00 -2.09154361e+00  9.02990738e-01  1.09217521e+00 -1.82076743e+00  1.24925345e+00 -7.45977970e-01  8.29557187e-01 -4.02792521e-01  5.91853385e-01  2.04362006e-01  7.28670960e-01 -1.73671014e-01  3.59672312e-01  4.27603314e-01  5.66727673e-01  5.35679505e-01 -6.41576680e-01 -1.94281507e+00 -1.51821776e+00 -1.21684922e-01  1.42247929e+00 -1.32664669e+00  1.30054532e+00  1.93266053e+00 -1.99123768e+00  1.44916935e+00 -8.85084646e-01 -2.24850482e+00  1.02695447e+00 -1.41575383e+00 -1.17691056e+00  2.32786325e+00 -5.11050249e-01  4.60999697e-01  7.36012244e-01 -1.02636677e+00 -9.95354904e-01  1.19881597e+00 -7.32372662e-01 -4.37685648e-01 -1.04179621e+00 -8.61347686e-01  8.87828421e-02 -6.10382786e-01  3.35414982e-01  1.14892679e+00  8.99548124e-01  1.10451583e+00  1.33230363e+00 -9.11506291e-01 -8.92311795e-01  9.64641524e-01  9.05625210e-01 -5.07129938e-01  8.63696484e-01  3.56115819e-02  1.29949406e+00 -2.45406009e-01  6.54836989e-01  1.69568478e+00  4.70923050e-01  1.08256402e+00 -1.39884927e+00  1.47862740e+00  4.92136373e-01 -5.10603151e-01 -3.65404523e-01  1.18318965e-01  2.24867741e+00 -3.71107069e-01  1.34904702e+00  3.36416992e-02 -3.49262042e-01 -7.98738407e-01 -1.16375243e+00 -1.52676663e+00 -3.29110232e-01  3.36157374e-01  7.23704134e-02  3.35605631e-01 -6.51097580e-01 -6.09199311e-01  6.33053117e-01  8.80450243e-01  1.36613371e+00 -1.55159133e+00 -4.36599247e-01 -1.43070706e+00  1.03311660e+00  6.38766986e-01 -1.93859033e+00 -2.06307995e-01 -1.68479160e+00  3.77530131e-01 -1.81611988e-01  2.67891857e-01  1.29489625e+00 -2.16736694e-01 -1.07732497e-01  1.05190036e+00 -1.80788578e+00  3.80721775e-01  1.14105859e+00  1.68905614e+00 -1.59022271e-02 -5.33413658e-01  1.24989691e+00 -5.24879586e-01  1.08697047e+00 -1.10913733e+00 -7.08705958e-01 -8.51302798e-01  2.60096956e-02  1.05387026e+00 -8.82569631e-01  1.37442192e+00  1.36795316e+00 -6.02524975e-01 -1.53579196e+00 -1.07715948e+00  6.27126905e-01  1.32783375e+00  4.03299230e-01 -1.90829173e-01 -6.90021544e-01 -1.14841500e+00 -1.06470723e+00]";
        List<Double> l2 = LoadNetworkFromFileNumpy.parseDoublesFromArrayString(s2);
    }
}