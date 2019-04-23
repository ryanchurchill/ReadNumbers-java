package Network.Learning;

import Util.MyMathUtils;
import org.apache.commons.math3.linear.RealVector;

import java.util.*;

public class TrainingExample {
    public RealVector input;
    public RealVector desiredOutput;

    public TrainingExample(RealVector _input, RealVector _desiredOutput)
    {
        input = _input;
        desiredOutput = _desiredOutput;
    }

    public List<Double> getInputAsList()
    {
        return MyMathUtils.rvToList(input);
    }

    public List<Double> getDisiredOutputAsList()
    {
        return MyMathUtils.rvToList(desiredOutput);
    }
}
