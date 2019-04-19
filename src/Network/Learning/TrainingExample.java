package Network.Learning;

import org.apache.commons.math3.linear.RealVector;

import java.util.List;

public class TrainingExample {
    public RealVector input;
    public RealVector desiredOutput;

    public TrainingExample(RealVector _input, RealVector _desiredOutput)
    {
        input = _input;
        desiredOutput = _desiredOutput;
    }
}
