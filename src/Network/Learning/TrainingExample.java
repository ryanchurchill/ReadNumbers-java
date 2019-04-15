package Network.Learning;

import java.util.List;

public class TrainingExample {
    public List<Double> input;
    public List<Double> desiredOutput;

    public TrainingExample(List<Double> _input, List<Double> _desiredOutput)
    {
        input = _input;
        desiredOutput = _desiredOutput;
    }
}
