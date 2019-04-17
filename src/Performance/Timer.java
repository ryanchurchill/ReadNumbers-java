package Performance;

public class Timer {
    int timeInMs = 0;
    long start;

    public int getTimeInMs()
    {
        return timeInMs;
    }

    public void start()
    {
        start = System.currentTimeMillis();
    }

    public void stop()
    {
        timeInMs += System.currentTimeMillis() - start;
    }
}
