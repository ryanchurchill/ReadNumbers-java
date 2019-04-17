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
        if (Globals.timing) {
            start = System.currentTimeMillis();
        }
    }

    public void stop()
    {
        if (Globals.timing) {
            timeInMs += System.currentTimeMillis() - start;
        }
    }
}
