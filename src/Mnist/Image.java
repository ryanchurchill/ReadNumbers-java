package Mnist;

import java.util.*;

public class Image {
    public final static int PIXEL_LENGTH = 28;
    final static int ORIGINAL_MIN = 0;
    final static int ORIGINAL_MAX = 255;

    // X by Y (column by row)
    // from the file, we get a range of 0-255. We normalize into the 0-1 range.
    double[][] pixels = new double[PIXEL_LENGTH][PIXEL_LENGTH];


    public int getActualDigit() {
        return actualDigit;
    }

    int actualDigit = -1; // 0-9 AKA label

    // Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
    public Image(byte[] bytes) throws Exception
    {
        if (bytes.length != PIXEL_LENGTH * PIXEL_LENGTH) {
            throw new Exception("unexpected bytes.length: " + Integer.toString(bytes.length));
        }

        int xPos = 0;
        int yPos = 0;

        for (byte b : bytes)
        {
            int value = b & 0xff;
            pixels[xPos][yPos] = normalize(value);

            xPos++;
            if (xPos == PIXEL_LENGTH) {
                xPos = 0;
                yPos++;
            }
        }
    }

    private double normalize(int originalValue)
    {
        return ((double)(originalValue-ORIGINAL_MIN)/(double)(ORIGINAL_MAX-ORIGINAL_MIN));
    }

    public void setActualDigit(int digit) throws Exception
    {
        if (digit < 0 || digit > 9) {
            throw new Exception("Invalid digit: " + Integer.toString(digit));
        }

        actualDigit = digit;
    }

    public String toString()
    {
        StringBuffer sb = new StringBuffer();
        sb.append("Actual: " + Integer.toString(actualDigit));
        sb.append(getDrawing());
        for (int y = 0; y < PIXEL_LENGTH; y++) {
            sb.append("\n");
            for (int x=0; x < PIXEL_LENGTH; x++) {
                double num = pixels[x][y];
                sb.append(String.format("%03d", num));
                sb.append(' ');
            }
        }
        return sb.toString();
    }

    public String getDrawing()
    {
        StringBuffer sb = new StringBuffer();

        for (int y = 0; y < PIXEL_LENGTH; y++) {
            sb.append("\n");
            for (int x=0; x < PIXEL_LENGTH; x++) {
                double num = pixels[x][y];
                if (num > 100) {
                    sb.append('.');
                } else {
                    sb.append(' ');
                }
            }
        }
        return sb.toString();
    }

    public List<Double> getPixelsForNetwork()
    {
        List<Double> ret = new ArrayList<>();
        for (int y = 0; y < PIXEL_LENGTH; y++) {
            for (int x=0; x < PIXEL_LENGTH; x++) {
                ret.add(pixels[x][y]);
            }
        }
        return ret;
    }
}
