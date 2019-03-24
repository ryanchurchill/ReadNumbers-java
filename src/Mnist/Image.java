package Mnist;

import java.util.*;

public class Image {
    public final static int PIXEL_LENGTH = 28;

    // X by Y (column by row)
    int[][] pixels = new int[PIXEL_LENGTH][PIXEL_LENGTH];

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
            pixels[xPos][yPos] = b & 0xff;

            xPos++;
            if (xPos == PIXEL_LENGTH) {
                xPos = 0;
                yPos++;
            }
        }
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
                int num = pixels[x][y];
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
                int num = pixels[x][y];
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
                ret.add((double)pixels[x][y]);
            }
        }
        return ret;
    }
}
