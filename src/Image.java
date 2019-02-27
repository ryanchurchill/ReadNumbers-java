public class Image {
    final static int PIXEL_LENGTH = 28;

    // X by Y (column by row)
    int[][] pixels = new int[PIXEL_LENGTH][PIXEL_LENGTH];
    int actualDigit; // 0-9 AKA label

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
}
