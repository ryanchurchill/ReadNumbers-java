import java.io.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ReadMnist {

    static final String TRAINING_LABEL_FILE = "E:\\dev\\ai-az\\ReadNumbers-Java\\data\\train-labels-idx1-ubyte";
    static final String TRAINING_IMAGE_FILE = "E:\\dev\\ai-az\\ReadNumbers-Java\\data\\train-images-idx3-ubyte";
    static final int EXPECTED_TRAINING_COUNT = 60000;

    public List<Image> getImages() throws Exception
    {
        List<Image> images = new ArrayList<>();

        FileInputStream imageStream = new FileInputStream(TRAINING_IMAGE_FILE);
        // 1. ignore first 4 bytes
        byte[] imageStreamBuffer = new byte[4];
        imageStream.read(imageStreamBuffer);

        // 2. next 4 bytes are the number of images
        imageStream.read(imageStreamBuffer);
        ByteBuffer wrapped = ByteBuffer.wrap(imageStreamBuffer);
        int actualSize = wrapped.getInt();
        if (actualSize != EXPECTED_TRAINING_COUNT) {
            throw new Exception("actual Size incorrect: " + Integer.toString(actualSize));
        }

        // now we parse through the rest. Each image is 792 bytes.
        imageStreamBuffer = new byte[792];

        while((imageStream.read(imageStreamBuffer)) != -1) {
            images.add(getImage(imageStreamBuffer));
        }

        return images;
    }

    /**
     * Expect 792 bytes total
     * first four bytes: number of rows (always 28)
     * second four bytes: number of columns (always 28)
     * remaining 784 bytes are the pixels
     * TODO: this is not memory efficient!
     * @param bytes
     */
    private Image getImage(byte[] bytes) throws Exception
    {
        // validation
        if (bytes.length != 792) {
            throw new Exception("bytes did not have expected length: " + Integer.toString(bytes.length));
        }

        int byteCursor = 0;
        byte[] rowsAndColumns = Arrays.copyOfRange(bytes, 0, 7);
        // TODO: validate rowsAndColumns
        byte[] pixels = Arrays.copyOfRange(bytes,8, 791);
        return new Image(pixels);
    }

    // i think this makes things more confusing because we don't know the bit-size of each "decimal", so let's stick
    // with hex
    public void printLabelsAsDecimal()
    {
        String fileName = TRAINING_LABEL_FILE;

        try {
            // Use this for reading the data.
            byte[] currentByte = new byte[1];

            FileInputStream inputStream = new FileInputStream(fileName);

            while((inputStream.read(currentByte)) != -1) {
                byte b = currentByte[0];
                int i = b & 0xff;
                System.out.println(i);
            }

            inputStream.close();
        }
        catch(FileNotFoundException ex) {
            System.out.println(
                    "Unable to open file '" +
                            fileName + "'");
        }
        catch(IOException ex) {
            System.out.println(
                    "Error reading file '"
                            + fileName + "'");
        }
    }

    public void printLabelsAsHex()
    {
        String fileName = TRAINING_LABEL_FILE;

        try {
            // Use this for reading the data.
            byte[] currentByte = new byte[1];

            FileInputStream inputStream = new FileInputStream(fileName);

            // read fills buffer with data and returns
            // the number of bytes read (which of course
            // may be less than the buffer size, but
            // it will never be more).
            int total = 0;
            while((inputStream.read(currentByte)) != -1) {
                // Convert to String so we can display it.
                // Of course you wouldn't want to do this with
                // a 'real' binary file.
//                System.out.println(new String(buffer));
//                total += nRead;

                byte b = currentByte[0];
                String str  = Integer.toHexString((b & 0xff)+256).substring(1);
                System.out.println(str);
            }

            // Always close files.
            inputStream.close();

            System.out.println("Read " + total + " bytes");
        }
        catch(FileNotFoundException ex) {
            System.out.println(
                    "Unable to open file '" +
                            fileName + "'");
        }
        catch(IOException ex) {
            System.out.println(
                    "Error reading file '"
                            + fileName + "'");
            // Or we could just do this:
            // ex.printStackTrace();
        }
    }

    public static void main(String[] args) throws Exception {
        ReadMnist readMnist = new ReadMnist();
        readMnist.getImages();
    }
}
