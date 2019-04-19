package Mnist;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class ReadMnist {

    static final String TRAINING_LABEL_FILE = "E:\\dev\\ai-az\\ReadNumbers-Java\\data\\train-labels-idx1-ubyte";
    static final String TRAINING_IMAGE_FILE = "E:\\dev\\ai-az\\ReadNumbers-Java\\data\\train-images-idx3-ubyte";

//    static final String TRAINING_LABEL_FILE = "/Users/rchurchill/udemy/NN-Book/ReadNumbers-java/data/trainWithObjects-labels-idx1-ubyte";
//    static final String TRAINING_IMAGE_FILE = "/Users/rchurchill/udemy/NN-Book/ReadNumbers-java/data/trainWithObjects-images-idx3-ubyte";

    static final int EXPECTED_TRAINING_COUNT = 60000;

    public static List<Image> getTrainingImages() throws Exception
    {
        return getImages(TRAINING_IMAGE_FILE, TRAINING_LABEL_FILE, EXPECTED_TRAINING_COUNT);
    }

//    public List<Image> getTestImages() throws Exception
//    {
//        return getImages(...);
//    }

    private static List<Image> getImages(String imageFilePath, String labelFilePath, int expectedCount) throws Exception
    {
        // STEP 1: build images from TRAINING_IMAGE_FILE
        List<Image> images = getImagesFromImageFile(imageFilePath, expectedCount);

        // STEP 2: iterate through TRAINING_LABEL_FILE and add the labels to images
        setActualDigitsInImagesFromLabelFile(images, labelFilePath, expectedCount);

        return images;
    }

    private static List<Image> getImagesFromImageFile(String imageFilePath, int expectedCount) throws Exception
    {
        List<Image> images = new ArrayList<>();

        FileInputStream imageStream = new FileInputStream(imageFilePath);
        // ignore first 4 bytes (magic number)
        byte[] imageStreamBuffer = new byte[4];
        imageStream.read(imageStreamBuffer);

        // next 4 bytes are the number of images
        imageStream.read(imageStreamBuffer);
        ByteBuffer wrapped = ByteBuffer.wrap(imageStreamBuffer);
        int actualSize = wrapped.getInt();
        if (actualSize != expectedCount) {
            throw new Exception("actual Size incorrect: " + Integer.toString(actualSize));
        }

        // next 4 bytes are rows
        imageStream.read(imageStreamBuffer);
        wrapped = ByteBuffer.wrap(imageStreamBuffer);
        int numberOfRows = wrapped.getInt();
        if (numberOfRows != Image.PIXEL_LENGTH) {
            throw new Exception("numberOfRows incorrect: " + Integer.toString(numberOfRows));
        }

        // next 4 bytes are columns
        imageStream.read(imageStreamBuffer);
        wrapped = ByteBuffer.wrap(imageStreamBuffer);
        int numberOfColumns = wrapped.getInt();
        if (numberOfColumns != Image.PIXEL_LENGTH) {
            throw new Exception("numberOfColumns incorrect: " + Integer.toString(numberOfColumns));
        }

        // now we parse through the rest. Each image is 784 bytes.
        imageStreamBuffer = new byte[784];

        while((imageStream.read(imageStreamBuffer)) != -1) {
            images.add(getImage(imageStreamBuffer));
        }

        imageStream.close();

        if (images.size() != expectedCount) {
            throw new Exception("images.size() = " + Integer.toString(images.size()));
        }

        return images;
    }

    /**
     * This assumes that order of images matches the order of the labels in the label file
     * @param images
     * @param labelFilePath
     * @param expectedCount
     * @throws Exception
     */
    private static void setActualDigitsInImagesFromLabelFile(List<Image> images, String labelFilePath, int expectedCount) throws Exception
    {
        FileInputStream labelStream = new FileInputStream(labelFilePath);

        // 1. ignore first 4 bytes
        byte[] labelStreamBuffer = new byte[4];
        labelStream.read(labelStreamBuffer);

        // 2. next 4 bytes are the number of images
        labelStream.read(labelStreamBuffer);
        ByteBuffer wrapped = ByteBuffer.wrap(labelStreamBuffer);
        int actualSize = wrapped.getInt();
        if (actualSize != expectedCount) {
            throw new Exception("actual Size incorrect: " + Integer.toString(actualSize));
        }

        // now we parse through the rest. Each label is a single byte
        labelStreamBuffer = new byte[expectedCount];
        labelStream.read(labelStreamBuffer);
        int counter = 0;
        for (byte b : labelStreamBuffer) {
            images.get(counter).setActualDigit(b & 0xff);
            counter++;
        }

        // sanity check that we've reached the end of the buffer
        if (labelStream.read() != -1) {
            throw new Exception("labelStream longer than expected");
        }
        labelStream.close();
    }

    /**
     * Expect 784 bytes total: the pixels
     * TODO: this is not memory efficient!
     * @param bytes
     */
    private static Image getImage(byte[] bytes) throws Exception
    {
        // validation
        if (bytes.length != 784) {
            throw new Exception("bytes did not have expected length: " + Integer.toString(bytes.length));
        }
        return new Image(bytes);
    }

    // i think this makes things more confusing because we don't know the bit-size of each "decimal", so let's stick
    // with hex
    public static void printLabelsAsDecimal()
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

    public static void printLabelsAsHex()
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
//        ReadMnist readMnist = new ReadMnist();
        List<Image> images = ReadMnist.getTrainingImages();

        for (int i = 10; i < 20; i++) {
            System.out.println(images.get(i));
        }
    }
}
