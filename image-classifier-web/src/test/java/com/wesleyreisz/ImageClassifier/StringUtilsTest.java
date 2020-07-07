package com.wesleyreisz.ImageClassifier;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
class StringUtilsTest {

    @Test
    void cropImageToRect() {
        BufferedImage bImage = loadImage();

        BufferedImage cropppedBImage = StringUtils.cropImageToRect(bImage);
        assertNotNull(cropppedBImage);
    }

    @Test
    void scaleImage() {
        BufferedImage bImage = loadImage();
        BufferedImage cropppedBImage = StringUtils.scaleImage(bImage,299, 299);
        assertNotNull(cropppedBImage);
        assertTrue((cropppedBImage.toString().contains("height = 299")));
    }


    @Test
    void makeImageTensor() {
        BufferedImage bImage = loadImage();
        BufferedImage croppedBImage = StringUtils.cropImageToRect(bImage);
        BufferedImage scaledImage = StringUtils.scaleImage(croppedBImage,299, 299);
        INDArray test = StringUtils.makeImageTensor(scaledImage,0, (float)299);
        assertNotNull(test);
    }

    @Test
    void loadMapFromJson(){
        Map<String, String> map = StringUtils.loadJsonFile("model/imagenet1000_clsidx_to_labels.txt");
        assertNotNull(map);
        assertEquals("French bulldog",map.get("245"));
    }

    private Path getFilePath(String file2load){
        ClassLoader classLoader = StringUtilsTest.class.getClassLoader();
        File file = new File(classLoader.getResource(file2load).getFile());
        return file.toPath();
    }

    private BufferedImage loadImage(){
        BufferedImage bImage = null;
        try {
            ClassLoader classLoader = getClass().getClassLoader();
            File file = new File(classLoader.getResource("lion.jpg").getFile());
            bImage = ImageIO.read(file);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bImage;
    }
}