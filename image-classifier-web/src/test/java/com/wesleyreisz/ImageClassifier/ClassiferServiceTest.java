package com.wesleyreisz.ImageClassifier;

import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class ClassiferServiceTest {
    @Test
    void testImageClassificationDog() {
        BufferedImage bImage = loadImage("dog.jpg");
        assertEquals("Rottweiler", ClassiferService.classifyImage((bImage)));
    }

    @Test
    void testImageClassificationLion() {
        BufferedImage bImage = loadImage("lion.jpg");
        assertEquals("lion, king of beasts, Panthera leo", ClassiferService.classifyImage((bImage)));
    }

    private BufferedImage loadImage(String imageMame){
        BufferedImage bImage = null;
        try {
            ClassLoader classLoader = getClass().getClassLoader();
            File file = new File(classLoader.getResource(imageMame).getFile());
            bImage = ImageIO.read(file);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bImage;
    }

}