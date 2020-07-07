package com.wesleyreisz.ImageClassifier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;

@RestController
public class ImageClassifierController {
    private static final Logger logger = LoggerFactory.getLogger(ImageClassifierController.class);

    @RequestMapping("/")
    public String index() {
        return "Image Classifier please post an image to /classify!";
    }

    @RequestMapping(value = "/predict", method = RequestMethod.POST)
    public String handleFormUpload(
            @RequestParam("image") MultipartFile file) throws IOException {
        String result = "Invalid Image";
        if (!file.isEmpty()) {
            BufferedImage img = ImageIO.read(new ByteArrayInputStream(file.getBytes()));
            result = ClassiferService.classifyImage(img);
        }
        logger.info("Image result: " + result);
        return result;
    }
}