package com.wesleyreisz.ImageClassifier;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class StringUtils {
    public static BufferedImage cropImageToRect(BufferedImage input) {
        int w = input.getWidth();
        int h = input.getHeight();
        int outSize = Math.min(w, h);
        int shiftX = w > outSize ? (w - outSize) / 2 : 0;
        int shiftY = h > outSize ? (h - outSize) / 2 : 0;
        return input.getSubimage(shiftX, shiftY, outSize, outSize);
    }
    public static BufferedImage scaleImage(BufferedImage input, int outWidth, int outHeight) {
        int w = input.getWidth();
        int h = input.getHeight();
        BufferedImage output = new BufferedImage(outWidth, outHeight, BufferedImage.TYPE_INT_ARGB);
        AffineTransform at = new AffineTransform();
        at.scale((float) outWidth / w, (float) outHeight / h);
        AffineTransformOp scaleOp = new AffineTransformOp(at, AffineTransformOp.TYPE_BICUBIC);
        output = scaleOp.filter(input, output);

        return output;
    }
    public static INDArray makeImageTensor(BufferedImage img, int imageMean, float imageStd) {
        // DirectColorModel: rmask=ff0000 gmask=ff00 bmask=ff amask=ff000000
        int[] data = ((DataBufferInt) img.getData().getDataBuffer()).getData();

        // Get data in UINT8
        float[] fdata = new float[data.length * 3];

        //Select the right UINT8 value, and normalise the value.
        for (int i = 0; i < data.length; i++) {
            int val = data[i];
            fdata[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
            fdata[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
            fdata[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
        }
        final long BATCH_SIZE = 1;
        final long CHANNELS = 3;

        long[] shape = new long[]{BATCH_SIZE, img.getHeight(), img.getWidth(), CHANNELS};
        //return Tensor.create(shape, FloatBuffer.wrap(fdata));
        INDArray toreturn= Nd4j.create(fdata, shape);

        return toreturn;
    }

    public static Map<String,String> loadJsonFile(String fileName){
        File file = new File(fileName);
        //String absolutePath2File = file.getAbsolutePath();

        ObjectMapper mapper = new ObjectMapper();
        mapper.getFactory().enable(JsonParser.Feature.ALLOW_SINGLE_QUOTES);
        Map<String, String> map = new HashMap();

        //read file
        String json = "";

        File f = null;
        try {
            //json = Files.readString(file.toPath());
            json = new String(Files.readAllBytes(file.toPath()));
        } catch (IOException e) {
            e.printStackTrace();
        }

        //load map
        try {
            map = mapper.readValue(json, Map.class);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return map;
    }
}
