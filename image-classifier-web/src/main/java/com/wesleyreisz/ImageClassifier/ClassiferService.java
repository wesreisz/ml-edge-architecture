package com.wesleyreisz.ImageClassifier;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelBuilder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Map;


public class ClassiferService {
    private static final String MODEL = "model/xception.h5";
    private static Map<String,String> IMAGENET = StringUtils.loadJsonFile("model/imagenet1000_clsidx_to_labels.txt");

    public static String classifyImage(BufferedImage input) {
        //load the model
        ComputationGraph model = importModel(MODEL);

        //crop, scale, and set RBG on image
        BufferedImage croppedImage = StringUtils.cropImageToRect(input);
        BufferedImage scaledImage = StringUtils.scaleImage(croppedImage, 299, 299);

        //this is the correct input shape to the model (I have to get the image into this shape)
        //INDArray dummyData = Nd4j.create(new int[] {1, 299, 299, 3});
        INDArray data = StringUtils.makeImageTensor(scaledImage, 128, 128);

        //Used this to check the answers in Python version Java
        //System.out.println("min:" + data.min());
        //System.out.println("max:" + data.max());

        INDArray result = model.output(data)[0]; // selects the only output tensor.
        //System.out.println("Size of output: " + result.length());
        //System.out.println("Class index:" + result.argMax(1)); //this is an answer
        //System.out.println("Max activation:" + result.max(1));
        String value2get = result.argMax(1).toStringFull().substring(1,result.argMax(1).toStringFull().length()-1);
        return IMAGENET.get(value2get);
    }
    private static ComputationGraph importModel(String modelName) {
        // load the model
        File file = new File(modelName);
        String simpleMlp = file.getAbsolutePath();

        ComputationGraph graph = null;
        try {
            try {
                int[] inputShape = new int[]{299, 299, 3};
                KerasModelBuilder builder = new KerasModel()
                        .modelBuilder()
                        .inputShape(inputShape)
                        .modelHdf5Filename(simpleMlp)
                        .enforceTrainingConfig(false);
                KerasModel model = builder.buildModel();
                graph = model.getComputationGraph();
            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (InvalidKerasConfigurationException e) {
            e.printStackTrace();
        } catch (UnsupportedKerasConfigurationException e) {
            e.printStackTrace();
        }
        return graph;
    }
}
