package com.company;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.File;
import java.io.IOException;

/**
 * Created by luthfi on 28/09/15.
 */
public class WekaJava {
    private Instances data;
    private Classifier model;

    public static final int NAIVE_BAYES = 0;
    public static final int ID3 = 1;

    public WekaJava() {
        data = null;
        model = null;
    }

    public void loadData(String ArffFile) throws Exception {
        DataSource source = new DataSource(ArffFile);
        data = source.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
    }

    public void loadDataFromCSV(String CsvFile) throws IOException {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(CsvFile));
        data = loader.getDataSet();
    }

    public void buildClassifier(int type) throws Exception {
        switch (type) {
            case 0 :
                model = new NaiveBayes();
                break;
            case 1 :
                model = new Id3();
                break;
            case 2 :
                model = new CustomID3();
            default:
                break;
        }
        model.buildClassifier(data);
    }

    public void saveModel() throws Exception {
        if (model != null) {
            weka.core.SerializationHelper.write("weka.model", model);
        }
        else {
            System.out.println("Model is null");
        }
    }

    public void loadModel(String file) throws Exception {
        model = (Classifier) weka.core.SerializationHelper.read(file);
    }
}
