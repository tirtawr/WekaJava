package com.company;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Created by luthfi on 28/09/15.
 */
public class WekaJava {
    private Instances data;
    private Classifier model;

    public static final int NAIVE_BAYES = 0;
    public static final int ID3 = 1;
    public static final int CUSTOM_ID3 = 2;
    public static final int CUSTOM_C45 = 3;

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

    public void removeAttribute(String attribute) throws Exception {
        Remove remove = new Remove();
        remove.setAttributeIndices(attribute);
        remove.setInputFormat(data);
        data = Filter.useFilter(data, remove);
    }

    public void resample() throws Exception {
        Resample resample = new Resample();
        resample.setInputFormat(data);
        data = Filter.useFilter(data, resample);
    }

    public void testModel(Instances datatest) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(model, datatest);

        System.out
                .println(eval.toSummaryString("=== Summary ===\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }

    public void tenFoldCrossValidation() throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data,
                10, new Random(1));
        System.out
                .println(eval.toSummaryString("=== Summary ===\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }

    public void percentageSplit(Double percentage) {
        Instances dataSet = new Instances(data);
        dataSet.randomize(new Random(1));

        int trainSize = (int) Math.round(dataSet.numInstances() * percentage / 100);
        int testSize = dataSet.numInstances() - trainSize;
        Instances trainSet = new Instances(dataSet, 0, trainSize);
        Instances testSet = new Instances(dataSet, trainSize, testSize);

        try {
            model.buildClassifier(trainSet);
            Evaluation eval = new Evaluation(trainSet);
            eval.evaluateModel(model, testSet);

            System.out
                    .println(eval.toSummaryString("=== Summary ===\n", false));
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
        } catch (Exception e) {
            e.printStackTrace();
        }
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
                break;
            case 3 :
                model = new CustomC45();
                break;    
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

    public void classify(String file) throws Exception {
        Instances unlabeled = DataSource.read(file);
        unlabeled.setClassIndex(unlabeled.numAttributes() - 1);

        Instances labeled = new Instances(unlabeled);

        // label instances
        for (int i = 0; i < unlabeled.numInstances(); i++) {
            double clsLabel = model.classifyInstance(unlabeled.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
            System.out.println(labeled.instance(i));
        }
    }
}
