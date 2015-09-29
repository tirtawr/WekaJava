package com.company;

import java.util.Enumeration;
import java.util.HashSet;
import java.util.List;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Tirta Wening Rachman
 */

public class CustomC45 extends Classifier {

    private final double MISSING_VALUE = Double.NaN;
    private final double DOUBLE_COMPARE_VALUE = 1e-6;
    private List<Rule> ruleList;

    private CustomC45[] m_Children;
    private Attribute m_Attribute;
    private double m_Label;
    private double[] m_ClassDistribution;
    private Attribute m_ClassAttribute;

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {

        // Periksa data
        getCapabilities().testWithFail(data);

        // Hapus instances dengan missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        makePrunedTree(data);
    }

    private void makePrunedTree(Instances data) throws Exception {
        makeTree(data);
        pruneTree(data);
    }
    
    private void makeTree(Instances data) throws Exception {

        // Periksa instans dalam node
        if (data.numInstances() == 0) {
            m_Attribute = null;
            m_Label = MISSING_VALUE;
            m_ClassDistribution = new double[data.numClasses()];
        } else {
            // Cari IG maksimum
            double[] gainRatios = new double[data.numAttributes()];

            data = toNominalInstances(data);

            Enumeration attEnum = data.enumerateAttributes();
            while (attEnum.hasMoreElements()) {
                Attribute att = (Attribute) attEnum.nextElement();
                gainRatios[att.index()] = computeGainRatio(data, att);
            }

            m_Attribute = data.attribute(maxIndex(gainRatios));

            // Membuat daun jika IG = 0
            if (doubleEqual(gainRatios[m_Attribute.index()], 0)) {
                m_Attribute = null;

                m_ClassDistribution = new double[data.numClasses()];
                for (int i = 0; i < data.numInstances(); i++) {
                    Instance inst = (Instance) data.instance(i);
                    m_ClassDistribution[(int) inst.classValue()]++;
                }

                normalizeDouble(m_ClassDistribution);
                m_Label = maxIndex(m_ClassDistribution);
                m_ClassAttribute = data.classAttribute();
            } else {
                // Membuat pohon baru di bawah node
                Instances[] splitData = splitData(data, m_Attribute);
                m_Children = new CustomC45[m_Attribute.numValues()];
                for (int j = 0; j < m_Attribute.numValues(); j++) {
                    m_Children[j] = new CustomC45();
                    m_Children[j].makeTree(splitData[j]);
                }
            }
        }
    }
    
    private void pruneTree(Instances data) {
        //belum
    }
    
    private Instances toNominalInstances(Instances data) throws Exception {
        for (int ix = 0; ix < data.numAttributes(); ++ix) {
            Attribute att = data.attribute(ix);
            if (data.attribute(ix).isNumeric()) {

                // Get an array of integer that consists of distinct values of the attribute
                HashSet<Double> numericSet = new HashSet<Double>();
                for (int i = 0; i < data.numInstances(); ++i) {
                    numericSet.add(data.instance(i).value(att));
                }

                Double[] numericValues = new Double[numericSet.size()];
                int iterator = 0;
                for (Double i : numericSet) {
                    numericValues[iterator] = i;
                    iterator++;
                }

                // Sort the array
                sortArray(numericValues);

                // Search for threshold and get new Instances
                double[] infoGains = new double[numericValues.length - 1];
                Instances[] tempInstances = new Instances[numericValues.length - 1];
                for (int i = 0; i < numericValues.length - 1; ++i) {
                    tempInstances[i] = convertInstances(data, att, numericValues[i]);
                    try {
                        infoGains[i] = computeGainRatio(tempInstances[i], tempInstances[i].attribute(att.name()));
                    } catch (Exception e) {
                    }
                }

                data = new Instances(tempInstances[maxIndex(infoGains)]);
            }
        }
        return data;
    }

    private static Instances convertInstances(Instances data, Attribute att, double threshold) throws Exception {
        Instances newData = new Instances(data);

        // Add attribute        
        try {
            Add filter = new Add();
            filter.setAttributeIndex((att.index() + 2) + "");
            filter.setNominalLabels("<=" + threshold + ",>" + threshold);
            filter.setAttributeName(att.name() + "temp");
            filter.setInputFormat(newData);
            newData = Filter.useFilter(newData, filter);
        } catch (Exception e) {
            e.printStackTrace();
        }

        for (int i = 0; i < newData.numInstances(); ++i) {
            if ((double) newData.instance(i).value(newData.attribute(att.name())) <= threshold) {
                newData.instance(i).setValue(newData.attribute(att.name() + "temp"), "<=" + threshold);
            } else {
                newData.instance(i).setValue(newData.attribute(att.name() + "temp"), ">" + threshold);
            }
        }

        Remove remove = new Remove();
        remove.setAttributeIndices((att.index() + 1) + "");
        remove.setInputFormat(newData);
        Instances finalData = Filter.useFilter(newData, remove);
        finalData.renameAttribute(finalData.attribute(att.name() + "temp"), att.name());

        return finalData;
    }

    private static void sortArray(Double[] arr) {
        double temp;
        for (int i = 0; i < arr.length - 1; i++) {
            for (int j = 1; j < arr.length - i; j++) {
                if (arr[j - 1] > arr[j]) {
                    temp = arr[j - 1];
                    arr[j - 1] = arr[j];
                    arr[j] = temp;
                }
            }
        }
    }

    private void normalizeDouble(double[] array) {
        double sum = 0;
        for (double d : array) {
            sum += d;
        }

        if (!Double.isNaN(sum) && sum != 0) {
            for (int i = 0; i < array.length; ++i) {
                array[i] /= sum;
            }
        } else {
            // Do nothing       
        }
    }


    private boolean doubleEqual(double d1, double d2) {
        return (d1 == d2) || Math.abs(d1 - d2) < DOUBLE_COMPARE_VALUE;
    }

    private static int maxIndex(double[] array) {
        double max = 0;
        int index = 0;

        if (array.length > 0) {
            for (int i = 0; i < array.length; ++i) {
                if (array[i] > max) {
                    max = array[i];
                    index = i;
                }
            }
            return index;
        } else {
            return -1;
        }
    }

    @Override
    public double classifyInstance(Instance instance)
        throws NoSupportForMissingValuesException {

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("CustomC45: Cannot handle missing values");
        }
        if (m_Attribute == null) {
            return m_Label;
        } else {
            boolean isComparison = false;
            Enumeration enumeration = m_Attribute.enumerateValues();
            String val = null;
            while (enumeration.hasMoreElements()) {
                val = (String) enumeration.nextElement();
                if (val.contains("<")) {
                    isComparison = true;
                    break;
                }
            }

            if (isComparison) {
                double threshold = getThreshold(val);
                double instanceValue = (double) instance.value(m_Attribute);

                if (instanceValue <= threshold) {
                    instance.setValue(m_Attribute, "<=" + String.valueOf(threshold));
                } else {
                    instance.setValue(m_Attribute, ">" + String.valueOf(threshold));
                }
            }
            return m_Children[(int) instance.value(m_Attribute)].
                classifyInstance(instance);
        }
    }

    private double getThreshold(String val) {
        return Double.parseDouble(val.replace("<=", ""));
    }

    @Override
    public double[] distributionForInstance(Instance instance)
        throws NoSupportForMissingValuesException {

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("CustomC45: Cannot handle missing values");
        }
        if (m_Attribute == null) {
            return m_ClassDistribution;
        } else {
            return m_Children[(int) instance.value(m_Attribute)].
                distributionForInstance(instance);
        }
    }

    @Override
    public String toString() {

        if ((m_ClassDistribution == null) && (m_Children == null)) {
            return "CustomC45: No model built yet.";
        }
        return "CustomC45\n\n" + toString(0);
    }

    private static double computeGainRatio(Instances data, Attribute att)
        throws Exception {

        double infoGain = computeInfoGain(data, att);
        double splitInfo = computeSplitInformation(data, att);
        return infoGain > 0 ? infoGain / splitInfo : infoGain;
    }

    private static double computeInfoGain(Instances data, Attribute att)
        throws Exception {

        double infoGain = computeEntropy(data);
        Instances[] splitData = splitData(data, att);
        for (Instances splitdata : splitData) {
            if (splitdata.numInstances() > 0) {
                double splitNumInstances = splitdata.numInstances();
                double dataNumInstances = data.numInstances();
                double proportion = splitNumInstances / dataNumInstances;
                infoGain -= proportion * computeEntropy(splitdata);
            }
        }
        return infoGain;
    }

    private static double computeEntropy(Instances data) throws Exception {

        double[] labelCounts = new double[data.numClasses()];
        for (int i = 0; i < data.numInstances(); ++i) {
            labelCounts[(int) data.instance(i).classValue()]++;
        }

        double entropy = 0;
        for (int i = 0; i < labelCounts.length; i++) {
            if (labelCounts[i] > 0) {
                double proportion = labelCounts[i] / data.numInstances();
                entropy -= (proportion) * log2(proportion);
            }
        }
        return entropy;
    }

    private static double computeSplitInformation(Instances data, Attribute att) throws Exception {

        double splitInfo = 0;
        Instances[] splitData = splitData(data, att);
        double dataNumInstances = data.numInstances();

        for (Instances splitdata : splitData) {
            if (splitdata.numInstances() > 0) {
                double splitNumInstances = splitdata.numInstances();
                double proportion = splitNumInstances / dataNumInstances;
                splitInfo -= proportion * log2(proportion);
            }
        }
        return splitInfo;
    }

    private static double log2(double num) {
        return (num == 0) ? 0 : Math.log(num) / Math.log(2);
    }

    private static Instances[] splitData(Instances data, Attribute att) {

        Instances[] splitData = new Instances[att.numValues()];
        for (int j = 0; j < att.numValues(); j++) {
            splitData[j] = new Instances(data, data.numInstances());
        }

        for (int i = 0; i < data.numInstances(); i++) {
            splitData[(int) data.instance(i).value(att)].add(data.instance(i));
        }

        for (Instances splitData1 : splitData) {
            splitData1.compactify();
        }
        return splitData;
    }

    private String toString(int level) {

        StringBuilder text = new StringBuilder();

        if (m_Attribute == null) {
            if (Instance.isMissingValue(m_Label)) {
                text.append(": null");
            } else {
                text.append(": ").append(m_ClassAttribute.value((int) m_Label));
            }
        } else {
            for (int j = 0; j < m_Attribute.numValues(); j++) {
                text.append("\n");
                for (int i = 0; i < level; i++) {
                    text.append("|  ");
                }
                text.append(m_Attribute.name()).append(" = ").append(m_Attribute.value(j));
                text.append(m_Children[j].toString(level + 1));
            }
        }
        return text.toString();
    }

}
