package com.example;

import java.io.*;
import org.apache.spark.sql.*;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;

public class App {

    public static void main(String[] args) {
        // Create a Spark session
        SparkSession spark = SparkSession.builder()
            .appName("Linear Regression Example")
            .config("spark.master", "local")
            .getOrCreate();

        // Load the training data
        Dataset<Row> trainingData = spark.read()
            .format("csv")
            .option("inferSchema", "true")
            .option("header", "true")
            .load("TrainingDataset.csv");

        // Specify the features to use for the model
        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(new String[]{"fixed acidity";"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";})
            .setOutputCol("quality");

        // Transform the dataset
        Dataset<Row> output = assembler.transform(trainingData);

        // Define the Linear Regression model
        LinearRegression lr = new LinearRegression()
            .setFeaturesCol("features")
            .setLabelCol("label"); // replace "label" with the name of your label column

        // Fit the model
        LinearRegressionModel model = lr.fit(output);

        // Print the coefficients and intercept for linear regression
    }
}