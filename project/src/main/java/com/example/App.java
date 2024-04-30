package com.example;

import java.io.*;
import org.apache.spark.sql.*;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;

public class App {

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
            .appName("Linear Regression Example")
            .config("spark.master", "local")
            .getOrCreate();

        Dataset<Row> trainingData = spark.read()
            .format("csv")
            .option("inferSchema", "true")
            .option("header", "true")
            .load("TrainingDataset.csv");

        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(new String[]{"fixed acidity", "volatile acidity", "citric acid", "residual sugar", 
                "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"})
            .setOutputCol("features");

        Dataset<Row> output = assembler.transform(trainingData);

        LinearRegression lr = new LinearRegression()
            .setFeaturesCol("features")
            .setLabelCol("quality");
        
        LinearRegressionModel model = lr.fit(output);

        model.write().overwrite().save("project/model");

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{assembler, lr});
        PipelineModel pipelineModel = pipeline.fit(output);
        pipelineModel.write().overwrite().save("project/pipeline");

        spark.stop();
    }
}
