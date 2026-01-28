from pyspark import SparkContext, SparkConf
from pyspark.shell import spark
from pyspark.sql import SQLContext, SparkSession
import numpy as np
from pyspark.sql import DataFrame
# from pyspark.sql import SQLContext
from pyspark.sql.functions import when
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
# from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GeneralizedLinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.types import StructType, StructField, NumericType
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
import time


def isSick(x):
    if x in (3, 7):
        return 0
    else:
        return 1


def classify():
    cols = ['age',
            'sex',
            'chest pain',
            'resting blood pressure',
            'serum cholesterol',
            'fasting blood sugar',
            'resting electrocardiographic results',
            'maximum heart rate achieved',
            'exercise induced angina',
            'ST depression induced by exercise relative to rest',
            'the slope of the peak exercise ST segment',
            'number of major vessels ',
            'thal',
            'last']

    data = pd.read_csv('heart.csv', delimiter=' ', names=cols)
    data = data.iloc[:, 0:13]
    data['label'] = data['thal'].apply(isSick)
    df = spark.createDataFrame(data)

    features = ['age',
                'sex',
                'chest pain',
                'resting blood pressure',
                'serum cholesterol',
                'fasting blood sugar',
                'resting electrocardiographic results',
                'maximum heart rate achieved',
                'exercise induced angina',
                'ST depression induced by exercise relative to rest',
                'the slope of the peak exercise ST segment',
                'number of major vessels ']

    assembler = VectorAssembler(inputCols=features, outputCol="features")
    raw_data = assembler.transform(df)
    raw_data.select("features").show(truncate=False)

    standardscaler = StandardScaler().setInputCol("features").setOutputCol("Scaled_features")
    raw_data = standardscaler.fit(raw_data).transform(raw_data)
    raw_data.select("features", "Scaled_features").show(5)

    from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

    training, test = raw_data.randomSplit([0.5, 0.5], seed=12345)
    from pyspark.ml.classification import LogisticRegression

    # ----------------------------- LOGISTIC REGRESSION -----------------------------
    # lr = LogisticRegression(labelCol="label", featuresCol="Scaled_features", maxIter=100)
    # model = lr.fit(training)
    # plt.figure(figsize=(5, 5))
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.plot(model.summary.roc.select('FPR').collect(),
    #          model.summary.roc.select('TPR').collect())
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.show()
    # predict_train = model.transform(training)
    # predict_test = model.transform(test)
    # predict_test.select("label", "prediction").show(10)
    # print("Multinomial coefficients: " + str(model.coefficientMatrix))
    # print("Multinomial intercepts: " + str(model.interceptVector))
    # # import pyspark.sql.functions as F
    # # check = predict_test.withColumn('correct', F.when(F.col('isSick') == F.col('prediction'), 1).otherwise(0))
    # # check.groupby("correct").count().show()
    # evaluator = BinaryClassificationEvaluator()
    # print("Test Area Under ROC: " + str(evaluator.evaluate(predict_test, {evaluator.metricName: "areaUnderROC"})))

    # ----------------------------- RANDOM FOREST -----------------------------
    rf = RandomForestClassifier(labelCol="label", featuresCol="Scaled_features", numTrees=200)
    model = rf.fit(training)
    predict_train = model.transform(training)
    predict_test = model.transform(test)
    predict_test.select("label", "prediction").show(10)

    # print("Multinomial coefficients: " + str(model.coefficientMatrix))
    # print("Multinomial intercepts: " + str(model.interceptVector))
    evaluator = BinaryClassificationEvaluator()
    print("Test Area Under ROC: " + str(evaluator.evaluate(predict_test, {evaluator.metricName: "areaUnderROC"})))

    return 0


def main():
    classify()


if __name__ == "__main__":
    main()