#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum
from pyspark.mllib.util import MLUtils
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
from pyspark.ml.linalg import DenseVector
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum
from pyspark.mllib.util import MLUtils
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.regression import LabeledPoint


from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
from pyspark.ml.linalg import DenseVector
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder


def create_spark_session():
    spark = SparkSession.builder.getOrCreate()
    return spark

def load_data(spark, file_path):
    data = spark.read.csv(file_path, header=True, inferSchema=True)
    return data

def preprocess_data(data):
    # Définir les colonnes numériques et catégorielles
    colonnes_numeriques = ["loan", "mortdue", "value", "yoj", "derog", "delinq", "clage", "ninq", "clno", "debtinc"]
    colonnes_categorielles = ["reason", "job"]

    # Remplacement des valeurs manquantes pour les variables numériques
    imputer_numerique = Imputer(strategy="mean", inputCols=colonnes_numeriques, outputCols=colonnes_numeriques)
    data = imputer_numerique.fit(data).transform(data)

    # Remplacement des valeurs manquantes pour les variables catégorielles
    for colonne in colonnes_categorielles:
        mode = data.groupBy(colonne).count().orderBy("count", ascending=False).first()[colonne]
        data = data.na.fill({colonne: mode})
    
    return data

def encode_features(data):
    indexer_reason = StringIndexer(inputCol="reason", outputCol="reason_index")
    indexer_job = StringIndexer(inputCol="job", outputCol="job_index")

    # Encodage one-hot des variables catégorielles indexées
    encoder_reason = OneHotEncoder(inputCols=["reason_index"], outputCols=["reason_encoded"])
    encoder_job = OneHotEncoder(inputCols=["job_index"], outputCols=["job_encoded"])

    # Assemblage des colonnes en un vecteur de features
    assembler = VectorAssembler(inputCols=["loan", "mortdue", "value", "yoj", "derog", "delinq", "clage", "ninq", "clno", "debtinc", "reason_encoded", "job_encoded"], outputCol="features")

    # Création de la pipeline pour exécuter les transformations en séquence
    pipeline = Pipeline(stages=[indexer_reason, indexer_job, encoder_reason, encoder_job, assembler])

    # Application de la pipeline sur le DataFrame data
    data_final = pipeline.fit(data).transform(data)

    # Sélection des colonnes label et features
    df = data_final.withColumn("label", data["bad"].cast("double")).select("label", "features")

    return df

def split_data(df, ratio):
    training_df, test_df = df.randomSplit([ratio, 1-ratio])
    return training_df, test_df

def train_model(training_df):
    rf_classifier = RandomForestClassifier(labelCol="label", numTrees=100).fit(training_df)
    return rf_classifier

def evaluate_model(classifier, test_df):
    rf_prediction = classifier.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(rf_prediction)
    return accuracy




