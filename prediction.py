#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from process import *

def main():
    spark = create_spark_session()
    file_path = "/Users/oceanetaleb/Desktop/projet_big_data/tab4.csv"
    data = load_data(spark, file_path)
    data = preprocess_data(data)
    df = encode_features(data)
    training_df, test_df = split_data(df, 0.7)
    classifier = train_model(training_df)
    accuracy = evaluate_model(classifier, test_df)
    print("Accuracy: {:.2f}".format(accuracy))

if __name__ == "__main__":
    main()