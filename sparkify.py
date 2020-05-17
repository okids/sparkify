#!/usr/bin/env python
# coding: utf-8


# import libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.functions import desc,asc,col,sum as Fsum,udf


from pyspark.ml.feature import VectorAssembler,StringIndexer,StandardScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression,GBTClassifier,NaiveBayes,RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


import matplotlib.dates as mdates
import datetime
import httpagentparser


import numpy as np
import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

# create a Spark session

# Create spark session
spark = SparkSession.builder.appName("Sparkify").getOrCreate()




# Read in full sparkify dataset
def read_dataset(file_path):

    #Returns spark dataframe.

    #Parameters:
    #    file_path where the source located

    #Returns:
    #    spark df from filepath

    return spark.read.json(file_path)  

def clean_dataset(df):

    #Returns spark dataframe.

    #Parameters:
    #    spark df input

    #Returns:
    #    spark df after filtering empty userId

    return df.filter(df.userId != '')    

def get_browser(user_agent):

    #Returns browser name.

    #Parameters:
    #    user_agent string

    #Returns:
    #    r : name of the browser extracted from user agent string
    try :
        r = httpagentparser.detect(user_agent)['browser']['name']
    except:
        r = ''
    return r
    
def prepare_feature(df):

    #Returns spark dataframe with extracted features and list of features columns.

    #Parameters:
    #    spark df input

    #Returns:
    #    full_feature : spark df with more feature columns and churn columns
    #    features_col : list of feature names

    df.createOrReplaceTempView("event")

    #Create new view with churn columns
    event_with_churn = spark.sql("""
    SELECT e.*, COALESCE(churn,0) AS churn
    FROM event e
    LEFT JOIN(
            SELECT userid,1 AS churn
             FROM event 
             WHERE page = 'Cancellation Confirmation'
             GROUP BY 1
             ) USING (userid)
              """)
    event_with_churn.createOrReplaceTempView("event_with_churn")

    spark.udf.register("get_browser", get_browser)
    #Crafting the feature
    full_feature = spark.sql("""
    SELECT userId,
           churn,  
           --Gender
           MAX(CASE WHEN gender = 'M' THEN 1 ELSE 0 END)  AS gender_male,
           MAX(CASE WHEN gender = 'F' THEN 1 ELSE 0 END) AS gender_female,
           
           -- browser
           MAX(CASE WHEN get_browser(userAgent) = 'Chrome' THEN 1 ELSE 0 END)  AS chrome_user,
           MAX(CASE WHEN get_browser(userAgent) = 'Firefox' THEN 1 ELSE 0 END)  AS firefox_user,
           MAX(CASE WHEN get_browser(userAgent) = 'Safari' THEN 1 ELSE 0 END)  AS safari_user,
           MAX(CASE WHEN get_browser(userAgent) LIKE '%Microsoft%' THEN 1 ELSE 0 END)  AS ie_user,
          
           
           
           -- Paid and free session
           COUNT(DISTINCT CASE WHEN level = 'paid' THEN sessionId ELSE 0 END) AS cnt_paid_session,
           COUNT(DISTINCT CASE WHEN level = 'free' THEN sessionId ELSE 0 END) AS cnt_free_session,
           
           -- Artist and song behavior
           COUNT(DISTINCT song) AS cnt_unique_song,
           SUM(CASE WHEN page = 'NextSong' THEN 1 ELSE 0 END) AS cnt_song_played,
        
           -- Other behavior
           COUNT(DISTINCT CASE WHEN page = 'NextSong' THEN from_unixtime(ts/1000, 'yyyy-MM-dd') ELSE 0 END) AS cnt_day_played,
           COUNT(DISTINCT sessionId) AS cnt_session,
           SUM(CASE WHEN page = 'Add Friend' THEN 1 ELSE 0 END) AS cnt_add_friend,
           SUM(CASE WHEN page = 'Thumbs Down' THEN 1 ELSE 0 END) AS cnt_thumb_down,
           SUM(CASE WHEN page = 'Thumbs Up' THEN 1 ELSE 0 END) AS cnt_thumbs_up,
           SUM(CASE WHEN page = 'Home' THEN 1 ELSE 0 END) AS cnt_home,
           SUM(CASE WHEN page = 'Error' THEN 1 ELSE 0 END) AS cnt_error,
           MAX(ts-registration)/(1000*3600*365*24) AS user_age
        
    FROM event_with_churn 
    GROUP BY 1,2     
    """)


    features_col = ['gender_male','gender_female','chrome_user','firefox_user','safari_user',
                   'ie_user',
                   'cnt_free_session',
                   'cnt_unique_song',
                   'cnt_song_played','cnt_day_played','cnt_session','cnt_add_friend','cnt_thumb_down',
                   'cnt_thumbs_up','cnt_home',
                   'cnt_error','user_age'
                   ]

    return full_feature, features_col

def train_test_model(df, model, modelName, paramGrid, features_col):

    #Returns machine learning model in crossvalidator object

    #Parameters:
    #    df    : spark df input
    #    model : machine learning model from pyspark.ml.classification
    #    modelName : name of ML model
    #    paramGrid : Param grid object to run the model over few params
    #    features_col : List of feature names

    #Returns:
    #    cvModel : Machine learning model stored in Crossvalidator object


    assembler = VectorAssembler(inputCols=features_col, outputCol="feature_vec")

    df = assembler.setHandleInvalid("skip").transform(df)
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    indexer = StringIndexer(inputCol="churn", outputCol="label")
    scaler = StandardScaler(inputCol="feature_vec", outputCol="scaled_features")
    pipeline = Pipeline(stages=[indexer, scaler, model])

    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=MulticlassClassificationEvaluator(metricName='f1'),
                              numFolds=3)

    cvModel = crossval.fit(train)
    print('F1 score for {} model in train :'.format(modelName),  cvModel.avgMetrics)
    results = cvModel.transform(test)

    print("Accuracy for {} Model in test: ".format(modelName), results.filter(results.label == results.prediction).count()/ results.count())
    evaluator = MulticlassClassificationEvaluator(metricName='f1')
    score = evaluator.evaluate(results)
    print("F1 score for {} model in test : ".format(modelName), score)
    return cvModel




event_data = "mini_sparkify_event_data.json"
df = read_dataset(event_data)
df_cleaned = clean_dataset(df)
full_feature, features_col = prepare_feature(df_cleaned)



# Logistic Regression
lr =  LogisticRegression(maxIter=10, regParam=0.0, elasticNetParam=0, featuresCol='scaled_features')
paramGrid_lr = ParamGridBuilder().addGrid(lr.regParam,[0.0, 0.1, 1]).build()
cvModel_lr = train_test_model(full_feature, lr, 'Logistic Regression',paramGrid_lr,features_col)


# Gradient Boosting
gbt = GBTClassifier(labelCol="label", featuresCol="scaled_features", maxIter=10)
paramGrid_gbt = ParamGridBuilder().build()
cvModel_gbt = train_test_model(full_feature, gbt, 'Gradient Boosting',paramGrid_gbt,features_col)



#Naive Bayes
nb = NaiveBayes(smoothing=1.0, modelType="multinomial",featuresCol='scaled_features')
paramGrid_nb = ParamGridBuilder().build()
cvModel_nb = train_test_model(full_feature, nb, 'Naive Bayes',paramGrid_nb,features_col)



#Random Forrest
rf = RandomForestClassifier(labelCol="label", featuresCol="scaled_features", numTrees=10)
paramGrid_rf = ParamGridBuilder().build()
cvModel_rf = train_test_model(full_feature, rf, 'Random Forrest',paramGrid_rf,features_col)






