# Sparkify Churn Prediction - Using spark


### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#projectmotivation)
3. [File Descriptions](#filedescriptions)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensingauthorsandacknowledgements)

## Installation <a name="installation"></a>

Conda python3 with these additional requirements :
- Pyspark
- Pandas
- httpagentparser
- Seaborn

Please note you need to install Spark before running the scripts. On this project I used spark from Udacity workspace for small dataset and applied the code on full dataset using [Amazon EMR](https://towardsdatascience.com/getting-started-with-pyspark-on-amazon-emr-c85154b6b921). 

## Project Motivation <a name="projectmotivation"></a>
This project will help Sparkify streaming service to retain customer who will be churn in the future.

## File Descriptions <a name="filedescriptions"></a>
- Sparkify.ipynb     : Jupyter notebook using mini dataset (128 MB) and python3 kernel, all features are used in this notebook.
- Sparkify-emr.py    : Jupyter notebook using full dataset (12 GB) and pyspark kernel, only subset of features are used in this notebook to overcome performance of computation power in Amazon EMR.
- sparkify.py        : Python script to run the training and testing (without saving the model) and print the F1 & accuracy.

## Result <a name="results"></a>
The scripts are classified into 2, script for small dataset and full dataset. The modelling approach is different between those two as I reduce the numbers of features in full dataset to make the performance better.

4 machine learning models are used : Naive Bayes, Random Forrest, Gradient Boosting and Logistic Regression. On small dataset, Logistic Regression has the highest F1-Score with 0.772 while Naive Bayes has the lowest F1-Score with 0.61. On full dataset, Gradient Boosting has the highest F1-score with 0.844. One of the possible explanation why Gradient Boosting do well in Full dataset is because in full dataset number of userId is enough to generalized the model (22,278 vs 225 userId). 


Detailed explanation can be found on Medium link [here](NA).

## Licensing, Authors, and Acknowledgements  <a name="licensingauthorsandacknowledgements"></a>
Thanks to [udacity](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t) for providing the data and workspace for this Sparkify Project