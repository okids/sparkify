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

## Project Motivation <a name="projectmotivation"></a>
This project will help Sparkify streaming service to retain customer who will be churn in the future.

## File Descriptions <a name="filedescriptions"></a>
- Sparkify.ipynb     : Jupyter notebook using mini dataset and python3 kernel, all features are used in this notebook.
- Sparkify-emr.py    : Jupyter notebook using full dataset and pyspark kernel, only subset of features are used in this notebook to overcome performance of Amazon EMR.
- sparkify.py        : Python script to run the training and testing (without saving the model) and print the F1 & accuracy.

## Result <a name="results"></a>


Result can be found on Medium link [here](NA).

## Licensing, Authors, and Acknowledgements  <a name="licensingauthorsandacknowledgements"></a>
Thanks to [udacity](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t) for providing the data and jupyter notebook workspace for this Sparkify Project