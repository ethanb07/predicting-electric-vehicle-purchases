# Predicting Electric Vehicle Purchases

A machine learning project predicting whether a person is likely to buy an electric vehicle.

## Project Overview

The aim of this project is to build a model that can predict `Will_Buy_EV` from information about potential customers.

I used this project to practise the full machine learning workflow, including:

* Data exploration and cleaning
* Feature engineering
* Encoding categorical and ordinal variables
* Building preprocessing pipelines
* Training different classification models
* Cross-validation
* Hyperparameter tuning
* Evaluating model performance
* Generating predictions for the test data

## Models

I compared several models, including:

* Logistic Regression
* SGD Classifier
* HistGradientBoosting

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 score
* ROC-AUC

The HistGradientBoosting model performed best overall during evaluation.

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

The training code can then be run with:

```bash
python train.py
```

Requires data provided on [Kaggle](https://www.kaggle.com/competitions/playground-series-s6e9/data) to be saved in data/.

The predictions are saved to:

```text
submissions/submission.csv
```

## Results

My best cross-validation results were approximately:

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 0.898 |
| Precision | 0.895 |
| Recall    | 0.898 |
| F1        | 0.896 |
| ROC-AUC   | 0.941 |

## What I Learned

This project gave me practice with building a machine learning project from start to finish rather than just fitting a model to a dataset.

One of the main things I focused on was getting the preprocessing and feature engineering right, particularly when dealing with different types of categorical data.

I also learned more about using cross-validation and hyperparameter tuning to compare models properly.
