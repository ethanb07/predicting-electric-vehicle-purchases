# Imports
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# Retrieve metric names
from sklearn.metrics import get_scorer_names, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Feature Engineering

def feature_engineering(df):
    df["Total_Nearby_Charging"] = (
        df["Charging_Stations_Near_Home"]
        + df["Charging_Stations_Near_Work"]
    )
    df["Income_per_Car"] = (
        df["Annual_Income_USD"] /
        (df["Number_of_Cars_Owned"] + 1)
    )

    df["Charging_per_Commute"] = (
        df["Total_Nearby_Charging"] /
        (df["Daily_Commute_km"] + 1)
    )
    
    return df


def main():

    # Read dataframe
    df = pd.read_csv("../data/raw/train.csv", index_col = 'id')

    # Select data

    X = df.drop(columns = ['Will_Buy_EV'])
    y = df['Will_Buy_EV'].map({'Yes': 1,
                            'No': 0})

    numerical_columns = X.select_dtypes(include = 'number').columns
    nominal_columns = X.select_dtypes(exclude = 'number').drop(columns = ['Range_Anxiety_Level']).columns
    ordinal_columns = ['Range_Anxiety_Level']

    # Train Test Split

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size = 0.2,
                                                        shuffle = True,
                                                        stratify = y,
                                                        random_state = 42)

    # Preprocessing pipelines

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy = 'mean')),
        ('scaler', StandardScaler())
    ])

    nominal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('scaler', OneHotEncoder())
    ])

    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('scaler', OrdinalEncoder())
    ])

    preprocessing = ColumnTransformer([
        ('numerical', numerical_pipeline, numerical_columns),
        ('nominal', nominal_pipeline, nominal_columns),
        ('ordinal', ordinal_pipeline, ordinal_columns)
    ])

    # Model Instantiation

    models = {
        'Linear': LogisticRegression(),
        'SGD': SGDClassifier(loss = 'log_loss'),
        'HistGradientBoosting': HistGradientBoostingClassifier()
    }

    # Cross-Validation

    cv = StratifiedKFold(
        n_splits = 5,
        shuffle = True,
        random_state = 42
    )

    model_score = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessing),
            ('model', model)
        ])

        search = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv = cv,
            n_jobs = -1,
            verbose = True,
            scoring = {'accuracy': 'accuracy',
                    'precision': 'precision_weighted',
                    'recall': 'recall_weighted',
                    'f1': 'f1_weighted',
                    'roc_auc': 'roc_auc_ovr'} # One-vs-Rest ROC-AUC strategy
        )

        model_score[name] = {
            'accuracy': search['test_accuracy'].mean(),
            'precision': search['test_precision'].mean(),
            'recall': search['test_recall'].mean(),
            'f1': search['test_f1'].mean(),
            'ROC-AUC': search['test_roc_auc'].mean()
        }

    # Cross-Validation Results
    df_cv = pd.DataFrame(model_score).T
    print(df_cv)

    # Select promising models from CV
    promising_models = {
        'HistGradientBoosting': models['HistGradientBoosting'],
        'Linear': models['Linear']
    }

    # Configure the parameters for each model

    rng = np.random.default_rng(seed = 1)

    param_distributions = {
        "HistGradientBoosting": {
            "model__learning_rate": rng.uniform(0.01, 0.19, size = 20),
            "model__max_iter": rng.integers(100, 500, size = 20),
            "model__max_leaf_nodes": rng.integers(10, 64, size = 20),
            "model__max_depth": [None, 3, 5, 7, 10],
            "model__min_samples_leaf": rng.integers(10, 50, size = 20),
            "model__l2_regularization": np.exp(
                rng.uniform(
                    np.log(1e-3),
                    np.log(10),
                    size=20)
                )
        },
        "Linear": {
            "model__C": np.exp(
                rng.uniform(
                    np.log(1e-3),
                    np.log(100),
                    size=20,)
                ),
            "model__solver": ["lbfgs", "liblinear"],
            "model__max_iter": [500, 1000, 2000]
        }
    }

    # So roughly 10% of samples will be below 1, while roughly 90% will be above 1.

    # That's why I said it would "overwhelmingly" generate larger values.

    # Why log-uniform is different

    # For regularisation, we might care about values like:

    # 0.001
    # 0.01
    # 0.1
    # 1
    # 10

    # Notice that these aren't evenly spaced numerically. They're evenly spaced in terms of multiplication by 10.

    # A log-uniform distribution treats these ranges roughly equally:

    # 0.001 ── 0.01 ── 0.1 ── 1 ── 10
    #    ×10      ×10      ×10    ×10

    # So you get approximately equal chances of landing in:

    # 0.001–0.01
    # 0.01–0.1
    # 0.1–1
    # 1–10

    # Uniform:

    # "Give every numerical distance the same probability."

    # 0    1    2    3    4    5
    # |----|----|----|----|----|

    # Log-uniform:

    # "Give every multiplicative scale the same probability."

    # 0.001   0.01   0.1    1     10
    #    |------|------|------|------|

    # Hyperparameter optimisation

    tuned_models = {}

    for name, model in promising_models.items():

        pipeline = Pipeline([
            ('preprocessor', preprocessing),
            ('model', model)
        ])

        search = RandomizedSearchCV(
            pipeline,
            param_distributions = param_distributions[name],
            n_iter = 10,
            n_jobs = 4,
            scoring = 'roc_auc',
            verbose = True,
            random_state = 42
        )

        search.fit(X_train, y_train)

        tuned_model = search.best_estimator_
        tuned_models[name] = {
            'model': tuned_model,
            'roc_auc': search.best_score_
        }

    # Output model scores after hyperparameter optimisation
    df_hyperparameter_optimised = pd.DataFrame(tuned_models).T
    print(df_hyperparameter_optimised)

    # Select final model
    final_model = tuned_models['HistGradientBoosting']['model']
    predictions = final_model.predict(X_test)

    # Scoring model
    print(confusion_matrix(y_test, predictions))

    classification_report_df = pd.DataFrame(classification_report(y_test, predictions, output_dict = True))
    print(classification_report_df)

    # Scoring model
    scoring_df = pd.DataFrame({
        'Accuracy': [accuracy_score(y_test, predictions)],
        'Precision': [precision_score(y_test, predictions)],
        'Recall': [recall_score(y_test, predictions)],
        'f1': [f1_score(y_test, predictions)],
        'ROC-AUC': [roc_auc_score(y_test, predictions)]
    }, index = ['HistGradientBoosting'])

    print(scoring_df)

    import joblib

    joblib.dump(final_model, "../models/model.pkl")

if __name__ == "__main__":
    main()