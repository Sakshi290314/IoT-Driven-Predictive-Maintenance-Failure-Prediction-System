# 02_preprocessing_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def preprocess_pipeline(df, target_column="machine_failure"):

    # ---------------------------
    # 1. Separate Features & Target
    # ---------------------------
    X = df.drop([target_column, "failure_type"], axis=1, errors="ignore")
    y = df[target_column]

    # ---------------------------
    # 2. Remove ID Columns (if any)
    # ---------------------------
    id_columns = ["machine_id", "product_id", "uid"]
    X = X.drop(columns=[col for col in id_columns if col in X.columns], errors="ignore")

    # ---------------------------
    # 3. Identify Column Types
    # ---------------------------
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    # ---------------------------
    # 4. Define Transformers
    # ---------------------------
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # ---------------------------
    # 5. Train-Test Split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ---------------------------
    # 6. Apply SMOTE + Preprocessing
    # ---------------------------
    pipeline = ImbPipeline(steps=[
        ("preprocessing", preprocessor),
        ("smote", SMOTE(random_state=42))
    ])

    X_train_processed, y_train_resampled = pipeline.fit_resample(X_train, y_train)

    # Only transform test (no SMOTE)
    X_test_processed = preprocessor.transform(X_test)

    print("Preprocessing Complete ✅")
    print("Training Shape:", X_train_processed.shape)
    print("Testing Shape:", X_test_processed.shape)

    return X_train_processed, X_test_processed, y_train_resampled, y_test
