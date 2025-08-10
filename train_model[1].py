# train_model.py
# Full training pipeline for Capstone 2 project.
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

def detect_target(df):
    if '-1' in df.columns:
        return '-1'
    for col in df.columns:
        vals = set(df[col].dropna().unique())
        if vals.issubset({-1,1,'-1','1',-1.0,1.0}):
            return col
    return df.columns[-1]

def build_and_train(data_path, out_path):
    df = pd.read_csv(data_path)
    target_col = detect_target(df)
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df[target_col] = df[target_col].replace(-1, 0).replace(1, 1)
    df = df.dropna(subset=[target_col])
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    X = numeric_df.drop(columns=[target_col])
    y = numeric_df[target_col]
    X = X.fillna(X.median())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sm = SMOTE(random_state=42)
    from collections import Counter
    train_counts = Counter(y_train)
    if min(train_counts.values()) / max(train_counts.values()) < 0.8:
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        smote_used = True
    else:
        X_train_res, y_train_res = X_train, y_train
        smote_used = False
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    n_components = min(50, X_train_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    models = {
        "RandomForest": (RandomForestClassifier(random_state=42), {
            'n_estimators': [100],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }),
        "SVC": (SVC(random_state=42), {
            'C': [1, 5],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale']
        }),
        "GaussianNB": (GaussianNB(), {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        })
    }
    results = {}
    best_models = {}
    for name, (estimator, params) in models.items():
        gs = GridSearchCV(estimator, params, cv=3, scoring='accuracy', n_jobs=-1)
        gs.fit(X_train_pca, y_train_res)
        best = gs.best_estimator_
        preds = best.predict(X_test_pca)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)
        results[name] = {'best_params': gs.best_params_, 'test_accuracy': acc, 'report': report}
        best_models[name] = best
    best_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    best_model = best_models[best_name]
    artifact = {'scaler': scaler, 'pca': pca, 'model': best_model, 'target_col': target_col, 'feature_columns': list(X.columns), 'smote_used': smote_used}
    joblib.dump(artifact, out_path)
    print("Saved best model:", best_name)
    print("Results summary:")
    for k,v in results.items():
        print(k, v['test_accuracy'], v['best_params'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to signal-data.csv")
    parser.add_argument("--out", default="best_model.pkl", help="output path for saved model")
    args = parser.parse_args()
    build_and_train(args.data, args.out)
