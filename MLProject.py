
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option("display.max_rows", 5)
pd.set_option("display.max_columns", 5)

LAST_NAME = "Khan"


# UTILITY FUNCTIONS

def find_file(base_path, basename):
    """Find a file (.csv, .xlsx, or .txt) recursively by its base name."""
    for root, _, files in os.walk(base_path):
        for f in files:
            name, ext = os.path.splitext(f)
            if name.lower() == basename.lower() and ext.lower() in [".csv", ".xlsx", ".txt"]:
                return os.path.join(root, f)
    return None


def read_any(path):
    """Safely read .csv, .xlsx, or .txt without hogging memory."""
    if path.endswith(".csv"):
        df = pd.read_csv(path, low_memory=False)
        print(f"Loaded {os.path.basename(path)} ({df.shape[0]} rows, {df.shape[1]} columns)")
        return df
    elif path.endswith(".txt"):
        # Try multiple separators or fallback
        for sep in [",", "\t", " "]:
            try:
                df = pd.read_csv(path, sep=sep, header=None, low_memory=False)
                if df.shape[1] > 1:
                    print(f"Loaded {os.path.basename(path)} ({df.shape[0]} rows, {df.shape[1]} columns, sep='{sep}')")
                    return df
            except Exception:
                continue
        print(f"Fallback: reading {os.path.basename(path)} as fixed-width text.")
        return pd.read_fwf(path, header=None)
    elif path.endswith(".xlsx"):
        df = pd.read_excel(path)
        print(f"Loaded {os.path.basename(path)} ({df.shape[0]} rows, {df.shape[1]} columns)")
        return df
    else:
        raise ValueError(f"Unsupported file type: {path}")


def clean_data(df):
    """Ensure all numeric, fill missing values robustly."""
    df = df.replace([np.inf, -np.inf, 1.0e+99, -1.0e+99], np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")        # force numeric
    df = df.fillna(df.mean(numeric_only=True))           # fill numeric NaNs with mean
    df = df.fillna(0)                                   # fallback for leftover NaNs
    return df


# CLASSIFICATION SECTION 

def run_classification_tasks(base_path):
    print("=== Running Classification Tasks ===")

    for i in range(1, 5):
        print(f"\n--- Loading Dataset {i} ---")
        td = find_file(base_path, f"TrainData{i}")
        tl = find_file(base_path, f"TrainLabel{i}")
        tst = find_file(base_path, f"TestData{i}")

        if not td or not tl or not tst:
            print(f"Dataset {i} missing — skipping.\n")
            continue

        X_train = read_any(td)
        y_train = read_any(tl)
        X_test = read_any(tst)

        # Use first column if multiple exist
        if y_train.shape[1] > 1:
            y_train = y_train.iloc[:, 0]
        y_train = y_train.values.ravel()

        X_train = clean_data(X_train)
        X_test = clean_data(X_test)

        # Align columns
        X_train.columns = range(X_train.shape[1])
        X_test.columns = range(X_test.shape[1])

        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Split for validation
        Xtr, Xval, ytr, yval = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

        # Choose models dynamically
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5)
        }

        # Only use SVM for small/medium datasets
        if X_train.shape[1] < 5000:
            models["SVM"] = SVC(kernel="rbf", probability=True)

        best_model, best_score = None, -1
        for name, model in models.items():
            model.fit(Xtr, ytr)
            preds = model.predict(Xval)
            f1 = f1_score(yval, preds, average="weighted")
            acc = accuracy_score(yval, preds)
            print(f"{name} | Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
            if f1 > best_score:
                best_score, best_model = f1, model

        print(f"Best Model for Dataset {i}: {type(best_model).__name__} (F1={best_score:.4f})")

        preds_final = best_model.predict(X_test_scaled)
        np.savetxt(f"{LAST_NAME}Classification{i}.txt", preds_final, fmt="%d")
        print(f"Saved {LAST_NAME}Classification{i}.txt")

    print("\nClassification section complete.\n")


# SPAM EMAIL DETECTION SECTION 


def run_spam_detection(base_path):
    print("=== Running Spam Email Detection ===")

    train1 = find_file(base_path, "spam_train1")
    train2 = find_file(base_path, "spam_train2")
    test = find_file(base_path, "spam_test")

    if not train1 or not train2 or not test:
        raise FileNotFoundError("Could not find spam_train1, spam_train2, or spam_test files.")

    print(f"Found: \n{train1}\n{train2}\n{test}\n")

    train1_data = read_any(train1)
    train2_data = read_any(train2)
    test_data = read_any(test)

    print("Train1 columns:", list(train1_data.columns))
    print("Train2 columns:", list(train2_data.columns))
    print("Test columns:", list(test_data.columns))

    train1_data = train1_data.rename(columns={'v1': 'label', 'v2': 'text'})
    if 'Unnamed: 2' in train1_data.columns:
        train1_data = train1_data[['label', 'text']]

    if 'label' not in train2_data.columns or 'text' not in train2_data.columns:
        possible_label = [c for c in train2_data.columns if "label" in c.lower()]
        possible_text = [c for c in train2_data.columns if "text" in c.lower()]
        if possible_label and possible_text:
            train2_data = train2_data.rename(columns={possible_label[0]: "label", possible_text[0]: "text"})

    test_data = test_data.rename(columns={'message': 'text'})

    X_train = pd.concat([train1_data, train2_data], ignore_index=True)
    y_train = X_train['label'].astype(str).values.ravel()
    X_train_text = X_train['text'].astype(str)
    X_test_text = test_data['text'].astype(str)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "MultinomialNB": MultinomialNB(),
        "SVM": SVC(kernel="linear", probability=True)
    }

    Xtr, Xval, ytr, yval = train_test_split(X_train_tfidf, y_train, test_size=0.2, random_state=42)
    best_model, best_score = None, -1

    print("\nModel Performance:")
    for name, model in models.items():
        model.fit(Xtr, ytr)
        preds = model.predict(Xval)
        f1 = f1_score(yval, preds, average='weighted')
        acc = accuracy_score(yval, preds)
        print(f"{name} | Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
        if f1 > best_score:
            best_score, best_model = f1, model

    print(f"\nBest Spam Model: {type(best_model).__name__} (F1={best_score:.4f})")

    preds_final = best_model.predict(X_test_tfidf)
    np.savetxt(f"{LAST_NAME}Spam.txt", preds_final, fmt="%s")
    print(f"Spam predictions saved to {LAST_NAME}Spam.txt\n")



# MAIN


if __name__ == "__main__":
    print("Starting Machine Learning Project...\n")

    # Always run from script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    classification_path = "classification"
    spam_path = "Spam Email Detection"

    run_classification_tasks(classification_path)
    run_spam_detection(spam_path)

    print("All tasks completed successfully! Predictions are ready.")
