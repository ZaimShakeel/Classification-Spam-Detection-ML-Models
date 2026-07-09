# Machine Learning Classification Project

This project contains a Python machine learning workflow for two tasks:

1. Multi-dataset classification using tabular training and test files
2. Spam email detection using text data from CSV files

The main script, [MLProject.py](MLProject.py), automatically finds the required data files, trains several models, evaluates them, and saves prediction outputs as text files.

## Project Overview

The project is organized around two main workflows:

- Classification task: uses files in the [classification](classification) folder
- Spam detection task: uses files in the [Spam Email Detection](Spam%20Email%20Detection) folder

The script performs the following:

- Loads training and test data from the project folders
- Cleans and prepares the data
- Scales numeric features for classification tasks
- Trains and compares multiple classifiers
- Saves predictions to output text files in the project root

## Project Structure

```text
ML Project/
├── MLProject.py
├── classification/
│   ├── TrainData1.txt
│   ├── TrainLabel1.txt
│   ├── TestData1.txt
│   ├── TrainData2.txt
│   ├── TrainLabel2.txt
│   ├── TestData2.txt
│   ├── TrainData3.txt
│   ├── TrainLabel3.txt
│   ├── TestData3.txt
│   ├── TrainData4.txt
│   ├── TrainLabel4.txt
│   ├── TestData4.txt
├── Spam Email Detection/
│   ├── spam_train1.csv
│   ├── spam_train2.csv
│   └── spam_test.csv
├── ShakeelClassification1.txt
├── ShakeelClassification2.txt
├── ShakeelClassification3.txt
├── ShakeelClassification4.txt
└── ShakeelSpam.txt
```

## What the Script Does

### 1. Classification Section

For each dataset pair in the [classification](classification) folder, the script:

- Locates the training data, training labels, and test data
- Loads the data with automatic file handling
- Cleans missing or invalid values
- Standardizes the feature values
- Splits the training data for validation
- Trains multiple classifiers and compares their performance
- Saves the best predictions to output files

The models used for classification include:

- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM) when the dataset is not too large

### 2. Spam Email Detection Section

For the spam detection workflow, the script:

- Loads the training and test CSV files from [Spam Email Detection](Spam%20Email%20Detection)
- Prepares the label/text columns
- Converts the email text into numerical features using TF-IDF
- Trains and compares multiple text classifiers
- Saves the spam predictions to an output file

The text models used include:

- Logistic Regression
- Multinomial Naive Bayes
- Support Vector Machine

## Requirements

Make sure Python is installed, then install the required packages:

```bash
pip install pandas numpy scikit-learn
```

## How to Run

From the project root, run:

```bash
python MLProject.py
```

The script will automatically:

- Change to the script directory
- Find the data files
- Run the classification workflow
- Run the spam detection workflow
- Save output files in the project root

## Output Files

The script creates prediction files in the root directory. The filenames are controlled by the `LAST_NAME` variable at the top of [MLProject.py](MLProject.py). By default, the script uses the prefix defined there when writing output files.

Typical output files include:

- Classification predictions such as `Classification1.txt`
- Spam predictions such as `Spam.txt`

## Customization

You can customize the project by editing [MLProject.py](MLProject.py):

- Change the output prefix by updating `LAST_NAME`
- Adjust model parameters if needed
- Modify the train/test split ratio
- Change the feature extraction settings for spam detection

## Notes

- The script is designed to find files by name, so the file names should remain consistent with the expected dataset naming convention.
- If a dataset is missing, the script will skip that classification task and continue with the others.
- The spam detection section expects the CSV files to contain relevant label and text columns.

## Troubleshooting

If the script does not run correctly:

- Confirm that all required Python packages are installed
- Ensure the data folders contain the expected files
- Check that the CSV/text files use recognizable column names
- Verify that you are running the script from the project directory or that Python can access the files
