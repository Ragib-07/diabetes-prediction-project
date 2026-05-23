# Diabetes Prediction and Health Monitoring System

A complete beginner-to-intermediate Python machine learning project that predicts diabetes risk from health-related inputs and provides simple health monitoring suggestions.

> **Important medical disclaimer:** This project is for education, portfolio, and final-year demonstration purposes only. It is not a medical diagnosis system. Always consult a qualified healthcare professional for medical advice.

## Project Features

- Loads and cleans a diabetes prediction dataset
- Handles missing/invalid zero values in medical columns
- Trains and compares multiple machine learning models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine
  - K-Nearest Neighbors
- Evaluates models using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix
- Saves the best model as `diabetes_model.pkl`
- Provides a clean Streamlit web application
- Predicts diabetes risk from user input
- Shows risk level: Low, Medium, or High
- Gives basic health monitoring suggestions based on BMI, glucose, blood pressure, and age
- Saves useful charts in the `screenshots/` folder

## Project Structure

```text
diabetes_prediction_project/
│
├── app.py
├── model_training.py
├── diabetes_model.pkl
├── model_metrics.json
├── dataset/
│   └── diabetes.csv
├── requirements.txt
├── README.md
└── screenshots/
    ├── model_comparison.png
    ├── confusion_matrix_best_model.png
    └── correlation_heatmap.png
```

## Dataset

The project uses a CSV with the same columns as the Pima Indians Diabetes Dataset:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome

This package includes a demo CSV so the project runs immediately. For final submission or research-quality use, replace `dataset/diabetes.csv` with the real Pima Indians Diabetes Dataset or run:

```bash
python model_training.py --download
```

That command attempts to download a public diabetes CSV and then trains the model.

## Installation

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## How to Train the Model

Run:

```bash
python model_training.py
```

This will:

1. Load `dataset/diabetes.csv`
2. Clean the dataset
3. Replace impossible zero values with missing values
4. Split the data into training and testing sets
5. Train four machine learning models
6. Compare their performance
7. Save the best model as `diabetes_model.pkl`
8. Save charts in the `screenshots/` folder

## How to Run the Streamlit App

Run:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How the Project Works

### 1. Data Loading

`model_training.py` reads the diabetes CSV file from the `dataset/` folder. If the file is missing, it attempts to download a public CSV. If downloading fails, it creates a demo dataset with the same column structure.

### 2. Data Cleaning

Some medical columns should not realistically have zero values. For example, glucose, blood pressure, insulin, skin thickness, and BMI cannot usually be zero in real measurements. The script replaces those zero values with missing values and uses median imputation during model training.

### 3. Model Training

The script trains four models:

- Logistic Regression
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors

Each model is trained using a Scikit-learn pipeline. Pipelines are useful because they keep preprocessing and model prediction together.

### 4. Model Evaluation

The models are compared using accuracy, precision, recall, and F1-score. The best model is selected mainly by F1-score, then by accuracy.

### 5. Model Saving

The best model and useful metadata are saved in `diabetes_model.pkl` using Joblib. The Streamlit app loads this file to make predictions.

### 6. Web App Prediction

The user enters health details in the sidebar. The app sends those values to the trained model and displays:

- Prediction result
- Estimated probability
- Risk level
- General health suggestions
- Basic health indicator chart

## How to Test Predictions

After running the app, try these sample inputs:

### Lower-risk example

```text
Pregnancies: 1
Glucose: 95
BloodPressure: 70
SkinThickness: 20
Insulin: 80
BMI: 23.5
DiabetesPedigreeFunction: 0.30
Age: 25
```

### Higher-risk example

```text
Pregnancies: 6
Glucose: 180
BloodPressure: 90
SkinThickness: 35
Insulin: 170
BMI: 38.0
DiabetesPedigreeFunction: 0.90
Age: 55
```

## GitHub Portfolio Tips

To make this project look more professional on GitHub:

1. Add screenshots of the Streamlit app to the `screenshots/` folder.
2. Add a short project demo video or GIF.
3. Mention model metrics in the README.
4. Add a clear medical disclaimer.
5. Keep code comments simple and helpful.
6. Add a `requirements.txt` file so others can run it easily.

## Common Problems and Fixes

### Problem: `diabetes_model.pkl` not found

Run:

```bash
python model_training.py
```

### Problem: Streamlit command not found

Install dependencies again:

```bash
pip install -r requirements.txt
```

### Problem: Dataset columns are missing

Make sure your CSV contains exactly these columns:

```text
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
```

## Future Improvements

- Add login and user history
- Store predictions in a database
- Add more advanced charts
- Add hyperparameter tuning
- Add cross-validation
- Deploy the app on Streamlit Community Cloud
- Add SHAP or feature importance explanations
