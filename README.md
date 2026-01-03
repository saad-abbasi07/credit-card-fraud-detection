# Credit Card Fraud Detection

This project detects fraudulent credit card transactions using machine learning. It demonstrates:

- Data preprocessing and scaling
- Handling class imbalance with SMOTE
- Model training with Keras and Random Forest
- Evaluation metrics: accuracy, precision, recall, F1-score, ROC-AUC
- Deployment-ready setup with `app.py`, `requirements.txt`, and `environment.yml`

## Files

- `CreditCardFraud.ipynb` – Main notebook for exploration and training  
- `app.py` – Python script for training and saving Keras model  
- `requirements.txt` – Pip dependencies for deployment  
- `environment.yml` – Conda environment file  

## Dataset

The dataset used is [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle, containing anonymized transactions with highly imbalanced classes.

## Usage

1. Install dependencies via pip or conda  
2. Run `app.py` to train the model and save `fraud_model.h5` and `scaler.save`  
3. Deploy via a web framework (Flask/FastAPI) or cloud service (Render, Heroku, etc.)

