# House Price Prediction API

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E.svg)](https://scikit-learn.org/)

A Machine Learning project focused on predicting residential property prices. 

## 🚀 Project Overview

The core of this project is a regression model trained on housing data, served through a high-performance API. It demonstrates a complete MLOps lifecycle:
1.  **Data Engineering**: Systematic cleaning and feature transformation.
2.  **Machine Learning Pipeline**: Automated encoding and model training using Scikit-Learn Pipelines.
3.  **Deployment**: A FastAPI service that provides real-time predictions.

## 🛠️ Tech Stack

* **Language**: Python 3.13+
* **API Framework**: FastAPI & Uvicorn
* **Machine Learning**: Scikit-Learn (Linear Regression, OneHotEncoder, Pipelines)
* **Data Analysis**: Pandas & NumPy
* **Environment**: Linux (Debian 13)

## 📁 Repository Structure

* `api.py`: FastAPI server implementation with prediction endpoints.
* `data_cleaning.py`: Script for handling missing values and ordinal feature encoding.
* `train_model.py`: Training pipeline that exports the final `model.pkl`.
* `predict.py`: Internal class-based logic for model loading and inference.
* `test_api.py`: Integration test script to validate API responses.
* `requirements.txt`: Project dependencies.

## ⚙️ Features & Evolution

* **Robust Data Cleaning**: Automatically handles missing values by dropping high-nullity columns and filling categorical gaps based on domain logic.
* **Automated Pipeline**: Uses Scikit-Learn's `Pipeline` to ensure that data transformations (like OneHotEncoding) are consistently applied during both training and inference.
* **Real-time Inference**: The API accepts complex JSON inputs and returns predicted prices in seconds.
* **Environment Agnostic**: Developed on Debian 13, ensuring stability in professional Linux-based server environments.

## 🚦 How to Run

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
