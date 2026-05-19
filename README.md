# Machine Learning-Driven Prediction and Optimization of Energy Storage Performance in Lead-Free Relaxor Ferroelectric Ceramics

## 1\. Project Description

This is an example implementation of six predictive machine learning models used to estimate the multifunctional properties of BaTiO3-based lead-free ceramics for optimized materials design. The models implement an extensive database of 280 experimental samples and 32 input features, collected from 53 experimental campaigns available in the literature. The models included in this repository are:

* Linear Regression (LR)
* Support Vector Regression (SVR)
* Decision Tree (DT)
* Random Forest (RF)
* Gradient Boosting Regression Tree (GBRT)
* Extreme Gradient Boosting Tree (XGB)

## 2\. Database

The code presented in this repository focuses on the prediction of the recoverable energy density (Wrec) and energy storage efficiency (η). The complete database is available open-source at: "Dataset BaTiO3.xlsx" file within this repository.

## 3\. Model Training

Each model is trained and tested using a Repeated k-fold cross-validation approach. A 10-fold split is implemented and repeated ten times, representing a 100-fold approach, with each dataset randomly reshuffled between repetitions.

## 4\. Instructions

Follow the instructions below to execute the script and build the models:

1. Follow these steps to run the regression pipeline for all six algorithms (LR, SVR, DT, RF, GBRT, XGB): Unzip the downloaded project folder, keeping all files together without changing the relative paths between them.
2. Set up Python environment: Open a Python environment (e.g., Anaconda, virtualenv). Upload or navigate to the project folder in your environment.
3. Install dependencies: Ensure all required Python packages are installed. See requirements.txt for the full list of dependencies. To install, run: pip install -r requirements.txt
4. Run the pipeline - Execute the main script: python run.py
5. This will train and test all six models with hyperparameter tuning.
6. Important note on runtime: The pipeline uses Repeated K-Fold cross-validation for robust training. Depending on your machine’s CPU, the script may take several hours to complete fully.



## 5\. Code Structure

The Wrec_Neta.py file is organized in the following format:

* Data Preparation.
* Model building, repeated K-fold cross-validation, and predictions.
* Model performance and error evaluation.
* Hyperparameter optimization.

The Dataset BaTiO3.xlsx file is always required to execute the Wrec_Neta.py file.

## 6\. Dependencies

The application includes the following dependencies to run:

* Python == 3.11.0
* Pandas> == 1.2.0
* numpy> == 1.20
* matplotlib> == 3.3
* seaborn> == 0.11
* TensorFlow == 2.16.1
* Keras == 3.3.3
* Scikit-Learn == 1.3.2
* PyGAM == 0.9.1
* PyEarth == 0.1.25
* xgboost> == 1.7.6
