# Avocado Price Prediction & Ripeness Classification

This project provides a dual framework for analyzing avocados. It allows users to predict avocado prices using historical data and classify the ripeness stage of avocados from images using deep learning models. 


# Table of Contents

Project Overview

Features

Folder Structure

Installation

Usage

Results

Dependencies

Project Workflow

Author 


# Project Overview

The project consists of two main modules:

# Price Prediction:

Uses regression models (CatBoost, Decision Tree, ExtraTree, RandomForest) to estimate the average retail price of avocados based on historical sales, PLU codes, region, and type (conventional or organic).

# Ripeness Classification:

Uses convolutional neural networks (CNNs) to classify avocado images into five ripeness stages:

1.Underripe

2.Breaking

3.Ripe (First Stage)

4.Ripe (Second Stage)

5.Overripe

The combined system provides an end-to-end solution for avocado market analysis and quality assessment. 


# Features

Predict avocado prices for a given date, type, and region.

Classify the ripeness stage of any avocado image.

Visualize results using scatter plots, accuracy/loss curves, and confusion matrices.

Modular structure for easy extension and maintenance.


# Folder Structure

- **data/**: Contains the dataset files used for price prediction and ripeness classification.  
- **notebooks/**: Jupyter notebooks for data exploration, preprocessing, and model experimentation.  
- **src/**: Python source code for the project.  
  - `preprocess.py` → preprocessing functions for price and image data.  
  - `price_model.py` → training and evaluation scripts for price prediction models.  
  - `ripeness_model.py` → CNN model training and evaluation for ripeness classification.  
  - `app.py` → Streamlit application that integrates both modules.  
- **results/**: Stores generated outputs such as evaluation metrics, confusion matrices, and training/validation curves.  
- **requirements.txt**: List of all dependencies required to run the project.  
- **README.md**: Project documentation, setup instructions, and workflow description.  
- **.gitignore**: Specifies files and directories to be ignored by version control.  


## Installation  

1. **Clone the repository:**  

   ```bash
   git clone https://github.com/raksha408/Avocado_Project.git
   cd Avocado_Project

2. **Create a virtual environment:**  

   ```bash
   python -m venv interface_avocado

3. **Activate the virtual environment:**  

   - **Windows:**  
     ```bash
     interface_avocado\Scripts\activate
     ```

   - **Linux / Mac:**  
     ```bash
     source interface_avocado/bin/activate
     ```

4. **Install dependencies:**

   pip install -r req.txt


# Usage

# Run the Streamlit App

Start the unified interface: streamlit run main_interface.py

Price Prediction: Enter avocado type, region, and date to get the predicted average price.

Ripeness Classification: Upload an avocado image to classify its ripeness stage. 

# Results
Price Prediction: Regression metrics and scatter plots are stored in assets/plots/.

Ripeness Classification: Accuracy, loss curves, and confusion matrices are stored in assets/matrix/. 

# Dependencies

Python 3.10+

Pandas, NumPy

Scikit-learn, CatBoost

TensorFlow / Keras

Streamlit

Matplotlib, Seaborn

Pillow

All dependencies are listed in req.txt.

# Project Workflow
The project is organized into two main modules: Price Prediction and Ripeness Classification, each with its own data, models, and preprocessing steps.

# 1. Price Prediction Module

Dataset is collected from Kaggle.It has 18249 rows with 13 columns.

Data Preparation

Historical avocado sales data is loaded from CSV files.

Date fields are converted into ordinal format.

Categorical features like type and region are encoded.

Model Training

Candidate regressors: CatBoost, Decision Tree, ExtraTree, RandomForest.

Hyperparameter tuning is performed for each model.

Prediction & Evaluation

The final model predicts average avocado prices.

Evaluation metrics: MAE, MSE, RMSE, R².

Scatter plots and feature importance charts are generated.

# 2. Ripeness Classification Module

Dataset is collected from mendeley 'Hass' Avocado Ripening Photographic Dataset

This dataset consists of 14,710 labeled photographs of Hass avocados (Persea Americana Mill. cv Hass), resized to 800 x 800 pixels and saved in the .jpg format, designed to facilitate the development of deep 
learning models for predicting ripening stages and estimating shelf-life.

A total of 478 Hass avocados were acquired three days post-harvest and categorized into three storage groups: T10 at 10 ºC with 85% relative humidity (RH); T20 at 20 ºC with 85% RH; and Tamb under ambient 

conditions.

The ripening process of these avocados was documented through daily photographs (2 of each sample of opposite sides), each linked to a detailed Ripening Index with five stages: 

1 - Underripe;

2 - Breaking;

3 - Ripe (First Stage);

4 - Ripe (Second Stage);

5 - Overripe.

The fourth stage marked the end of the avocados' shelf-life, with any sample classified as stage 5 considered beyond its prime. Time stamps on each photograph enable the tracking of the duration from any 

photograph, to the point each avocado reached the end of its shelf-life.

Additionally, an Excel spreadsheet accompanies the dataset, listing all photograph filenames, their corresponding sample number, storage group, classification according to the 5-stage Ripening Index, the day the 

photograph was taken relative to the start of the experiment, and the side of the fruit photographed (a or b).

Data Preparation

Images are resized and normalized.

Augmentation and oversampling applied only to training set.

Dataset split: train, validation, and test.

Model Training

CNN architectures: EfficientNetB4, MobileNetV2, ResNet50, DenseNet201.

Hyperparameter tuning and early stopping applied.

Classification & Evaluation

The model classifies avocado images into 5 ripeness stages.

Evaluation metrics: accuracy, precision, recall, F1-score.

Visualizations include confusion matrices and training/validation curves.

# 3. Streamlit Interface
Users can input details to predict prices or upload images for ripeness classification.

Results are displayed interactively and saved in the assets/plots or assets/matrix directories.

# Author
Shriraksha Kulkarni

Aspiring Data Analyst & Machine Learning Enthusiast

GitHub: raksha408


