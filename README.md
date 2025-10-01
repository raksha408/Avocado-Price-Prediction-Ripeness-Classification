# Avocado Price Prediction & Ripeness Classification

This project presents a **dual framework** for comprehensive avocado analysis.  
It enables users to:

- **Predict avocado prices** using historical sales data and regression models.  
- **Classify the ripeness stage** of avocados from images using deep learning models (CNNs).  

The system integrates both modules into a **unified Streamlit interface** for end-to-end market and quality assessment.


## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Folder Structure](#folder-structure)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Results](#results)  
7. [Dependencies](#dependencies)  
8. [Project Workflow](#project-workflow)  
9. [Future Work](#future-work)  
10. [Conclusion](#conclusion)  
11. [Author](#author)  
12. [Acknowledgments](#acknowledgments)


## Project Overview

The project consists of two main modules:

### Price Prediction

Uses regression models including **CatBoost**, **Decision Tree**, **ExtraTree**, and **RandomForest** to estimate the average retail price of avocados. Predictions are based on historical sales data, PLU codes, region, and type (**conventional** or **organic**).

### Ripeness Classification

Uses **Convolutional Neural Networks (CNNs)** to classify avocado images into five ripeness stages:

1. **Underripe**  
2. **Breaking**  
3. **Ripe (First Stage)**  
4. **Ripe (Second Stage)**  
5. **Overripe**

The combined system provides an end-to-end solution for **avocado market analysis** and **quality assessment**.


## Features

- Predict avocado prices for a given **date**, **type**, and **region**.  
- Classify the **ripeness stage** of any avocado image.  
- Visualize results using **scatter plots**, **accuracy/loss curves**, and **confusion matrices**.  
- Modular structure for **easy extension** and **maintenance**.


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


## Usage

### Run the Streamlit App

1. **Start the unified interface:**  
   ```bash
   streamlit run main_interface.py

2. **Price Prediction:**
   ```bash
   Enter avocado type, region, and date to get the predicted average price.

3. **Ripeness Classification:**
   ```bash
   Upload an avocado image to classify its ripeness stage.

## Results

- **Price Prediction:**  
  Regression metrics and scatter plots are stored in `assets/plots/`.

- **Ripeness Classification:**  
  Accuracy, loss curves, and confusion matrices are stored in `assets/matrix/`.


## Dependencies

- **Python:** 3.10+  
- **Libraries:**  
  - Pandas, NumPy  
  - Scikit-learn, CatBoost  
  - TensorFlow / Keras  
  - Streamlit  
  - Matplotlib, Seaborn  
  - Pillow  

All dependencies are listed in `req.txt`.


## Project Workflow

The project is organized into two main modules: **Price Prediction** and **Ripeness Classification**.  
Each module has its own **data**, **models**, and **preprocessing steps**, allowing independent development and evaluation.  
The unified Streamlit interface integrates both modules for end-to-end avocado analysis.

## 1. Price Prediction Module

**Dataset:** Collected from Kaggle, containing **18,249 rows** with 13 columns.

### Data Preparation
- Load historical avocado sales data from CSV files.  
- Convert **date fields** into ordinal format.  
- Encode categorical features like **type** and **region**.

### Model Training
- Candidate regressors: **CatBoost, Decision Tree, ExtraTree, RandomForest**.  
- Perform **hyperparameter tuning** for each model.

### Prediction & Evaluation
- The final model predicts **average avocado prices**.  
- Evaluation metrics: **MAE, MSE, RMSE, R²**.  
- Generate **scatter plots** and **feature importance charts**.

---

## 2. Ripeness Classification Module

**Dataset:** Mendeley 'Hass' Avocado Ripening Photographic Dataset with **14,710 labeled images** (800×800 pixels).  
- 478 Hass avocados, acquired three days post-harvest.  
- Categorized into three storage groups:  
  - **T10:** 10 ºC, 85% RH  
  - **T20:** 20 ºC, 85% RH  
  - **Tamb:** Ambient conditions  
- Daily photographs of each sample captured from two opposite sides.  
- Each image linked to a detailed **Ripening Index (5 stages)**:  
  1. Underripe  
  2. Breaking  
  3. Ripe (First Stage)  
  4. Ripe (Second Stage)  
  5. Overripe  
- Stage 4 marks the end of shelf-life; stage 5 indicates beyond prime.  
- An Excel spreadsheet accompanies the dataset with image filenames, sample number, storage group, ripening stage, day, and side.

### Data Preparation
- Resize and normalize images.  
- Apply **augmentation and oversampling** only to the training set.  
- Split dataset into **train, validation, and test** sets.

### Model Training
- CNN architectures: **EfficientNetB4, MobileNetV2, ResNet50, DenseNet201**.  
- Apply **hyperparameter tuning** and **early stopping**.

### Classification & Evaluation
- Classify avocado images into **5 ripeness stages**.  
- Evaluation metrics: **accuracy, precision, recall, F1-score**.  
- Visualizations include **confusion matrices** and **training/validation curves**.


## 3. Streamlit Interface

- Users can input avocado **type**, **region**, and **date** to predict prices or upload images for **ripeness classification**.  
- Results are displayed **interactively** in the interface.  
- Outputs are saved in the corresponding directories:  
  - Price prediction plots: `assets/plots/`  
  - Ripeness classification results: `assets/matrix/`

## Future Work

- Integrate real-time market data for dynamic price prediction.  
- Expand the ripeness classification dataset with more avocado varieties.  
- Implement a mobile app interface for broader accessibility.  
- Explore advanced deep learning architectures for improved accuracy.

## Conclusion

This project provides an end-to-end solution for **avocado market analysis** and **quality assessment**, combining price prediction and ripeness classification.  
The modular structure ensures easy extension and maintenance for future enhancements.

## Author

**Shriraksha [Kulkarni]**    
- GitHub: [https://github.com/raksha408](https://github.com/raksha408)

## Acknowledgments

- Kaggle for avocado price datasets.  
- Mendeley for the 'Hass' Avocado Ripening Photographic Dataset.  
- Open-source libraries and frameworks used: Pandas, NumPy, Scikit-learn, CatBoost, TensorFlow/Keras, Streamlit, Matplotlib, Seaborn, Pillow.



