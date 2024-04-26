# Bank Marketing Campaign Analysis

## Project Overview
This project aims to analyze a bank marketing dataset and build a predictive model to determine the likelihood of a customer subscribing to a term deposit. The dataset is sourced from a direct marketing campaign of a Portuguese banking institution.

## Data Source
The dataset can be accessed on Kaggle at the following URL: [Bank Marketing Dataset](https://www.kaggle.com/datasets/dhirajnirne/bank-marketing).

## Contributors
- Raed Abdullah Basahih
- Osama Mohammad Alkhazan
- Mohammad Mudhhi Alotaibi

## Data Exploration
We explored various aspects of the bank marketing data, analyzing distributions and relationships of key features. The data exploration phase included visualization of different attributes such as job, marital status, and education levels, as well as detailed analysis of numerical features.

### Data Cleaning
- Unified categories in the 'job' column.
- Handled missing values and checked for duplicates.
- Applied standardization to numerical fields to ensure effective model training.

### Feature Engineering
- Performed label encoding on categorical variables to prepare them for model input.
- Dropped unnecessary columns based on the correlation analysis and domain understanding.

## Modeling
The project utilizes a neural network model built with Keras and TensorFlow. The model architecture includes dense layers with dropout to prevent overfitting.

### Model Configuration
- Input Layer: Matches the number of features from the processed data.
- Hidden Layers: Multiple layers with ReLU activation and dropout regularization.
- Output Layer: Sigmoid activation to predict binary outcomes.

### Training
- The model was trained using a subset of the data with a validation split to monitor performance.
- Optimization was performed using the Adam optimizer with a binary cross-entropy loss function.

## Evaluation
The model's performance was evaluated using accuracy, precision, and ROC-AUC metrics. Detailed results from the confusion matrix and classification reports are provided.

## How to Run
To replicate this analysis and model training:
1. Clone the repository.
2. Install required dependencies:
3. Run the Jupyter notebooks or Python scripts provided.

## Dependencies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- TensorFlow
- Keras

## License
This project is licensed under the terms of the MIT license.

## Acknowledgments
- Thanks to Kaggle for providing the dataset.
- Thanks to all contributors for their insights and efforts in this project.
