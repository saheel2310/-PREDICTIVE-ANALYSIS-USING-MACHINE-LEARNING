# -PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING
# Car Sales Price Prediction Project

## Overview

This project aims to develop a machine learning model that can accurately predict car prices based on various features such as car specifications, brand, and technical details.  The project utilizes a Random Forest Regressor model and incorporates feature engineering, selection, and hyperparameter tuning to optimize performance.

## Project Structure

The project consists of the following main components:

*   **`Car_Sales_Regression.ipynb`:** A Jupyter Notebook containing the complete data analysis and model building workflow. This notebook includes data loading, preprocessing, feature engineering, model training, evaluation, and error analysis.
*   **`README.md`:** This file, providing an overview of the project, instructions for setup, and details on data usage and model implementation.

## Setup and Installation

To run this project, you will need the following:

1.  **Python 3.6+:** Python is the primary programming language used in this project.

2.  **Required Libraries:** The following Python libraries are essential and can be installed using pip:

    ```bash
    pip install pandas scikit-learn matplotlib seaborn
    ```

3.  **Google Colab:** This project is designed to run in Google Colab, a free cloud-based Jupyter Notebook environment.

## Data

### Dataset Source

The dataset used for this project should be a CSV file containing car specifications and prices. A suitable dataset can be found on Kaggle by searching for "car price prediction dataset" or similar.

### Data Preparation

1.  **Download the Dataset:** Download the CSV file from Kaggle or your chosen source.

2.  **Upload to Google Colab:** In Google Colab, upload the CSV file using the file upload functionality.

### Dataset Columns and Description
Below is the list of dataset columns and their descriptions:

* car_ID: Unique id for each car (Numeric).
* symboling: Assigned insurance risk rating, +3 indicates risky, -3 indicates safe (Numeric).
* CarName: Name of the car manufacturer and model (Categorical).
* fueltype: Fuel type, i.e., gas or diesel (Categorical).
* aspiration: Aspiration used in a car (Categorical).
* doornumber: Number of doors in a car (Categorical).
* carbody: Body of the car (Categorical).
* drivewheel: Drive wheel type (Categorical).
* enginelocation: Location of the engine in the car (Categorical).
* wheelbase: Distance between the front and rear wheels (Numeric).
* carlength: Length of the car (Numeric).
* carwidth: Width of the car (Numeric).
* carheight: Height of the car (Numeric).
* curbweight: The weight of the car without passengers or baggage (Numeric).
* enginetype: Type of engine (Categorical).
* cylindernumber: Number of cylinders in the engine (Categorical).
* enginesize: Size of the engine (Numeric).
* fuelsystem: Fuel system of the car (Categorical).
* boreratio: Bore ratio (Numeric).
* stroke: Stroke or valve travel (Numeric).
* compressionratio: Compression ratio (Numeric).
* horsepower: Horsepower (Numeric).
* peakrpm: Peak RPM (Numeric).
* citympg: Miles per gallon in the city (Numeric).
* highwaympg: Miles per gallon on the highway (Numeric).
* price: The price of the car (Target Variable, Numeric).

## Usage

1.  **Open the Notebook:** Open `Car_Sales_Regression.ipynb` in Google Colab.

2.  **Upload the Dataset:** Run the first code cell to upload your CSV file.

3.  **Adjust File Path:** Modify the `file_path` variable in the notebook to match the name of your uploaded CSV file.  For example:

    ```python
    file_path = list(uploaded.keys())[0]
    df = pd.read_csv(file_path)
    ```

4.  **Review and Adapt Column Names:** Review the code and adapt column names to match your dataset (e.g., target variable name).

5.  **Run the Notebook:** Execute the cells sequentially, following the instructions and comments provided in the notebook.

## Key Steps in the Notebook

1.  **Data Loading:** Loads the dataset into a pandas DataFrame.

2.  **Data Exploration and Preprocessing:**
    *   Inspects the data types, missing values, and descriptive statistics.
    *   Handles missing values by imputing with the mean for numerical features and the mode for categorical features.
    *   Converts categorical features to numerical using one-hot encoding.
    *   The CarName column has been specifically taken into account, because of its large number of values

3.  **Feature Engineering:** Creates polynomial features to capture non-linear relationships.

4.  **Feature Selection:** Selects the most relevant features using Recursive Feature Elimination (RFE).

5.  **Data Splitting and Scaling:** Splits the data into training and testing sets and scales the numerical features using `StandardScaler`.

6.  **Model Training and Tuning:** Trains a Random Forest Regressor model and tunes its hyperparameters using `RandomizedSearchCV`.

7.  **Model Evaluation:** Evaluates the model's performance using MAE, MSE, and R².

8.  **Error Analysis:** Plots residuals and shows predicted vs. actual values.

## Hyperparameter Tuning

The notebook uses `RandomizedSearchCV` for hyperparameter tuning. The key hyperparameters for the Random Forest Regressor include:

*   `n_estimators`: The number of trees in the forest.
*   `max_depth`: The maximum depth of the trees.
*   `min_samples_split`: The minimum number of samples required to split an internal node.
*   `min_samples_leaf`: The minimum number of samples required to be at a leaf node.
*   `bootstrap`: Whether bootstrap samples are used when building trees.

The `param_distributions` variable defines the ranges or discrete values to search over for each hyperparameter.  The `n_iter` parameter controls the number of random combinations that `RandomizedSearchCV` will evaluate.

## Model Evaluation Metrics

The model's performance is evaluated using the following metrics:

*   **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual prices.

*   **MSE (Mean Squared Error):** The average squared difference between predicted and actual prices.

*   **R² (R-squared):** The proportion of variance in the target variable that is explained by the model.

## License

This project is licensed under the [MIT License](LICENSE). See the `LICENSE` file for details.  (Create a LICENSE file if you want to apply a license to your project.)

## Contributing

Contributions to this project are welcome! Feel free to submit pull requests with bug fixes, improvements, or new features.
