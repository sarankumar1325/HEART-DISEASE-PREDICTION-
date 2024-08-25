# HEART-DISEASE-PREDICTOR

This project is a web-based application designed to predict the likelihood of an individual having heart disease based on input data. The model used for prediction is trained on a dataset containing various health metrics and patient information. The project involves exploratory data analysis (EDA), machine learning model training, evaluation, and deployment using Streamlit.

## Project Structure

The project is organized into the following directories:

- **data/**: Contains the dataset used for training the model.
- **EDA/**: Contains notebooks and scripts used for exploratory data analysis.
- **training/**: Includes scripts for preprocessing, model training, and evaluation.
- **models/**: Stores the trained machine learning models.
- **streamlit/**: Contains the main application code for the Streamlit web app.
- **readme/**: Additional documentation files.
- **requirements.txt**: A list of Python packages required to run the project.

## About the Dataset

The dataset used for this project is the Heart Failure Prediction dataset, which was sourced from Kaggle. The dataset includes features such as age, gender, blood pressure, cholesterol levels, and more. Although the dataset is relatively small, it is used here for demonstration purposes.

**Dataset Source**: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data)

## Exploratory Data Analysis

Before building the prediction model, it is crucial to understand the data. The EDA process involves:

1. **Loading the Dataset**: The dataset is loaded into a Pandas DataFrame.
2. **Data Cleaning**: Handling missing values, if any, and removing or imputing them.
3. **Data Visualization**: Visualizing relationships between different features and the target variable (heart disease).
4. **Feature Engineering**: Creating new features or modifying existing ones to improve model performance.

You can view the EDA process in the following notebook: [Heart Disease EDA Notebook](EDA/Heart_Disease_EDA.ipynb).

## Machine Learning: Preprocessing, Training, and Evaluating

The machine learning process includes:

1. **Data Preprocessing**: Scaling features, encoding categorical variables, and splitting the data into training and testing sets.
2. **Model Selection**: A Logistic Regression model is chosen due to its simplicity and effectiveness for binary classification problems.
3. **Training**: The model is trained using the training dataset.
4. **Evaluation**: The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.

You can explore the training and evaluation process in the following notebook: [Training and Evaluation Notebook](training/Training_and_Evaluation.ipynb).

## The Web Application

The project includes a web-based interface built using Streamlit, allowing users to input their health data and receive a prediction about their likelihood of having heart disease.

### Running the Web Application

1. **Install Dependencies**: Ensure that all required Python packages are installed by running:

    ```bash
    pip install -r requirements.txt
    ```

2. **Navigate to the Streamlit Directory**:

    ```bash
    cd streamlit
    ```

3. **Run the Streamlit App**:

    ```bash
    streamlit run main.py
    ```

4. **Use the Application**: Open your web browser and go to the local URL provided by Streamlit. You can input your health information to receive a prediction.



## Future Work

While this project provides a basic framework for heart disease prediction, there are several areas for improvement:

- **Expand the Dataset**: Using a larger, more diverse dataset could improve the model's accuracy.
- **Model Optimization**: Experiment with different machine learning models and hyperparameters.
- **Feature Selection**: Investigate which features are most predictive of heart disease and refine the model accordingly.

## Conclusion

This project demonstrates how machine learning can be applied to health data to predict the likelihood of heart disease. It also shows how to deploy a machine learning model in a user-friendly web application using Streamlit.
