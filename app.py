import streamlit as st
import pandas as pd
from apputil import *

st.write(
    '''
    # Group Estimate Model

    This app allows you to input data, choose an estimation method (mean or median), and perform predictions based on your input features.
    '''
)

# Title for the user input section
st.header("Enter Data for Prediction")

# Input: Select estimation method (mean or median)
estimate_method = st.selectbox(
    "Choose estimation method:",
    ["mean", "median"]
)

# Input: Upload a CSV file containing features and target variable (X and y)
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV
    data = pd.read_csv(uploaded_file)

    # Display a preview of the uploaded file
    st.write("Preview of the uploaded data:")
    st.write(data.head())

    # Make sure the user selects the right columns (X, y)
    feature_column = st.selectbox("Select feature column for grouping:", data.columns)
    target_column = st.selectbox("Select target column:", data.columns)

    if st.button("Train the model and make predictions"):
        # Check if the selected columns are valid
        if feature_column == target_column:
            st.error("Feature column and target column cannot be the same.")
        else:
            # Initialize the GroupEstimate model with the selected estimate method
            model = GroupEstimate(estimate_method)

            # Fit the model using the selected feature and target columns
            X = data[feature_column]
            y = data[target_column]
            model.fit(X, y)

            # Make predictions
            predictions = model.predict(X)

            # Display predictions
            st.write("Predictions based on group estimates:")
            prediction_df = pd.DataFrame({
                feature_column: X,
                'Predicted Value': predictions
            })
            st.write(prediction_df)

            st.success("Model trained and predictions made successfully!")

else:
    st.warning("Please upload a CSV file to proceed.")
