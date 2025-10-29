import streamlit as st
import pandas as pd
from apputil import *

# Add custom CSS for styling and animations
st.markdown("""
    <style>
        body {
            background: linear-gradient(45deg, #4CAF50, #FFC107);
            font-family: 'Arial', sans-serif;
        }
        .header {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: white;
            animation: fadeIn 2s ease-in-out;
        }
        .subheader {
            text-align: center;
            font-size: 28px;
            color: #FFC107;
            margin-top: 20px;
            animation: slideIn 1.5s ease-out;
        }
        .description {
            text-align: center;
            font-size: 18px;
            color: #757575;
            margin-top: 20px;
            animation: fadeIn 2s ease-in-out;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stSelectbox>div>label {
            font-size: 18px;
            font-weight: bold;
            color: white;
        }
        .stFileUploader>div>label {
            font-size: 18px;
            font-weight: bold;
            color: white;
        }
        .prediction-table {
            border-collapse: collapse;
            width: 100%;
            border: 1px solid #ddd;
            margin-top: 20px;
        }
        .prediction-table th, .prediction-table td {
            padding: 12px;
            text-align: left;
        }
        .prediction-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .prediction-table th {
            background-color: #4CAF50;
            color: white;
        }
        .prediction-table td {
            color: #4CAF50;
        }

        /* CSS Animations */
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }

        @keyframes slideIn {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(0); }
        }

        /* Button Hover Effect */
        .stButton>button {
            transition: transform 0.2s ease;
        }
        .stButton>button:hover {
            transform: scale(1.1);
        }

    </style>
    """, unsafe_allow_html=True)

# Main Heading with Animation
st.markdown("<div class='header'>Group Estimate Model</div>", unsafe_allow_html=True)

st.write(
    '''
    This app allows you to input data, choose an estimation method (mean or median), and perform predictions based on your input features.
    '''
)

# Title for the user input section with Slide-in animation
st.markdown("<div class='subheader'>Enter Data for Prediction</div>", unsafe_allow_html=True)

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
            X = data[[feature_column]]  # Pass X as DataFrame (with double square brackets)
            y = data[target_column]
            model.fit(X, y)

            # Prepare the feature data for predictions (as list of lists)
            X_for_pred = [[val] for val in X[feature_column]]  # Convert the column to list of lists

            # Make predictions
            predictions = model.predict(X_for_pred)  # Pass a list of lists for predictions

            # Display predictions
            st.write("Predictions based on group estimates:")

            # Display prediction results in a nicely styled table
            prediction_df = pd.DataFrame({
                feature_column: data[feature_column],  # Display the original feature column
                'Predicted Value': predictions
            })

            st.write('<table class="prediction-table">', unsafe_allow_html=True)
            st.write(prediction_df.to_html(index=False), unsafe_allow_html=True)
            st.write('</table>', unsafe_allow_html=True)

            st.success("Model trained and predictions made successfully!")

else:
    st.warning("Please upload a CSV file to proceed.")
