# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import OneClassSVM
from mlxtend.frequent_patterns import apriori, association_rules

# Function to perform a basic data mining operation (mean calculation)
def calculate_mean(data):
    return data.mean()

# Function to perform clustering using K-Means
def perform_clustering(data):
    kmeans = KMeans(n_clusters=2)
    labels = kmeans.fit_predict(data)
    return labels

# Function to perform classification using Random Forest
def perform_classification(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    clf = RandomForestClassifier()
    clf.fit(X, y)

    # Return predicted labels or probabilities
    return clf.predict(X)


# Function to perform association rule mining
def perform_association_rule_mining(data):
    # Assuming data is binary (0 or 1)
    frequent_itemsets = apriori(data, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    return rules

# Function to perform regression using Random Forest
def perform_regression(data):
    X = data.drop('target_column', axis=1)  # Replace 'target_column' with your actual target column
    y = data['target_column']

    reg = RandomForestRegressor()
    reg.fit(X, y)

    # Return predicted values
    return reg.predict(X)

# Function to perform outlier detection using One-Class SVM
def detect_outliers(data):
    ocsvm = OneClassSVM()
    labels = ocsvm.fit_predict(data)
    return labels

# Streamlit App
# Streamlit App
def main():
    st.title("Data Warehouse and Mining App")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Read the dataset
        df = pd.read_csv(uploaded_file)

        # Display uploaded dataset
        st.subheader("Uploaded Dataset:")
        st.write(df)

        # Data mining operation selection
        operation = st.selectbox("Select Data Mining Operation", ["Mean", "Clustering", "Classification", "Association Rule Mining", "Regression", "Outlier Detection"])

        # User input for target column
        if operation in ["Classification", "Regression"]:
            target_column = st.selectbox("Select Target Column", df.columns)

        # Perform data mining operation
        if st.button("Perform Operation"):
            if operation == "Mean":
                result = calculate_mean(df)
            elif operation == "Clustering":
                result = perform_clustering(df)
            elif operation == "Classification":
                result = perform_classification(df, target_column)
            elif operation == "Association Rule Mining":
                result = perform_association_rule_mining(df)
            elif operation == "Regression":
                result = perform_regression(df, target_column)
            elif operation == "Outlier Detection":
                result = detect_outliers(df)

            st.subheader("Result:")
            st.write(result)

if __name__ == "__main__":
    main()
