# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import OneClassSVM
from mlxtend.frequent_patterns import apriori, association_rules

# Function to perform a basic data mining operation (mean calculation)
def calculate_mean(data, target_column):
    # Visualization: Histogram
    st.subheader("Visualization: Histogram")
    plt.figure(figsize=(8, 6))
    sns.histplot(data[target_column], kde=True)
    plt.title(f'Distribution of {target_column}')
    plt.xlabel(target_column)
    plt.ylabel('Frequency')
    st.pyplot(plt)
    return data[target_column].mean()


# Function to perform clustering using K-Means
def perform_clustering(data,target_column, num_clusters):

    return data

# def perform_clustering(data,num_clusters):
    # Select only relevant columns for clustering
    
    # target_column = st.selectbox("Select Target Column for Clustering", df.columns)
    # Select only relevant columns for clustering
    # clustering_data = data[target_column]
    # Reshape the data to 2D array if it's 1D
    # if len(clustering_data.shape) == 1:
        # clustering_data = clustering_data.values.reshape(-1, 1)

        # kmeans = KMeans(n_clusters=num_clusters)
        # labels = kmeans.fit_predict(clustering_data)

        # # Optionally, you can add the cluster labels as a new column in the dataframe
        # data['Cluster'] = labels

    # return data

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
def perform_regression(data, target_column, independent_columns):
    X = data[independent_columns]  # Use selected independent columns
    y = data[target_column]

    reg = RandomForestRegressor()
    reg.fit(X, y)
    # Visualization: Scatter Plot
    st.subheader("Visualization: Scatter Plot")
    plt.figure(figsize=(8, 6))
    plt.scatter(data[independent_columns].iloc[:, 0], y, color='blue', label='Actual')
    plt.scatter(data[independent_columns].iloc[:, 0], reg.predict(X), color='red', label='Predicted')
    plt.title(f'Scatter Plot for {target_column} vs. {independent_columns[0]}')
    plt.xlabel(independent_columns[0])
    plt.ylabel(target_column)
    plt.legend()
    st.pyplot(plt)

    # Visualization: Box Plot
    st.subheader("Visualization: Box Plot")
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data[target_column], y=data[independent_columns[0]])
    plt.title(f'Box Plot for {target_column} vs. {independent_columns[0]}')
    plt.xlabel(target_column)
    plt.ylabel(independent_columns[0])
    st.pyplot(plt)

    # Visualization: Feature Importance
    st.subheader("Visualization: Feature Importance")
    feature_importance = reg.feature_importances_
    importance_df = pd.DataFrame({'Feature': independent_columns, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance for Regression')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    st.pyplot(plt)

    # Return predicted values
    return reg.predict(X)

# Function to perform outlier detection using OneClass SVM
def detect_outliers(data):
    ocsvm = OneClassSVM()
    labels = ocsvm.fit_predict(data)
    return labels

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
        if operation in ["Classification", "Regression", "Mean"]:
            target_column = st.selectbox("Select Target Column", df.columns)

        if operation in["Clustering"]:
            num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
            target_column1 = st.selectbox("Select Target Column", df.columns, key="target_column1")
            target_column2 = st.selectbox("Select Target Column", df.columns, key="target_column2")

        # User input for independent columns (for regression)
        independent_columns = []
        if operation == "Regression":
            independent_columns = st.multiselect("Select Independent Columns", df.columns)

        # Perform data mining operation
        if st.button("Perform Operation"):
            if operation == "Mean":
                result = calculate_mean(df, target_column)
            elif operation == "Clustering":
               
                result = perform_clustering(df,num_clusters)

                # Visualization: Scatter Plot with Clusters
                st.subheader("Visualization: Scatter Plot with Clusters")
                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=result, x=target_column1, y=target_column2, hue='Cluster', palette='viridis', legend='full')
                plt.title(f'Scatter Plot with {num_clusters} Clusters')
                st.pyplot(plt)

            elif operation == "Classification":
                result = perform_classification(df, target_column)
            elif operation == "Association Rule Mining":
                result = perform_association_rule_mining(df)
            elif operation == "Regression":
                result = perform_regression(df, target_column, independent_columns)
            elif operation == "Outlier Detection":
                result = detect_outliers(df)

            st.subheader("Result:")
            st.write(result)

if __name__ == "__main__":
    main()
