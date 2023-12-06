# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Function to perform a basic data mining operation (mean calculation)
def calculate_mean(data, target_columns):
    # Display histogram for selected columns
    for col in target_columns:
        st.subheader(f"Histogram for Column '{col}':")
        fig, ax = plt.subplots()
        data[col].hist(bins=20, ax=ax)
        plt.title(f'Histogram: {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        st.pyplot(fig)

    # Calculate and return mean
    return data[target_columns].mean()

# Function to perform regression using Random Forest
def perform_regression(data, independent_variable, dependent_variable):
    X = data[[independent_variable]]
    y = data[dependent_variable]

    reg = RandomForestRegressor()
    reg.fit(X, y)

    # Return predicted values
    return reg.predict(X)

# Function to perform outlier detection using OneClassSVM
def detect_outliers(data):
    ocsvm = OneClassSVM()
    labels = ocsvm.fit_predict(data)
    return labels

# Function to handle missing values based on user choice for a specific column
def handle_missing_values(data, target_column, strategy, manual_value):
    cleaned_data = data.copy()  # Create a copy to retain changes
    
    if strategy == "Ignore":
        cleaned_data = cleaned_data.dropna(subset=[target_column])
    elif strategy == "Manual":
        # Fill missing values with user-inputted value
        cleaned_data[target_column] = cleaned_data[target_column].fillna(manual_value)
    elif strategy == "Mean":
        cleaned_data[target_column] = cleaned_data[target_column].fillna(cleaned_data[target_column].mean())
    elif strategy == "Most Probable":
        # You can implement a more sophisticated method to fill in missing values based on probabilities
        # For simplicity, let's fill missing values with the most common value in the column
        most_common_value = cleaned_data[target_column].mode().iloc[0]
        cleaned_data[target_column] = cleaned_data[target_column].fillna(most_common_value)

    return cleaned_data


# Modify the clustering function
def perform_clustering(column_data, num_clusters):
    # Reshape the data to a 2D array (required for KMeans)
    X = column_data.values.reshape(-1, 1)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

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

        # Check if dataset has missing values
        if df.isnull().any().any():
            # Detect columns with missing values
            columns_with_missing_values = df.columns[df.isnull().any()].tolist()

            st.subheader("Columns with Missing Values:")
            st.write(columns_with_missing_values)

            # Iterate through columns with missing values
            for col in columns_with_missing_values:
                st.subheader(f"Data Cleaning Options for Column '{col}':")
                missing_values_strategy = st.selectbox(f"How to Handle Missing Values in Column '{col}'?", ["Ignore", "Manual", "Mean", "Most Probable", "None"])

                if missing_values_strategy == "Manual":
                    # Fill missing values with user-inputted value
                    manual_value = st.text_input(f"Enter the value to fill in missing values in Column '{col}':", "")
                    df = handle_missing_values(df, target_column=col, strategy=missing_values_strategy, manual_value=manual_value)
                else:
                    df = handle_missing_values(df, target_column=col, strategy=missing_values_strategy, manual_value=None)

                # Display cleaned dataset for the specific column
                st.subheader(f"Cleaned Dataset for Column '{col}':")
                st.write(df)

            # Merge cleaned columns into the original dataset
            cleaned_df = df

            # Display final cleaned dataset
            st.subheader("Final Cleaned Dataset:")
            st.write(cleaned_df)

            # Option to drop and add columns
            if st.checkbox("Drop and Add Columns"):
                columns_to_drop = st.multiselect("Select Columns to Drop", cleaned_df.columns)
                cleaned_df = cleaned_df.drop(columns=columns_to_drop, errors='ignore')

                st.subheader("Updated Dataset:")
                st.write(cleaned_df)


            
        else:
            cleaned_df=df   


            # Data mining operation selection
            st.subheader("Data Visualization:")
            operation = st.selectbox("Select Data Visualizations:", ["Mean", "Regression", "Outlier Detection", "Pie Chart", "Bar Graph", "Box Plot"])

            # User input for target column
            if operation == "Mean":
                # Allow users to choose multiple columns for mean calculation
                target_columns = st.multiselect("Select Columns for Mean Calculation", cleaned_df.columns)

            # User input for regression variables
            if operation == "Regression":
                independent_variable = st.selectbox("Select Independent Variable", cleaned_df.columns)
                dependent_variable = st.selectbox("Select Dependent Variable", cleaned_df.columns)

            # User input for Pie Chart
            if operation == "Pie Chart":
                # Create Pie Chart for selected columns
                pie_chart_columns = st.multiselect("Select Columns for Pie Chart", cleaned_df.columns)

            # User input for Bar Graph
            if operation == "Bar Graph":
                # Create Bar Graph for selected columns
                x_axis_column = st.selectbox("Select X-axis Column for Bar Graph", cleaned_df.columns)
                y_axis_column = st.selectbox("Select Y-axis Column for Bar Graph", cleaned_df.columns)

            # User input for Box Plot
            if operation == "Box Plot":
                # Create Box Plot for selected columns
                box_plot_columns = st.multiselect("Select Columns for Box Plot", cleaned_df.columns)

            # Initialize result variable
            result = None

            # Perform data mining operation
            if st.button("Perform Operation"):
                if operation == "Mean":
                    result = calculate_mean(cleaned_df, target_columns)
                    # Display means in tabular form
                    st.subheader("Means for Selected Columns:")
                    st.write(result)
                elif operation == "Regression":
                    result = perform_regression(cleaned_df, independent_variable, dependent_variable)
                    # Display regression line graph
                    st.subheader("Linear Regression Line Graph:")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.regplot(x=cleaned_df[independent_variable], y=result, scatter_kws={'s': 10, 'alpha': 0.3}, ax=ax)
                    plt.title(f'Linear Regression: {independent_variable} vs. Predicted {dependent_variable}')
                    plt.xlabel(independent_variable)
                    plt.ylabel(f'Predicted {dependent_variable}')
                    st.pyplot(fig)
                elif operation == "Outlier Detection":
                    result = detect_outliers(cleaned_df)
                elif operation == "Pie Chart":
                    # Create Pie Chart for selected columns
                    for col in pie_chart_columns:
                        pie_chart_data = cleaned_df[col].value_counts()
                        st.subheader(f"Pie Chart for Column '{col}':")
                        fig, ax = plt.subplots()
                        ax.pie(pie_chart_data, labels=pie_chart_data.index, autopct='%1.1f%%', startangle=90)
                        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                        st.pyplot(fig)
                elif operation == "Bar Graph":
                    # Create Bar Graph for selected columns
                    bar_graph_data = cleaned_df.groupby(x_axis_column)[y_axis_column].count()
                    st.subheader(f"Bar Graph: {y_axis_column} vs. {x_axis_column}")
                    fig, ax = plt.subplots()
                    bar_graph_data.plot(kind='bar', ax=ax)
                    plt.title(f'Bar Graph: {y_axis_column} vs. {x_axis_column}')
                    plt.xlabel(x_axis_column)
                    plt.ylabel('Count')
                    st.pyplot(fig)
                elif operation == "Box Plot":
                    # Create Box Plot for selected columns
                    for col in box_plot_columns:
                        st.subheader(f"Box Plot for Column '{col}':")
                        fig, ax = plt.subplots()
                        sns.boxplot(x=cleaned_df[col], ax=ax)
                        plt.title(f'Box Plot: {col}')
                        st.pyplot(fig)

                if operation != "Mean" and operation != "Regression":
                    st.subheader("Result:")
                    st.write(result)

            # Add an option to download the cleaned dataset
            if st.button("Download Cleaned Dataset"):
                cleaned_df_csv = cleaned_df.to_csv(index=False)
                b64 = base64.b64encode(cleaned_df_csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_dataset.csv">Download Cleaned Dataset</a>'
                st.markdown(href, unsafe_allow_html=True)


            st.subheader("Data Mining:")
            operationM = st.selectbox("Select Data Visualizations:", ["Clustering", "Classification", "Association Rule"])
            if operationM=="Clustering":
                    target_cluster = st.selectbox("Select Dependent Variable", cleaned_df.columns)
                    num_clusters = st.slider("Select the number of clusters", min_value=2, max_value=10, value=3)
                   
            if st.button("Perform Operation", key="mining"):
                   if operationM == "Clustering":
                        # Get the column to perform clustering on
                        # Perform clustering on the selected column
                        result = perform_clustering(cleaned_df[target_cluster], num_clusters)
                        # Display clustering results, e.g., cluster labels
                        st.subheader("Clustering Results:")
                        st.write(result)
            # elif operationM == "Classification":


            # elif operationM =="Association Rule":



            # End the app
            # st.balloons()

if __name__ == "__main__":
    main()