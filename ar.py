import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def main():
    # Specify the path to the CSV file
    file_path = "DataSetA 1.csv"

    # Load dataset
    transactions = pd.read_csv(file_path, header=None)

    # Print the column indices of your transactions DataFrame
    print("Column Indices:", transactions.columns)

    # If the column indices are not integers starting from 0, reset them
    transactions.reset_index(drop=True, inplace=True)

    # Replace NaN values with an empty string (you can choose a different placeholder)
    transactions.fillna('', inplace=True)

    # Create a copy of the DataFrame for Association Rule mining
    df = transactions.copy()

    # Initialize TransactionEncoder
    te = TransactionEncoder()

    # Fit and transform the transactions
    te_ary = te.fit(df.values).transform(df.values)

    # Create a DataFrame with the transformed data
    df = pd.DataFrame(te_ary, columns=te.columns_)

    print("Original Transactions:")
    print(transactions)

    frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

    print("\nFrequent Itemsets:")
    print(frequent_itemsets)

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

    print("\nAssociation Rules:")
    print(rules)

if __name__ == "__main__":
    main()
