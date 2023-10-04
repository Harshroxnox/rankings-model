import pandas as pd
import tensorflow_decision_forests as tfdf

# reading the data from the csv file and converting it to pandas dataframe
df = pd.read_csv("data.csv")

# dropping the rows for which the result label is empty or is equal to forfeit as it is of no use
df = df.dropna(subset=["result"])
df = df[df['result'] != 'forfeit']

# replacing the values in result with 1 or 0 in case of win or loss
# Note: the win in result indicates that teamA won and vice versa
df["result"] = df["result"].replace({"win": 1, "loss": 0})

# removing hyphens from the date columns
df['startDate'] = df['startDate'].str.replace('-', '')
df['endDate'] = df['endDate'].str.replace('-', '')

# sorting the data by dates
df = df.sort_values(by='startDate')

# filling empty values of priority with 500 note priority ranges from 1 to 1000
# 1 means that the league is of highest priority
df["priority"] = df["priority"].replace(" ", 500)

# checking for null values
null_count_per_column = df.isnull().sum()
print(null_count_per_column)

# Making sure every column contains consistent data types
df['startDate'] = df['startDate'].astype(int)
df['endDate'] = df['endDate'].astype(int)
df['priority'] = df['priority'].astype(int)
df['result'] = df['result'].astype(int)

df['stageName'] = df['stageName'].str.replace('_', ' ')
df['stageName'] = df['stageName'].str.replace('1', 'A')
df['stageName'] = df['stageName'].str.replace('2', 'B')
df['stageName'] = df['stageName'].astype(str)

df['sectionName'] = df['sectionName'].str.replace('-', ' ')
df['sectionName'] = df['sectionName'].str.replace('_', ' ')
df['sectionName'] = df['sectionName'].str.replace('1', 'A')
df['sectionName'] = df['sectionName'].str.replace('2', 'B')
df['sectionName'] = df['sectionName'].str.replace('5', 'E')
df['sectionName'] = df['sectionName'].astype(str)

print("\n")
print(df.dtypes)

# exporting the dataframe to a csv file to just make sure and see that everything has been done correctly
"""
df.to_csv("pd_df.csv", index=False)
"""

# Splitting the dataframe into training and testing dataframe
train_df = df.sample(frac=0.9, random_state=0)
test_df = df.drop(train_df.index)

print("\n")
print("training df:\n")
print(train_df)

print("\n")
print("testing df:\n")
print(test_df)

# Convert the dataframe into a TensorFlow dataset.
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="result")
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="result")
