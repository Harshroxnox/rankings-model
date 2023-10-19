import pandas as pd

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
df.to_csv("data_final.csv", index=False)
"""

# Splitting the dataframe into training and testing dataframe
test_size = 0.1

# Calculate the number of rows for the testing set
test_set_size = int(len(df) * test_size)

# Split the DataFrame into training and testing while keeping order
train_df = df.iloc[:-test_set_size]
test_df = df.iloc[-test_set_size:]

# Exporting training and testing dataframes into a csv file
"""
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
"""
