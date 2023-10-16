import pandas as pd
import numpy as np
import tensorflow as tf
import random
import sys

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

ratings = [[-sys.maxsize-1, -sys.maxsize-1]]

# Define the ratings environment
# rating_a, rating_b, rating_diff, team_a_wins

# Define possible actions (different values of k)
ACTIONS = [
    random.randint(1, 10),
    random.randint(10, 20),
    random.randint(20, 30),
    random.randint(30, 40),
    random.randint(40, 50),
    random.randint(50, 60),
    random.randint(60, 70),
    random.randint(70, 80),
    random.randint(80, 90),
    random.randint(90, 100),
]
print(ACTIONS)
# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),  # State (x, y)
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(ACTIONS))  # Output Q-values for each action
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Hyperparameters
DISCOUNT_FACTOR = 0.9
EPISODES = 30
EPSILON = 0.4

count = 0
score = 0
# Deep Q-learning algorithm
for index, row in train_df.iterrows():
    if count == 700:
        EPSILON = 0.1
    count += 1

    print(count)
    print(" ")
    print(score)
    print(" ")
    print(ratings)

    rating_a = None
    rating_b = None
    index_a = None
    index_b = None
    rating_diff = None
    index_itr = 0
    for sublist in ratings:
        if sublist[0] == row['teamA']:
            rating_a = sublist[1]
            index_a = index_itr
            break
        elif ratings[-1] == sublist:
            new_team = [row['teamA'], 400]
            ratings.append(new_team)
            index_a = len(ratings)-1
            rating_a = 400
        index_itr += 1

    index_itr = 0
    for sublist in ratings:
        if sublist[0] == row['teamB']:
            rating_b = sublist[1]
            index_b = index_itr
            break
        elif ratings[-1] == sublist:
            new_team = [row['teamB'], 400]
            ratings.append(new_team)
            index_b = len(ratings) - 1
            rating_b = 400

    if rating_a > rating_b:
        rating_diff = rating_a - rating_b
    else:
        rating_diff = rating_b - rating_a

    team_a_wins = row["result"]
    state = [rating_a, rating_b, rating_diff, team_a_wins]

    # Epsilon-greedy exploration
    if np.random.rand() < EPSILON:
        action = np.random.choice(len(ACTIONS))
    else:
        q_values = model.predict(np.array([state]))
        action = np.argmax(q_values)
    
    # Update the ratings in ranking dictionary
    if team_a_wins == 1:
        ratings[index_a][1] += ACTIONS[action]
        ratings[index_b][1] -= ACTIONS[action]
    else:
        ratings[index_a][1] -= ACTIONS[action]
        ratings[index_b][1] += ACTIONS[action]

    try:
        # Look one row ahead
        next_index, next_row = next(df.iterrows())
        next_rating_a = None
        next_rating_b = None
        next_rating_diff = None
        for sublist in ratings:
            if sublist[0] == next_row['teamA']:
                next_rating_a = sublist[1]
                break
            elif ratings[-1] == sublist:
                next_rating_a = 400

        for sublist in ratings:
            if sublist[0] == next_row['teamB']:
                next_rating_b = sublist[1]
                break
            elif ratings[-1] == sublist:
                next_rating_b = 400

        if next_rating_a > next_rating_b:
            next_rating_diff = next_rating_a - next_rating_b
        else:
            next_rating_diff = next_rating_b - next_rating_a
        next_team_a_wins = next_row["result"]
        next_state = [next_rating_a, next_rating_b, next_rating_diff, next_team_a_wins]
    except StopIteration:
        print("No next row (End of DataFrame)")
        break

    # Calculate the reward (negative for wrong prediction, positive for right prediction)
    reward = 0
    if rating_a > rating_b:
        if team_a_wins == 1:
            reward = 1
        else:
            reward = -1
    elif rating_a < rating_b:
        if team_a_wins == 0:
            reward = 1
        else:
            reward = -1
    else:
        reward = 0
    score += reward
    # Q-value update using the neural network
    q_values = model.predict(np.array([state]))
    q_values_next = model.predict(np.array([next_state]))
    q_values[0][action] = reward + DISCOUNT_FACTOR * np.max(q_values_next)
    print(DISCOUNT_FACTOR * np.max(q_values_next))
    model.fit(np.array([state]), q_values, verbose=0)

"""
# Test the trained policy
state = START_STATE
path = [state]

while state != GOAL_STATE:
    q_values = model.predict(np.array([state]))
    action = np.argmax(q_values)
    next_state = (state[0] + ACTIONS[action][0], state[1] + ACTIONS[action][1])
    path.append(next_state)
    state = next_state

# Print the optimal path
print("Optimal Path:")
for step in path:
    print(step)
"""
