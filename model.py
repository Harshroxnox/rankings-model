import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import csv

train_df = pd.read_csv('train.csv')
ratings = [[-sys.maxsize-1, -sys.maxsize-1]]

# Define the ratings environment
# rating_a, rating_b, rating_diff, team_a_wins
# Define possible actions (different values of k)
ACTIONS = [
    5,
    15,
    25,
    35,
    45,
    55,
    65,
    75,
    85,
    95,
]

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
EPSILON = 0.4

# Parameters to track model training
count = 0
score = 0


def get_rating_and_index(team_id, ratings_arr):
    for index_teams, teams in enumerate(ratings_arr):
        if teams[0] == team_id:
            return teams[1], index_teams
    else:
        # Team not found, add a new entry
        new_team = [team_id, 400]
        ratings.append(new_team)
        return 400, len(ratings) - 1


# Initialize a variable to store the previous row
prev_row = None

# Deep Q-learning algorithm
for row in train_df.itertuples():
    # Reducing the value of EPSILON to avoid exploring in later stages
    if count == 700:
        EPSILON = 0.1
    count += 1

    # prev_row will actually act as our current state
    if prev_row is not None:
        # Print the parameters for tracking the training of model
        print(ratings)
        print(" ")
        print(count)
        print(" ")
        print(score)

        # Extract the variables of our current row that define our current state which we will pass
        # to the neural network to predict what action to take.
        rating_diff = None
        rating_a, index_a = get_rating_and_index(prev_row.teamA, ratings)
        rating_b, index_b = get_rating_and_index(prev_row.teamB, ratings)

        if rating_a > rating_b:
            rating_diff = rating_a - rating_b
        else:
            rating_diff = rating_b - rating_a

        team_a_wins = prev_row.result
        state = [rating_a, rating_b, rating_diff, team_a_wins]

        # Epsilon-greedy exploration if EPSILON is small it will do less and less exploration
        if np.random.rand() < EPSILON:
            action = np.random.choice(len(ACTIONS))
        else:
            q_values = model.predict(np.array([state]))
            action = np.argmax(q_values)

        # Update the ratings of teamA and teamB in ratings array
        if prev_row.result == 1:
            ratings[index_a][1] += ACTIONS[action]
            ratings[index_b][1] -= ACTIONS[action]
        else:
            ratings[index_a][1] -= ACTIONS[action]
            ratings[index_b][1] += ACTIONS[action]

        # Looking one step into the future i.e. the next step or the next row or the next sample and
        # extracting all the variables that define that next state we would be needing this to calculate
        # q values for next state which we will further use to update the q value for that particular
        # action we have taken.
        # Check if there is a next row if there is then retrieve the next state values
        # if there isn't just break out of the loop

        # This is the next state in reinforcement learning which is actually the current row in our loop
        next_rating_a = None
        next_rating_b = None
        next_rating_diff = None
        for sublist in ratings:
            if sublist[0] == row.teamA:
                next_rating_a = sublist[1]
                break
            elif ratings[-1] == sublist:
                next_rating_a = 400

        for sublist in ratings:
            if sublist[0] == row.teamB:
                next_rating_b = sublist[1]
                break
            elif ratings[-1] == sublist:
                next_rating_b = 400

        if next_rating_a > next_rating_b:
            next_rating_diff = next_rating_a - next_rating_b
        else:
            next_rating_diff = next_rating_b - next_rating_a
        next_team_a_wins = row.result
        next_state = [next_rating_a, next_rating_b, next_rating_diff, next_team_a_wins]

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
        model.fit(np.array([state]), q_values, verbose=0)
    prev_row = row

# Exporting the model
tf.saved_model.save(model, "model1")

# Exporting the ratings of all teams in the end
with open("ratings.csv", 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(ratings)

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
