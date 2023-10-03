# Objective:
# The aim of this file is to combine import fields from both leagues.csv and tournaments.csv into a single
# file data.csv
import csv

leagues = []
tournaments = []

# opening tournaments.csv file and reading data from it
with open("tournaments.csv", 'r') as tournaments_file:
    tournaments_csv = csv.reader(tournaments_file)
    for row in tournaments_csv:
        tournaments.append(row)

# opening leagues.csv file and reading data from it
with open("leagues.csv", 'r') as leagues_file:
    leagues_csv = csv.reader(leagues_file)
    for row in leagues_csv:
        leagues.append(row)

# extracting the data of priority column from leagues.csv into tournaments.csv
tournaments[0].append("priority")
num_rows_tournaments = len(tournaments)
num_rows_leagues = len(leagues)
for i in range(1, num_rows_tournaments):
    league_id = tournaments[i][0]
    for j in range(1, num_rows_leagues):
        if leagues[j][0] == league_id:
            tournaments[i].append(leagues[j][2])
            break
        if j == num_rows_leagues-1:
            tournaments[i].append(" ")

# writing the mixed data into a new data.csv file
"""
with open("data.csv", 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(tournaments)
"""
