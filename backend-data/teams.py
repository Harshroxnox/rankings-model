# This file basically combines all the useful data of ratings.csv file and the teams.json
# file into a single team_ratings.csv file that is going to be useful for our backend server
# This team_ratings.csv file will further get converted into a mongodb database with which
# our server will be able to communicate
import json
import csv

with open("teams.json", "r") as json_file:
    data_teams = json.load(json_file)

print(len(data_teams))
data_ratings = []
with open("ratings.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        data_ratings.append(row)

data = []
headers = ["team_id", "rating", "name", "acronym"]
data.append(headers)

for team in data_ratings:
    data_row = []
    team_id = int(team[0])
    data_row.append(str(team[0]))
    data_row.append(int(team[1]))
    for dictionary in data_teams:
        if int(dictionary["team_id"]) == team_id:
            data_row.append(dictionary["name"])
            data_row.append(dictionary["acronym"])
            data.append(data_row)
            break

print(data)
# Combined all the useful data into a single data variable now exporting it to a csv file
"""
with open("team_ratings.csv", 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(data)
"""