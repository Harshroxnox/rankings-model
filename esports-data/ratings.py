import json
import csv

teams_file = open("teams.json", "r")
data = json.loads(teams_file.read())

teams = []
headers = ["teamId", "ratings"]
teams.append(headers)

for dictionary in data:
    row = []
    row.append(dictionary["team_id"])
    row.append(400)
    teams.append(row)

"""
with open("teams.csv", 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(teams)
"""
teams_file.close()
