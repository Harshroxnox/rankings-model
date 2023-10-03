# Objective:
# The aim of this file is to read all the data from tournaments.json file then clean all the data which
# includes removing data which is not useful, un-nesting some objects, and changing the names of some
# properties of objects.
# After cleaning the json data we are going to flatten it and convert it into a list of lists where
# each list represents a row and every single list will have equal number of items
# Finally, we will further convert the flatten_json_data into a csv file which can be useful for machine
# learning
import json
import csv

# loading the tournaments.json file to read data from it
tournaments = open("tournaments.json", "r")
data = json.loads(tournaments.read())

# cleaning of data starts from here which includes deleting those items in json which I do not found to
# be useful and also changing the names of some properties of objects with the help of change_key_name
# function and also un-nesting some of the objects


def change_key_name(dictionary, key, new_key):
    value = dictionary[key]
    dictionary[new_key] = value
    del dictionary[key]


for tournament in data:
    change_key_name(tournament, "id", "tournamentId")
    del tournament["name"]
    del tournament["slug"]
    del tournament["sport"]


for tournament in data:
    for stage in tournament["stages"]:
        change_key_name(stage, "name", "stageName")
        del stage["slug"]
        del stage["type"]


for tournament in data:
    for stage in tournament["stages"]:
        for section in stage["sections"]:
            change_key_name(section, "name", "sectionName")
            del section["rankings"]


for tournament in data:
    for stage in tournament["stages"]:
        for section in stage["sections"]:
            for match in section["matches"]:
                del match["type"]
                del match["state"]
                del match["mode"]
                strategy = match["strategy"]
                match["count"] = strategy["count"]
                del match["strategy"]


for tournament in data:
    for stage in tournament["stages"]:
        for section in stage["sections"]:
            for match in section["matches"]:
                change_key_name(match, "id", "matchId")
                for team in match["teams"]:
                    change_key_name(team, "id", "teamId")
                    del team["side"]
                    del team["record"]
                    del team["result"]
                for game in match["games"]:
                    change_key_name(game, "id", "gameId")


for tournament in data:
    change_key_name(tournament, "stages", "Stages")
    for stage in tournament["Stages"]:
        change_key_name(stage, "sections", "Sections")
        for section in stage["Sections"]:
            change_key_name(section, "matches", "Matches")
            for match in section["Matches"]:
                change_key_name(match, "teams", "Teams")
                change_key_name(match, "games", "Games")
                for team_match in match["Teams"]:
                    change_key_name(team_match, "players", "Players")
                team_id = match["Teams"][0]["teamId"]
                for game in match["Games"]:
                    for team in game["teams"]:
                        if team["id"] == team_id:
                            result = team["result"]
                    game["result"] = result
                    del game["state"]
                    del game["teams"]


for tournament in data:
    for stage in tournament["Stages"]:
        for section in stage["Sections"]:
            for match in section["Matches"]:
                for game in match["Games"]:
                    if game["result"] is None:
                        score = None
                    else:
                        score = game["result"]["outcome"]
                    game["score"] = score
                    del game["result"]
                    change_key_name(game, "score", "result")


# now most of the things are cleaned up in our data I just want to output the json and see what's happening
"""
with open("tournaments_cleaned.json", 'w') as json_file:
    json.dump(data, json_file, indent=2)
"""

# headings of the columns that I want to extract the data from
flatten_json_data = []
data_row = []
headers = ["leagueId", "startDate", "endDate", "tournamentId", "stageName", "sectionName", "count",
           "matchId", "teamId", "teamId", "number", "gameId", "result"]
flatten_json_data.append(headers)

# this function goes through my data and inserts all the data of the above column headings row by row
# into the flatten_json_data which I am further going to convert to a csv file that will be useful
# for machine learning


def get_data_rows(list_of_dict):
    global data_row
    global flatten_json_data
    for dictionary in list_of_dict:
        temp = data_row.copy()
        for key in dictionary:
            if key in headers:
                data_row.append(dictionary[key])
            if isinstance(dictionary[key], list) and key != "Players" and key != "Teams":
                get_data_rows(dictionary[key])
            if key == "Teams":
                data_row.append(dictionary[key][0]["teamId"])
                data_row.append(dictionary[key][0]["teamId"])
        if len(data_row) == 13:
            flatten_json_data.append(data_row)
        data_row = temp.copy()


get_data_rows(data)
print(flatten_json_data)

# write all flatten_json_data into a csv file
"""
with open("tournaments.csv", 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(flatten_json_data)
"""
tournaments.close()
