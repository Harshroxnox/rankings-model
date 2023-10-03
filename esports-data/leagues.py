import json
import csv

leagues = open("leagues.json", "r")
flatten_json_data = []
# Now my data after going through the JSON parser is now lists of dictionaries
data = json.loads(leagues.read())

# here are the headings of columns that I want to have the data for
headers = ['id', 'region', 'priority']
flatten_json_data.append(headers)

# insert all the data for the headings in flatten_json_data
for league in data:
    list_data = []
    for key in league:
        if key in headers:
            list_data.append(league[key])
    flatten_json_data.append(list_data)

# write all flatten_json_data into a csv file
with open("leagues.csv", 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(flatten_json_data)


leagues.close()