import requests
import csv

with open("princeton.csv") as fp:
    csv_reader = csv.DictReader(fp)
    with requests.Session() as session:
        for row in csv_reader:
            timestamp = row['Time (UTC)']
            for key in row.keys():
                if key[-1] == "W":
                    request = {}
                    request["timestamp"] = timestamp
                    request["user_id"] = key.split(".")[0]
                    request["power"] = row[key]
                    if not request["power"]:
                        request["power"] = 0.
                    r = session.post("http://ec2-3-18-9-170.us-east-2.compute.amazonaws.com/insert_consumption", json=request)
                    print(r.text)