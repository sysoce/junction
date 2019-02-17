import requests, json

with open("solcast.json") as fp:
    data = json.load(fp)

with requests.Session() as session:
    for entry in data["forecasts"]:
        request = {}
        request["user_id"] = "solcast"
        request["power"] = entry["pv_estimate"]
        timestamp = entry['period_end'][0:19] + "Z"
        request["timestamp"] = timestamp
        try:
            r = session.post("http://ec2-3-18-9-170.us-east-2.compute.amazonaws.com/insert_production_forecast", json=request)
        except:
            print("Error with timestamp {}".format(timestamp))
            continue
