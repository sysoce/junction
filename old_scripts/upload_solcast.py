import requests, json

with open("solcast.json") as fp:
    data = json.load(fp)

for entry in data["forecasts"]:
    request = {}
    request["user_id"] = "solcast"
    request["power"] = entry["pv_estimate"]
    timestamp = entry['period_end'][0:19] + "Z"
    request["timestamp"] = timestamp
    r = requests.post("http://ec2-3-18-9-170.us-east-2.compute.amazonaws.com/insert_production_forecast", json=request)
    print(r.text)