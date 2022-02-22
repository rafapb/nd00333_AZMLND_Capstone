import requests
import json

scoring_uri = 'http://c556a200-73bc-4f20-95d5-c8eb2f2dff67.southcentralus.azurecontainer.io/score'
key = 'QGYhcQWRpUi8avjzHlYWkRLFf3jtNo2g'

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
            "tau1": 6.53052661542757,
            "tau2": 4.34969522452047,
            "tau3": 6.78178989860924,
            "tau4": 8.67313819998627,
            "p1": 3.49280712530602,
            "p2": -1.53219257925325,
            "p3": -1.39028526338323,
            "p4": -0.570329282669537,
            "g1": 0.073055556762299,
            "g2": 0.378760930415059,
            "g3": 0.50544104823246,
            "g4": 0.942630832928815
          }
        ]
      }
      
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())