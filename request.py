import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Accidents risk':4.5, 'Day risk scale':4, 'Time risk scale':3,'Weather risk scale':4,'Age risk scale':4,'Vehicle Age risk scale':4,'OVERLOAD SCALE':1,'VEHICLE HEALTH SCALE':5})

print(r.json())
