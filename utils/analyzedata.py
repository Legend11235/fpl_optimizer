#we are analysing all the fields in acrape2025-2026.py
import requests
import pandas as pd

url = "https://fantasy.premierleague.com/api/bootstrap-static/"
res = requests.get(url).json()

players = pd.DataFrame(res['elements'])
print(players.columns.tolist())