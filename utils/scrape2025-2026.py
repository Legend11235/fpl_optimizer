# Once I am done training my model I will make a script to scrape data after every game week and use the data to predict player points for suture gameweeks
#This is just a basic prototype that just extracts player prices of the 2025/26 season
import requests
import pandas as pd

url = "https://fantasy.premierleague.com/api/bootstrap-static/"
res = requests.get(url).json()

# Get player data
players = pd.DataFrame(res['elements'])

# Save name and price to CSV
players[['first_name', 'second_name', 'now_cost']].to_csv('data/2025-26_initial_prices.csv', index=False)