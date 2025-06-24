#######################################
# @author Michael Kane
# @date 07/06/2025
# Main script of binance testnet project,
# used to centrally integrate several
# separate components
#######################################
import fetch.connect as connect
import fetch.interact as interact
import pandas as pd

settingsPath = "config/settings.json"

test_client = connect.client(settingsPath, "test") # For interacting with test account
main_client = connect.client(settingsPath, "main") # For collecting more kline data
test_account = test_client.get_account()

df_options = {
    "client": main_client,
    "pair": "ETHUSDT",
    "kline_period": "1h", 
    "timeframe": "90 days ago UTC",
    "future_window": 3,
    "threshold": 0.01
}

df = interact.retrieve_dataframe(**df_options)