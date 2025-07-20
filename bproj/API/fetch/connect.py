#######################################
# @author Michael Kane
# @date 07/06/2025
# Securing connections to Binance
# Testnet
#######################################
from binance.client import Client
import json, os


# Retrieve project settings
def get_project_settings(
        settingsPath
        ):

    if os.path.exists( settingsPath ):
        f = open( settingsPath, "r" )
        project_settings = json.load( f )
        f.close()
        return project_settings
    
    else:
        return ImportError
    

# Connect to Binance TestNet client to interact
def client(
        settingsPath,
        TestOrMain
        ):
    
    project_settings = get_project_settings( settingsPath )

    client = Client( api_key=project_settings["TestKeys"]["Test_API_Key"], api_secret=project_settings["TestKeys"]["Test_Secret_Key"] )

    if TestOrMain == "test":
        client.API_URL = "https://testnet.binance.vision/api"
    elif TestOrMain == "main":
        client.API_URL = "https://api.binance.com/api"

    return client