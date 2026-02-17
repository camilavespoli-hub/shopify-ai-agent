import os
import time
import requests

# 1. SECURITY LAYER: Loading Environment Variables
# [START client-credentials.config]
def load_dotenv(path=".env"):
    """
    Reads the .env file to load SHOPIFY_SHOP, CLIENT_ID, and CLIENT_SECRET.
    This keeps your credentials safe and out of the public code.
    """
    if not os.path.exists(path): #os.path.exists to know that the file .env exists. If it doesn't, it stops the function immediately
        return
    with open(path, "r", encoding="utf-8") as file: #Open the file in read mode, best practice
        for line in file: #reads the file one line on a time
            line = line.strip() #.strip() removes invisible "trash" like spaces or enter
            if not line or line.startswith("#") or "=" not in line: #ignore empty lines or commentor doesn't contain a =
                continue
            key, value = line.split("=", 1) #it cut the line at =, because key = value
            os.environ.setdefault(key, value.strip().strip('"').strip("'")) #saves the data into os.environ, it also remove extra ' or ""


load_dotenv()

SHOP = os.getenv("SHOPIFY_SHOP")
CLIENT_ID = os.getenv("SHOPIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SHOPIFY_CLIENT_SECRET")

if not SHOP or not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError("Set SHOPIFY_SHOP, SHOPIFY_CLIENT_ID, and SHOPIFY_CLIENT_SECRET.")
# [END client-credentials.config]

# [START client-credentials.get-token]
token = None
token_expires_at = 0.0


def get_token():
    global token, token_expires_at
    if token and time.time() < token_expires_at - 60: #It checks if we already have a token and if it still valid, -60 safety margin to ensure the token doesn't expire
        return token

    response = requests.post( #If we don't have a token, go to shopify for a new one
        f"https://{SHOP}.myshopify.com/admin/oauth/access_token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        timeout=30,
    )
    response.raise_for_status() #It crashes the program if the request failed
    data = response.json()
    token = data["access_token"] #Save the token
    token_expires_at = time.time() + data["expires_in"] #what time it expires
    return token
# [END client-credentials.get-token]


# [START client-credentials.query-products]
def graphql(query): #It takes our specific requests and delivers them to Shopify's servers. When my Agent needs read or write something, it will go through graphql
    response = requests.post( #Send information a specific web address
        f"https://{SHOP}.myshopify.com/admin/api/2025-01/graphql.json", #2025 version
        headers={
            "Content-Type": "application/json", #sending a JSON object
            "X-Shopify-Access-Token": get_token(), #CRUCIAL: Calls my token function to send the message, without it, it's impossible to communicate with Shopify
        },
        json={"query": query}, #The message - what I want to do. We wrap it in a JSON dictionary because that's the format the API expects
        timeout=30, #wait 30s for the response
    )
    response.raise_for_status() #Checkpoint: Stops the program if the server returns a 404 (no found) or 500 (server error)
    payload = response.json() #Unpacker: Converts the raw response from Shopify into a Python dictionary
    if payload.get("errors"):
        raise RuntimeError(payload["errors"]) #catch the error so we don't process the "garbage" data
    return payload["data"] #Returns the information we asked for, ignoring the metadata
# [END client-credentials.query-products]


def main() -> None: #Use none to indicate that the function performs actions but doesn't return a specific value
    query = "{ products(first: 3) { edges { node { id title handle } } } }" #Specific instruction
    data = graphql(query) #sent the query to graphql
    print(data) #Print the result



if __name__ == "__main__": #Standard Python protecton - Only run the main() function if I'm running this file directly
    main()
    