import os
import json
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ['https://www.googleapis.com/auth/gmail.send']

TOKEN_PATH = 'token.pickle'
CREDENTIALS_PATH = 'gmail_credential.json'

def generate_token():
    creds = None
    if os.path.exists(TOKEN_PATH):
        try:
            with open(TOKEN_PATH, 'rb') as token:
                creds = pickle.load(token)
        except (EOFError, pickle.PickleError) as e:
            print(f"Error loading {TOKEN_PATH}: {e}. Re-authenticating...")
            creds = None
            if os.path.exists(TOKEN_PATH):
                os.remove(TOKEN_PATH)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_PATH):
                print(f"Error: Credentials file '{CREDENTIALS_PATH}' not found. Please ensure you have downloaded the JSON credentials from Google Cloud Console and named it '{CREDENTIALS_PATH}' in the project root.")
                return

            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_PATH, SCOPES
            )
            creds = flow.run_local_server(port=0)
        
        with open(TOKEN_PATH, 'wb') as token:
            pickle.dump(creds, token)
    
    print(f"Gmail API token successfully generated and stored to: {TOKEN_PATH}")
    print("您現在可以使用 Flask 應用程式中的郵件發送功能了。")
    return creds

if __name__ == '__main__':
    generate_token()