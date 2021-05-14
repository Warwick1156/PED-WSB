import json

from data import path


class KeyHandler:
    def __init__(self):
        with open(path('key')) as f:
            file = json.load(f)

        self.client_id = file['id']
        self.secret_key = file['key']
        self.login_data = {
            'grant_type': 'password',
            'username': file['user'],
            'password': file['password']
        }

        self.headers = {}