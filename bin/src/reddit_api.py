import requests
import json
from os import path as osp

from security import KeyHandler
from data import path

VERSION = 'PedAPI/0.0.1'


class RedditClient:
    def __init__(self):
        self.kh = KeyHandler()
        self._login()
        self.save_path = path("comments")

    def _login(self):
        authorization = requests.auth.HTTPBasicAuth(
            self.kh.client_id,
            self.kh.secret_key)

        self.kh.headers['User-Agent'] = VERSION
        res = requests.post('https://www.reddit.com/api/v1/access_token',
                            auth=authorization,
                            data=self.kh.login_data,
                            headers=self.kh.headers)

        token = res.json()['access_token']
        self.kh.headers['Authorization'] = f'bearer {token}'

        assert self._test_authentication()
        print('Successfully logged as {}'.format(self.kh.login_data['username']))

    def _test_authentication(self):
        if requests.get('https://oautch.reddit.com/api/v1/me', headers=self.kh.headers).status_code == 200:
            return True
        return False

    def get_comments(self, post_id: str):
        link = 'https://oauth.reddit.com/r/wallstreetbets/comments/'
        res = requests.get(link + post_id, headers=self.kh.headers)

        with open(osp.join(self.save_path, post_id + '.json'), 'w') as outfile:
            json.dump(res.json(), outfile, indent=4)

