import requests
import json
from os import path as osp
from datetime import datetime, timedelta
import praw

from security import KeyHandler
from data import path

VERSION = 'PedAPI/0.0.3'


class RedditClient:
    def __init__(self, verbose: bool = True):
        self.verbose_print = print if verbose else lambda *a, **k: None
        self.save_path = path("comments")

        self.session_time_out = datetime.now()
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
        self.session_time_out = datetime.now() + timedelta(seconds=res.json()['expires_in'] - 60)

        token = res.json()['access_token']
        self.kh.headers['Authorization'] = f'bearer {token}'

        assert self._test_authentication()
        self.verbose_print('Successfully logged as {}'.format(self.kh.login_data['username']))
        self.verbose_print('Session expires on {}'.format(str(self.session_time_out)))

    def _test_authentication(self):
        if requests.get('https://oautch.reddit.com/api/v1/me', headers=self.kh.headers).status_code == 200:
            return True
        return False

    def _manage_session_time_out(self):
        if datetime.now() > self.session_time_out:
            self.verbose_print('Renewing session...')
            self._login()

    def get_comments(self, post_id: str):
        self._manage_session_time_out()

        link = 'https://oauth.reddit.com/r/wallstreetbets/comments/'
        res = requests.get(link + post_id, headers=self.kh.headers)

        with open(osp.join(self.save_path, post_id + '.json'), 'w') as outfile:
            json.dump(res.json(), outfile, indent=4)


class Praw:
    def __init__(self, verbose: bool = True):
        self.verbose_print = print if verbose else lambda *a, **k: None

        self.kh = KeyHandler()
        self._login()

    def _login(self):
        self.reddit = praw.Reddit(
            client_id=self.kh.client_id,
            client_secret=self.kh.secret_key,
            user_agent=VERSION,
            username=self.kh.login_data['username'],
            password=self.kh.login_data['password']
        )

    def get_submission(self, id_):
        return self.reddit.submission(id=id_)
