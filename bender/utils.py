import requests
import getpass
from jose import jwt
from datetime import datetime
import json
import os


def token_path():
    return os.path.join(os.path.expanduser('~'), '.bender_token')


def remove_saved_token():
    os.remove(token_path())


def load_and_check_token(url):

    with open(token_path(), 'r') as f:
        token = json.load(f)

    # Check payload
    payload = jwt.decode(
        token,
        key='',
        options={
            'verify_signature': False,
            'verify_exp': False})
    now = int(datetime.utcnow().timestamp())
    if now > (payload['exp'] - 7 * 24 * 3600):
        raise Exception("Token will expire")

    # Info
    request_info = requests.get("{}/user/".format(url),
                                headers={"Authorization": "JWT {}".format(token)})
    username = request_info.json()['username']
    pk = request_info.json()['pk']

    return token, username, pk


def save_token(token):
    """ Persists the credentials of the user to avoid asking the user again. """
    with open(token_path(), 'w') as f:
        json.dump(token, f)


def retrieve_token_and_username(url, cpt=0):
    """ Tries to load credentials from disk and prompt the user for it. """
    email = input('\nPlease enter your email\n')
    password = getpass.getpass()

    request_login = requests.post("{}/login/".format(url), json={
        "email": email,
        "password": password,
    })
    try:
        request_login.raise_for_status()
    except Exception:
        if cpt == 2:
            print("Wrong combination of username/password 3 times!")
        else:
            print("Wrong username/password try again!")
            retrieve_token_and_username(url=url, cpt=cpt + 1)

    token = request_login.json()["token"]
    save_token(token)

    request_info = requests.get("{}/user/".format(url),
                                headers={"Authorization": "JWT {}".format(token)})
    request_info.raise_for_status()
    username = request_info.json()['username']
    pk = request_info.json()['pk']

    return token, username, pk


def new_api_session(url):
    try:
        token, username, pk = load_and_check_token(url=url)
    except Exception:
        token, username, pk = retrieve_token_and_username(url=url)
    session = requests.Session()
    session.headers.update({'Authorization': 'JWT {}'.format(token)})
    return session, username, pk
