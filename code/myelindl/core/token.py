import jwt
import datetime
import uuid

DEFAULT_EXPIRE_MIN = 24 * 60 # one day
DEFAULT_ENCODE_ALGO = 'HS256'
SECRET_KEY = 'myelindl XDIMF#(VNBV#30239324325'


def gen_token(data, expire_min=DEFAULT_EXPIRE_MIN):
    if not isinstance(data, dict):
        raise ValueError('input should be dict')
    data['exp'] = datetime.datetime.utcnow() + datetime.timedelta(minutes=expire_min)
    return jwt.encode(data, SECRET_KEY, algorithm=DEFAULT_ENCODE_ALGO)


def decode_token(encoded_token):
    return jwt.decode(encoded_token, SECRET_KEY, algorithm=DEFAULT_ENCODE_ALGO)

def check_token(username, token):
    try:
        data = decode_token(token)
        if username != data['username']:
            return False
        return True
    except jwt.ExpiredSignatureError:
        return False
    except jwt.InvalidIssuerError:
        return False
