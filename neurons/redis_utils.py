import hashlib
import traceback
import redis
import bittensor as bt

redis_pool = redis.ConnectionPool(host='127.0.0.1', port=6379, decode_responses=True)


def get_conn():
    conn = redis.Redis(connection_pool=redis_pool)
    return conn


def exists(key, db=0):
    try:
        conn = get_conn()
        conn.select(db)
        return conn.exists(key) == 1
    except Exception as e:
        bt.logging.error(e)
        traceback.print_exc()


def get(key, db=0):
    try:
        conn = get_conn()
        conn.select(db)
        return conn.get(key)
    except Exception as e:
        bt.logging.error(e)
        traceback.print_exc()


def set(key, value, db=0):
    try:
        conn = get_conn()
        conn.select(db)
        conn.setex(key, 3600, value)
    except Exception as e:
        bt.logging.error(e)
        traceback.print_exc()


def gen_hash(token):
    m = hashlib.sha256(token.encode('UTF-8'))
    sha256_hex = m.hexdigest()
    return sha256_hex


if __name__ == '__main__':
    print(gen_hash("abc"))
    # set('abc', '123')
    # val = get('abc')
    # print(val)
    # e = exists('abc')
    # print(e)
