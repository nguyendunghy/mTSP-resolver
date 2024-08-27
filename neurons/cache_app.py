import time
from argparse import ArgumentParser
import json
from flask import Flask, request, jsonify
from neurons.redis_utils import get,set

def parse():
    parser = ArgumentParser()
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on.')
    return parser.parse_args()


args = parse()
app = Flask(__name__)



@app.route("/")
def hi_world():
    return "Hello! I am resolver with cache server"


@app.route('/get-cache', methods=['POST'])
def get_cache():
    start_time = time.time_ns()
    if request.is_json:
        data = request.get_json()
        hash = data['hash']
        value = get(str(hash))
        if value is None:
            route = None
        else:
            saved_data = json.loads(value)
            route = saved_data['route']
        print(f"get cache key = {hash}, value = {route}")
        print(f"time loading {int(time.time_ns() - start_time):,} nanosecond")
        return jsonify({"message": "Get cache success", "result":  route}), 200
    else:
        return jsonify({"error": "Request must be JSON"}), 400


@app.route('/set-cache', methods=['POST'])
def set_cache():
    start_time = time.time_ns()
    if request.is_json:
        data = request.get_json()

        key = str(data['hash'])

        cached_data = {'route': data['route']}
        value = json.dumps(cached_data)
        print(f"save cache key = {key}, value = {value}")
        set(key, value)
        print(f"time loading {int(time.time_ns() - start_time):,} nanosecond")
        return jsonify({"message": "Set cache success", "hash": key, "route": data['route']}), 200
    else:
        return jsonify({"error": "Request must be JSON"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=args.port)
