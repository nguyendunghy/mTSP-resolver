import asyncio
import time
from argparse import ArgumentParser
from cachetools import TTLCache
from flask import Flask, request, jsonify

from neurons.call_api import load_config, call_apis, call_set_cache, call_set_cache_nx
from neurons.redis_utils import get,set,set_if_not_exist
from graphite.protocol import GraphProblem, GraphSynapse
from neurons.call_method import (beam_solver_solution, baseline_solution, nns_vali_solver_solution,
                                 hpn_solver_solution, scoring_solution, tsp_annealer_solver, new_solver_solution,
                                 simulated_annealing_solver, or_solver_solution)
import json

def parse():
    parser = ArgumentParser()
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on.')
    parser.add_argument('--method', type=str, default='', help='Resolver to generate route')
    return parser.parse_args()


args = parse()


import fastapi
from fastapi import status
from fastapi.responses import JSONResponse

app = fastapi.FastAPI()
# SOLVER_CACHE = TTLCache(maxsize=1800, ttl=900)
SOLVER_CACHE = {}

def run_resolver(method, synapse_request):
    if method == 'BEAM':
        return asyncio.run(beam_solver_solution(synapse_request))
    elif method == 'BASE_LINE':
        return asyncio.run(baseline_solution(synapse_request))
    elif method == 'NNS_VALI':
        return asyncio.run(nns_vali_solver_solution(synapse_request))
    elif method == 'HPN':
        return asyncio.run(hpn_solver_solution(synapse_request))
    elif method == 'LEAF_SA':
        return asyncio.run(tsp_annealer_solver(synapse_request))
    elif method == 'JACKIE_SA':
        return asyncio.run(simulated_annealing_solver(synapse_request))
    elif method == 'NEW':
        return asyncio.run(new_solver_solution(synapse_request))
    elif method == 'OR':
        return asyncio.run(or_solver_solution(synapse_request))
    else:
        print(f"method not in accept list")
        resolver = None

    return resolver


def set_cache_mem(key, value):
    SOLVER_CACHE[key] = value


def get_cache_mem(key):
    return SOLVER_CACHE.get(key, None)  # Returns None if the key has expired or does not exist


def set_cache_redis(key, route):
    try:
        cached_data = {'route': route}
        value = json.dumps(cached_data)
        print(f"save cache key = {key}, value = {value}")
        set(key, value)
    except Exception as e:
        print(f"save cache redis error {e}")


def get_cache_redis(key):
    try:
        value = get(key)
        if value is None or len(value) < 1:
            return None
        else:
            saved_data = json.loads(value)
            return saved_data['route']
    except Exception as e:
        print(f"get cache redis error {e}")
        return None


async def wait_get_cache_redis(hash, graph_problem, config):
    start_time = time.time_ns()
    count = 0
    max_count = config['num_count']
    time_sleep = config['time_sleep']
    while True:
        route = get_cache_redis(key=hash)
        print(f"route from redis {route}")
        if route is None:
            # wait for other miner set cache
            if count >= max_count:
                break
            count = count + 1
            print(f"wait for other miner set cache count = {count}")
            await asyncio.sleep(time_sleep)
            # time.sleep(time_sleep)
        else:
            set_cache_mem(hash,route)
            print(f"time wait_get_cache_redis {int(time.time_ns() - start_time):,} nanosecond")
            return route

    # call apis fail, use or-solver
    print(f"call cache redis fail, using or-resolver")
    synapse_request = GraphSynapse(problem=graph_problem)
    synapse = asyncio.run(or_solver_solution(synapse_request))
    return synapse.solution

@app.get("/")
def hello_world():
    return "Hello! I am resolver service"


@app.post('/resolve')
def register(data: dict):
    start_time = time.time_ns()
    if "problem" not in data:
        # return jsonify({"error": "Request must contain 'problem'"}), 400
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Request must contain 'problem'"})
    # data = request.get_json()
    problem = data['problem']
    graph_problem = GraphProblem.parse_obj(problem)
    synapse = run_resolver(args.method, GraphSynapse(problem=graph_problem))
    print(f"synapse = {synapse}")
    score = scoring_solution(synapse)
    print(f"score = {score}")
    print(f"time loading {int(time.time_ns() - start_time):,} nanosecond")
    # return jsonify({"message": "SUCCESS", "result":  synapse.solution, "score": score}), 200
    return {
        "message": "SUCCESS",
        "result": synapse.solution,
        "score": score
    }


@app.post('/server')
async def server(data: dict):
    start_time = time.time_ns()
    if "problem" not in data:
        # return jsonify({"error": "Request must contain 'problem'"}), 400
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Request must contain 'problem'"})
    # data = request.get_json()
    problem = data['problem']
    graph_problem = GraphProblem.parse_obj(problem)
    hash = data['hash']
    config_file_path = data['config_file_path']
    print(f'run server hash = {hash}, config_file_path = {config_file_path}')
    synapse_request = GraphSynapse(problem=graph_problem)
    config = load_config(config_file=config_file_path)

    # call memory cache
    mem_value = await get_cache_mem(hash)
    if mem_value is not None and len(mem_value) > 1:
        print(f'hit mem cache hash = {hash}')
        print(f"time loading {int(time.time_ns() - start_time):,} nanosecond")
        # return jsonify({"message": "Success", "result": mem_value}), 200
        return {
            "message": "Success",
            "result": mem_value
        }
    setnx = set_if_not_exist(hash, '')
    if setnx:
        route = call_apis(synapse_request, config)
        if route is not None:
            set_cache_mem(hash, route)
            set_cache_redis(hash, route)
            print(f"time loading {int(time.time_ns() - start_time):,} nanosecond")
            # return jsonify({"message": "Success", "result": route}), 200
            return {
                "message": "Success",
                "result": route 
            }
        else:
            # call apis fail, use baseline
            print(f"call cache fail, using or-solver setnx = {setnx}")
            synapse = await or_solver_solution(synapse_request)
            print(f"time loading {int(time.time_ns() - start_time):,} nanosecond")
            # return jsonify({"message": "Success", "result": synapse.solution}), 200
            return {
                "message": "Success",
                "result": synapse.solution
            }
    else:
        route = await wait_get_cache_redis(hash, synapse_request, config)
        print(f"time loading {int(time.time_ns() - start_time):,} nanosecond")
        # return jsonify({"message": "Success", "result": route}), 200
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Success", "result": route})
if __name__ == '__main__':
    # app.run(host='0.0.0.0', debug=False, port=args.port)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
