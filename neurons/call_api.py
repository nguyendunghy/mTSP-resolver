import asyncio
import json
import time
import traceback

import aiohttp
import bittensor as bt
import requests

from neurons.call_method import baseline_solution
from neurons.compare_solutions import generate_problem
from neurons.redis_utils import gen_hash


# List of POST APIs
# api_urls = [
#     "http://127.0.0.1:8080/resolve",
#     "http://127.0.0.1:8081/resolve",
#     "http://127.0.0.1:8082/resolve",
#     "http://127.0.0.1:8083/resolve"
# ]
def load_config(config_file='config.json'):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

async def post_api(session, url, data):
    async with session.post(url, json=data) as response:
        result = await response.json()
        return result

async def main(payload):
    config = load_config()
    api_urls = config['api_urls']
    print(f"api_url = {api_urls}")

    async with aiohttp.ClientSession() as session:
        tasks = [post_api(session, url, payload) for url in api_urls]
        responses = await asyncio.gather(*tasks)

        response_list = []
        for idx, response in enumerate(responses):
            print(f"Response from API {idx + 1}:")
            print(response)
            response_list.append(response)

        min_score_dict = min(response_list, key=lambda x: x['score'])
        min_result = min_score_dict['result']
        return min_result

def call_apis(synapse_request):
    try:
        start_time = time.time_ns()
        json_data = synapse_request.problem.dict()
        payload = {
            'problem':json_data
        }
        print(f"synapse_request problem = {json_data}")
        min_result = asyncio.run(main(payload))

        print(f"min_result = {min_result}")

        return min_result
    except Exception as e:
        bt.logging.error(e)
        traceback.print_exc()
        return None
    finally:
        end_time = time.time_ns()
        print(f"time processing call api: {(end_time - start_time)/1e6} ms")

def call_get_cache(synapse_request):
    start_time = time.time_ns()
    try:
        problem_dict = synapse_request.problem.dict()
        json_problem = json.dumps(problem_dict)
        hash = gen_hash(json_problem)
        print(f"call get cache hash = {hash}")
        config = load_config()
        get_cache_url = config['get_cache_url']
        print(f"get_cache_url = {get_cache_url}")
        payload = json.dumps({
            "hash": hash
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", get_cache_url, headers=headers, data=payload, timeout=8)
        if response.status_code == 200:
            data = response.json()
            result = data['result']
            return result
        else:
            print('Failed to post data:status_code', response.status_code)
            print('Failed to post data:', response.content)
            return None
    except Exception as e:
        bt.logging.error(e)
        traceback.print_exc()
        return None
    finally:
        end_time = time.time_ns()
        print(f"time call get cache: {(end_time - start_time)/1e6} ms")


def call_set_cache(synapse_request, route):
    start_time = time.time_ns()
    try:
        problem_dict = synapse_request.problem.dict()
        json_problem = json.dumps(problem_dict)
        hash = gen_hash(json_problem)
        print(f"set cache hash: {hash}")
        config = load_config()
        set_cache_url = config['set_cache_url']
        print(f"set_cache_url = {set_cache_url}")
        payload = json.dumps({
            "hash": hash,
            "route":route
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", set_cache_url, headers=headers, data=payload, timeout=8)
        if response.status_code == 200:
            data = response.json()
            print(f"set cache finish, message: {data}")
            return data
        else:
            print('Failed to post data:status_code', response.status_code)
            print('Failed to post data:', response.content)
            return None
    except Exception as e:
        bt.logging.error(e)
        traceback.print_exc()
        return None
    finally:
        end_time = time.time_ns()
        print(f"time call set cache: {(end_time - start_time)/1e6} ms")


def call_set_cache_nx(synapse_request):
    start_time = time.time_ns()
    try:
        problem_dict = synapse_request.problem.dict()
        json_problem = json.dumps(problem_dict)
        hash = gen_hash(json_problem)
        print(f"set cache nx hash: {hash}")
        config = load_config()
        set_cache_nx_url = config['set_cache_nx']
        print(f"set_cache_nx url = {set_cache_nx_url}")
        payload = json.dumps({
            "hash": hash
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", set_cache_nx_url, headers=headers, data=payload, timeout=8)
        if response.status_code == 200:
            data = response.json()
            print(f"set cache nx finish, message: {data}")
            return data['result']
        else:
            print('Failed to post data:status_code', response.status_code)
            print('Failed to post data:', response.content)
            return None
    except Exception as e:
        bt.logging.error(e)
        traceback.print_exc()
        return None
    finally:
        end_time = time.time_ns()
        print(f"time call set cache nx: {(end_time - start_time)/1e6} ms")

async def handle_request(synapse_request):
    setnx = call_set_cache_nx(synapse_request)
    if setnx:
        route = call_apis(synapse_request)
        if route is not None:
            call_set_cache(synapse_request,route)
            return route

        # call apis fail, use baseline
        print(f"call cache fail, using baseline setnx = {setnx}")
        synapse = asyncio.run(baseline_solution(synapse_request))
        return synapse.solution
    else:
        count = 0
        while True:
            route = call_get_cache(synapse_request)
            if route is None:
                # wait for other miner set cache
                if count >= 30:
                    break
                count = count + 1
                print(f"wait for other miner set cache count = {count}")
                await asyncio.sleep(0.2)
            else:
                return route

        # call apis fail, use baseline
        print(f"call cache fail, using baseline setnx = {setnx}")
        synapse = asyncio.run(baseline_solution(synapse_request))
        return synapse.solution

if __name__ == '__main__':
    synapse_request = generate_problem()
    # call_apis(synapse_request)
    print(handle_request(synapse_request))
