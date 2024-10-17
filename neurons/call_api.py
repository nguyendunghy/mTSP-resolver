import asyncio
import json
import time
import traceback

import aiohttp
import bittensor as bt
import requests

from neurons.call_method import baseline_solution, build_lkh_input_file, scoring_solution
from neurons.compare_solutions import generate_problem
from neurons.redis_utils import gen_hash



def load_config(config_file='config.json'):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

async def post_api(session, url, data):
    async with session.post(url, json=data) as response:
        result = await response.json()
        return result

async def post_api_timeout(session, url, data, timeout=10):
    try:
        start_time = time.time()
        async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            result = await response.json()
            end_time = time.time()
            print(f'time call api {url}: {end_time-start_time}')
            return result
    except Exception as e:
        bt.logging.error(e)
        return {"result": None}



async def call_server(synapse_request, config_file_path='config.json'):
    start_time = time.time_ns()
    try:
        problem_dict = synapse_request.problem.dict()
        json_problem = json.dumps(problem_dict)
        # print(f"call_server synapse_request problem = {problem_dict}")
        hash = gen_hash(json_problem)
        print(f"call_server hash = {hash}")

        config = load_config(config_file=config_file_path)
        server_api = config['server_api']
        timeout = config['server_timeout']
        print(f"server_api = {server_api}, timeout = {timeout}")

        payload = json.dumps({
            "problem": problem_dict,
            "hash": hash,
            "config_file_path": config_file_path
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", server_api, headers=headers, data=payload, timeout=timeout)
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
        print(f"time call_server: {(end_time - start_time) / 1e6} ms")



async def do_call(synapse_request,payload,config):
    api_urls = config['lkh_urls']
    api_timeout = config['lkh_api_timeout']

    async with aiohttp.ClientSession() as session:
        tasks = [post_api_timeout(session, url, payload,timeout=api_timeout) for url in api_urls]
        responses = await asyncio.gather(*tasks)

        response_list = []
        for idx, response in enumerate(responses):
            print(f"Response from API {idx + 1}: {api_urls[idx]}")
            synapse_request.solution = response['result']
            score = scoring_solution(synapse_request)
            response['score'] = score
            # print(response)
            response_list.append(response)

        min_score_dict = min(response_list, key=lambda x: x['score'])
        print(f'return route which has score = {min_score_dict['score']}')
        min_result = min_score_dict['result']
        return min_result

async def call_apis(synapse_request,config):
    try:
        start_time = time.time_ns()
        lkh_input_file = build_lkh_input_file(synapse_request,config['lkh_input_dir'])
        problem = synapse_request.problem
        payload = {
            "input_file_path": lkh_input_file,
            "n_nodes": problem.n_nodes,
            "dataset_ref": problem.dataset_ref,
            "timeout": config['lkh_timeout']
        }

        print(f"payload = {payload}")
        min_result = await do_call(synapse_request,payload,config)

        print(f"min_result = {min_result}")

        return min_result
    except Exception as e:
        bt.logging.error(e)
        traceback.print_exc()
        return None
    finally:
        end_time = time.time_ns()
        print(f"time processing call api: {(end_time - start_time)/1e6} ms")

def call_get_cache(synapse_request,config):
    start_time = time.time_ns()
    try:
        problem_dict = synapse_request.problem.dict()
        json_problem = json.dumps(problem_dict)
        hash = gen_hash(json_problem)
        print(f"call get cache hash = {hash}")
        get_cache_url = config['get_cache_url']
        timeout = config['get_cache_timeout']
        print(f"get_cache_url = {get_cache_url}, timeout = {timeout}")
        payload = json.dumps({
            "hash": hash
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", get_cache_url, headers=headers, data=payload, timeout=timeout)
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


def call_set_cache(synapse_request, route,config):
    start_time = time.time_ns()
    try:
        problem_dict = synapse_request.problem.dict()
        json_problem = json.dumps(problem_dict)
        hash = gen_hash(json_problem)
        print(f"set cache hash: {hash}")
        set_cache_url = config['set_cache_url']
        timeout = config['set_cache_timeout']
        print(f"set_cache_url = {set_cache_url}, timeout = {timeout}")
        payload = json.dumps({
            "hash": hash,
            "route":route
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", set_cache_url, headers=headers, data=payload, timeout=timeout)
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


def call_set_cache_nx(synapse_request,config):
    start_time = time.time_ns()
    try:
        problem_dict = synapse_request.problem.dict()
        json_problem = json.dumps(problem_dict)
        hash = gen_hash(json_problem)
        print(f"set cache nx hash: {hash}")
        set_cache_nx_url = config['set_cache_nx']
        timeout = config['set_cache_nx_timeout']
        print(f"set_cache_nx url = {set_cache_nx_url}, timeout = {timeout}")
        payload = json.dumps({
            "hash": hash
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", set_cache_nx_url, headers=headers, data=payload, timeout=timeout)
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

async def handle_request(synapse_request, config_file_path='config.json'):
    config = load_config(config_file=config_file_path)
    setnx = call_set_cache_nx(synapse_request,config)
    if setnx:
        route = call_apis(synapse_request,config)
        if route is not None:
            call_set_cache(synapse_request,route,config)
            return route

        # call apis fail, use baseline
        print(f"call cache fail, using baseline setnx = {setnx}")
        synapse = asyncio.run(baseline_solution(synapse_request))
        return synapse.solution
    else:
        count = 0
        max_count = config['num_count']
        time_sleep = config['time_sleep']
        while True:
            route = call_get_cache(synapse_request,config)
            if route is None:
                # wait for other miner set cache
                if count >= max_count:
                    break
                count = count + 1
                print(f"wait for other miner set cache count = {count}")
                await asyncio.sleep(time_sleep)
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
