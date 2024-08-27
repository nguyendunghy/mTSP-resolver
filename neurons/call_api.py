import asyncio
import json
import time

import aiohttp

from neurons.compare_solutions import generate_problem

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
    start = time.time_ns()
    json_data = synapse_request.problem.dict()
    payload = {
        'problem':json_data
    }
    print(f"synapse_request problem = {json_data}")
    min_result = asyncio.run(main(payload))

    print(f"min_result = {min_result}")

    end = time.time_ns()
    print(f"time processing call api: {(end-start)/1e6} ms")
    return min_result

if __name__ == '__main__':
    synapse_request = generate_problem()
    call_apis(synapse_request)