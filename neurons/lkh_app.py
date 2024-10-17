import asyncio
import json
import time
from argparse import ArgumentParser


from graphite.protocol import GraphV1Problem, GraphV1Synapse,GraphV2Problem,GraphV2Synapse
from neurons.call_api import load_config, call_apis
from neurons.call_method import (beam_solver_solution, baseline_solution, nns_vali_solver_solution,
                                 hpn_solver_solution, scoring_solution, tsp_annealer_solver, new_solver_solution,
                                 simulated_annealing_solver, or_solver_solution, lkh_solver_solution)
from neurons.compare_solutions import recreate_edges
from neurons.redis_utils import get, set, set_if_not_exist


def parse():
    parser = ArgumentParser()
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on.')
    parser.add_argument('--method', type=str, default='', help='Resolver to generate route')
    return parser.parse_args()


args = parse()


import fastapi
from fastapi import status
from fastapi.responses import JSONResponse


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
    elif method == 'LKH':
        return asyncio.run(lkh_solver_solution(synapse_request))
    else:
        print(f"method not in accept list")
        resolver = None

    return resolver





@app.get("/")
def hello_world():
    return "Hello! I am resolver service"


@app.post('/lkh_resolve')
def register(data: dict):
    start_time = time.time_ns()
    if "input_file_path" not in data:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Request must contain 'input_file_path'"})
    problem = data['problem']
    dataset_ref = problem.get('dataset_ref')
    print(f'dataset_ref = {dataset_ref}')

    graph_problem = GraphV2Problem.parse_obj(problem)
    edges = recreate_edges(graph_problem)
    graph_problem.edges = edges
    graph_synapse = GraphV2Synapse(problem=graph_problem)

    synapse = run_resolver(args.method, graph_synapse)
    print(f"synapse = {synapse}")
    score = scoring_solution(synapse)
    print(f"score = {score}, {type(score)}")
    print(f"type of solution: {synapse.solution}, {type(synapse.solution)}, {type(synapse.solution[0])}")
    print(f"time loading {int(time.time_ns() - start_time):,} nanosecond")
    return {
        "message": "SUCCESS",
        "result": synapse.solution,
        "score": float(score)
    }



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
