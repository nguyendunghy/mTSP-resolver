import asyncio
import json
import random
import time

import bittensor as bt
from pydantic import ValidationError

from graphite.dataset.dataset_generator import MetricTSPGenerator, GeneralTSPGenerator
from graphite.protocol import GraphSynapse, GraphProblem
from neurons.call_method import beam_solver_solution, baseline_solution, nns_vali_solver_solution, hpn_solver_solution, \
    scoring_solution, new_solver_solution, tsp_annealer_solver,simulated_annealing_solver


def generate_problem():
    prob_select = random.randint(1, 2)

    try:
        if prob_select == 1:
            problems, sizes = MetricTSPGenerator.generate_n_samples(1)
            test_problem_obj = problems[0]
        else:
            problems, sizes = GeneralTSPGenerator.generate_n_samples(1)
            test_problem_obj = problems[0]
    except ValidationError as e:
        bt.logging.debug(f"{'Metric TSP' if prob_select==1 else 'General TSP'}")
        bt.logging.debug(f"GraphProblem Validation Error: {e.json()}")
        bt.logging.debug(e.errors())
        bt.logging.debug(e)

    try:
        graphsynapse_req = GraphSynapse(problem=test_problem_obj)
        return graphsynapse_req
    except ValidationError as e:
        bt.logging.debug(f"GraphSynapse Validation Error: {e.json()}")
        bt.logging.debug(e.errors())
        bt.logging.debug(e)



def compare():
    synapse_request = generate_problem()
    t1 = time.time()
    beam_synapse = asyncio.run(beam_solver_solution(synapse_request))
    t2 = time.time()
    baseline_synapse = asyncio.run(baseline_solution(synapse_request))
    t3 = time.time()
    nns_vali_synapse = asyncio.run(nns_vali_solver_solution(synapse_request))
    t4 = time.time()
    hpn_synapse = asyncio.run(hpn_solver_solution(synapse_request))
    t5 = time.time()
    new_synapse = asyncio.run(new_solver_solution(synapse_request))
    t6 = time.time()
    # tsp_annealer_synapse = asyncio.run(tsp_annealer_solver(synapse_request))
    t7 = time.time()
    simulated_annealing_synapse = asyncio.run(simulated_annealing_solver(synapse_request))
    t8 = time.time()

    d1 = t2 - t1
    d2 = t3 - t2
    d3 = t4 - t3
    d4 = t5 - t4
    d5 = t6 - t5
    d6 = t7 - t6
    d7 = t8 - t7
    print(f"d6 = {d6}, d7 ={d7}")

    if d1 > 10 :
        print(f"d1")
        exit(1)
    if d2 > 10 :
        print(f"d2")
        exit(1)
    if d3 > 10 :
        print(f"d3")
        exit(1)
    if d4 > 10 :
        print(f"d4")
        exit(1)
    if d5 > 10 :
        print(f"d5")
        exit(1)
    if d6 > 20 :
        print(f"d6")
        exit(1)
    if d7 > 20 :
        print(f"d7")
        exit(1)

    list_synapse = [beam_synapse, baseline_synapse,nns_vali_synapse,hpn_synapse]
    scores = [scoring_solution(synapse) for synapse in list_synapse]

    min_score = min(scores)
    scores.append(min_score)

    scores.append(scoring_solution(new_synapse))

    min_score1 = min(scores)
    scores.append(min_score1)


    scores.append(scoring_solution(simulated_annealing_synapse))

    min_score3 = min(scores)
    scores.append(min_score3)

    return scores



if __name__ == '__main__':
    synapse_request = generate_problem()
    # print(f"synapse_request = {synapse_request}")
    json_data = json.dumps(synapse_request.problem.dict())
    print(f"synapse_request problem = {json_data}")
    graph_problem_instance = GraphProblem.parse_raw(json_data)
    print(f"GraphProblem instance: {isinstance(graph_problem_instance, GraphProblem)}")

    # synapse = asyncio.run(beam_solver_solution(synapse_request))
    # print(f"route = {synapse.solution}  length = {len(synapse.solution)}")
    # score = scoring_solution(synapse)
    # print(f"score = {score}")