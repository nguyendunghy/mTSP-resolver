import asyncio
import json
import random
import time
import numpy as np
import bittensor as bt
from pydantic import ValidationError

from graphite.data.constants import ASIA_MSB_DETAILS, WORLD_TSP_DETAILS
from graphite.data.dataset_utils import load_dataset
from graphite.data.distance import geom_edges, euc_2d_edges, man_2d_edges
from graphite.dataset.dataset_generator import MetricTSPGenerator, GeneralTSPGenerator
from graphite.protocol import GraphV1Synapse,GraphV2Synapse, GraphV1Problem, GraphV2Problem
from neurons.call_method import beam_solver_solution, baseline_solution, nns_vali_solver_solution, hpn_solver_solution, \
    scoring_solution, new_solver_solution, tsp_annealer_solver, simulated_annealing_solver, or_solver_solution, \
    lkh_solver_solution, lin_kernighan_solution

loaded_datasets = {
    ASIA_MSB_DETAILS['ref_id']: load_dataset(ASIA_MSB_DETAILS['ref_id']),
    WORLD_TSP_DETAILS['ref_id']: load_dataset(WORLD_TSP_DETAILS['ref_id'])
}

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
        graphsynapse_req = GraphV1Synapse(problem=test_problem_obj)
        return graphsynapse_req
    except ValidationError as e:
        bt.logging.debug(f"GraphSynapse Validation Error: {e.json()}")
        bt.logging.debug(e.errors())
        bt.logging.debug(e)


def generate_problem_from_dataset(min_node=2000, max_node=5000):
    n_nodes = random.randint(min_node, max_node)
    # randomly select n_nodes indexes from the selected graph
    prob_select = random.randint(0, len(list(loaded_datasets.keys()))-1)
    dataset_ref = list(loaded_datasets.keys())[prob_select]
    bt.logging.info(f"n_nodes V2 {n_nodes}")
    bt.logging.info(f"dataset ref {dataset_ref} selected from {list(loaded_datasets.keys())}" )
    bt.logging.info(f"dataset length {len(loaded_datasets[dataset_ref]['data'])} from {loaded_datasets[dataset_ref]['data'].shape} " )
    selected_node_idxs = random.sample(range(len(loaded_datasets[dataset_ref]['data'])), n_nodes)
    test_problem_obj = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref=dataset_ref)

    try:
        graphsynapse_req = GraphV2Synapse(problem=test_problem_obj)
        bt.logging.info(f"GraphV2Synapse Problem, n_nodes: {graphsynapse_req.problem.n_nodes}")
    except ValidationError as e:
        bt.logging.debug(f"GraphV2Synapse Validation Error: {e.json()}")
        bt.logging.debug(e.errors())
        bt.logging.debug(e)
    return graphsynapse_req


def compare(gen_func=None, min_node = 2000, max_node = 5000):
    if gen_func == 'V2':
        synapse_request = generate_problem_from_dataset(min_node=min_node,max_node=max_node)
    else:
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
    simulated_annealing_synapse = asyncio.run(simulated_annealing_solver(synapse_request))
    t7 = time.time()

    time_point = [t1, t2, t3, t4, t5, t6, t7]
    time_processing_list = []
    for i in range(1, len(time_point)):
        time_processing_list.append(time_point[i] - time_point[i - 1])

    for i in range(len(time_processing_list)):
        if time_processing_list[i] > 20:
            print(f'time process of {i} > 20')
            exit(0)


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

def recreate_edges(problem: GraphV2Problem):
    node_coords_np = loaded_datasets[problem.dataset_ref]["data"]
    node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
    if problem.cost_function == "Geom":
        return geom_edges(node_coords)
    elif problem.cost_function == "Euclidean2D":
        return euc_2d_edges(node_coords)
    elif problem.cost_function == "Manhatten2D":
        return man_2d_edges(node_coords)
    else:
        return "Only Geom, Euclidean2D, and Manhatten2D supported for now."

if __name__ == '__main__':
    synapse_request = generate_problem_from_dataset(min_node=2000, max_node=2001)
    # synapse_request = generate_problem()
    print(f'Number of node: {synapse_request.problem.n_nodes}')
    t1 = time.time_ns()
    edges = recreate_edges(synapse_request.problem)
    from python_tsp.heuristics import solve_tsp_lin_kernighan
    result = solve_tsp_lin_kernighan(edges)
    t2 = time.time_ns()
    print(f'time processing: {(t2-t1)/1e6} ms')

    print(f'score = {result}')

    # synapse_request = generate_problem()
    # # print(f"synapse_request = {synapse_request}")
    # json_data = json.dumps(synapse_request.problem.dict())
    # print(f"synapse_request problem = {json_data}")
    # graph_problem_instance = GraphV1Problem.parse_raw(json_data)
    # print(f"GraphProblem instance: {isinstance(graph_problem_instance, GraphV1Problem)}")

    # synapse = asyncio.run(beam_solver_solution(synapse_request))
    # print(f"route = {synapse.solution}  length = {len(synapse.solution)}")
    # score = scoring_solution(synapse)
    # print(f"score = {score}")
    # import numpy as np
    # print('________________________')
    # print('Testing MetricTSPGenerator V2')
    # loaded_datasets = {}
    # try:
    #     with np.load('dataset/Asia_MSB.npz') as f:
    #         loaded_datasets["Asia_MSB"] = np.array(f['data'])
    # except:
    #     pass
    # try:
    #     with np.load('dataset/World_TSP.npz') as f:
    #         loaded_datasets["World_TSP"] = np.array(f['data'])
    # except:
    #     pass

    # synapse_req = generate_problem_from_dataset()
    # # print(f'problem = {synapse_req.problem}')
    # start_time = time.time_ns()
    # edges = recreate_edges(synapse_req.problem)
    # end_time = time.time_ns()
    # synapse_req.problem.edges = edges
    # print(f'time processing convert edge {(end_time - start_time)/1e6} ms with number of node: {synapse_req.problem.n_nodes}')
    # # print(f'edges = {edges}')
    # time.sleep(0.2)

