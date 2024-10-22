import asyncio
import json
import random
import time
from typing import Union

import bittensor as bt
import numpy as np
from pydantic import ValidationError

from graphite.data import MetricTSPGenerator, GeneralTSPGenerator
from graphite.data.constants import ASIA_MSB_DETAILS, WORLD_TSP_DETAILS
from graphite.data.dataset_utils import load_dataset
from graphite.data.distance import geom_edges, euc_2d_edges, man_2d_edges
from graphite.protocol import GraphV1Synapse, GraphV2Synapse, GraphV2Problem, GraphV2ProblemMulti
from neurons.call_method import baseline_solution, hpn_solver_solution, \
    scoring_solution, lkh_solver_solution, build_lkh_input_file, mTSP_or_solver_solution, nn_multi_solver_solution, \
    lkh3_mtsp_solver_solution

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
        bt.logging.debug(f"{'Metric TSP' if prob_select == 1 else 'General TSP'}")
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
    prob_select = random.randint(0, len(list(loaded_datasets.keys())) - 1)
    dataset_ref = list(loaded_datasets.keys())[prob_select]
    bt.logging.info(f"n_nodes V2 {n_nodes}")
    bt.logging.info(f"dataset ref {dataset_ref} selected from {list(loaded_datasets.keys())}")
    bt.logging.info(
        f"dataset length {len(loaded_datasets[dataset_ref]['data'])} from {loaded_datasets[dataset_ref]['data'].shape} ")
    selected_node_idxs = random.sample(range(len(loaded_datasets[dataset_ref]['data'])), n_nodes)
    test_problem_obj = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=selected_node_idxs,
                                      cost_function="Geom", dataset_ref=dataset_ref)

    try:
        graphsynapse_req = GraphV2Synapse(problem=test_problem_obj)
        bt.logging.info(f"GraphV2Synapse Problem, n_nodes: {graphsynapse_req.problem.n_nodes}")
    except ValidationError as e:
        bt.logging.debug(f"GraphV2Synapse Validation Error: {e.json()}")
        bt.logging.debug(e.errors())
        bt.logging.debug(e)
    return graphsynapse_req


def generate_problem_for_mTSP(min_node=500, max_node=2000, min_salesman=2, max_salesman=10):
    n_nodes = random.randint(min_node, max_node)
    prob_select = random.randint(0, len(list(loaded_datasets.keys())) - 1)
    dataset_ref = list(loaded_datasets.keys())[prob_select]
    bt.logging.info(f"n_nodes V2 {n_nodes}")
    bt.logging.info(f"dataset ref {dataset_ref} selected from {list(loaded_datasets.keys())}")
    selected_node_idxs = random.sample(range(len(loaded_datasets[dataset_ref]['data'])), n_nodes)
    m = random.randint(min_salesman, max_salesman)
    test_problem_obj = GraphV2ProblemMulti(problem_type="Metric mTSP", n_nodes=n_nodes, selected_ids=selected_node_idxs,
                                           cost_function="Geom", dataset_ref=dataset_ref, n_salesmen=m, depots=[0] * m)
    try:

        graphsynapse_req = GraphV2Synapse(problem=test_problem_obj)
        bt.logging.info(
            f"GraphV2Synapse {graphsynapse_req.problem.problem_type}, n_nodes: {graphsynapse_req.problem.n_nodes}")
    except ValidationError as e:
        bt.logging.debug(f"GraphV2Synapse Validation Error: {e.json()}")
        bt.logging.debug(e.errors())
        bt.logging.debug(e)

    return graphsynapse_req


def mTSP_solve(min_node, max_node, min_salesman=2, max_salesman=3,dataset_ref = 'World_TSP'):
    while True:
        synapse = generate_problem_for_mTSP(min_node=min_node, max_node=max_node, min_salesman=min_salesman,
                                        max_salesman=max_salesman)
        if synapse.problem.dataset_ref == dataset_ref:
            break
    # print(f'synapse = {synapse}')
    if synapse.problem.dataset_ref == 'World_TSP':
        edges = recreate_edges(synapse.problem, factor=1).tolist()
    else:
        edges = recreate_edges(synapse.problem, factor=100).tolist()
    originl_edges = recreate_edges(synapse.problem, factor=1).tolist()
    synapse.problem.edges = edges
    lkh_input_file = build_lkh_input_file(synapse,dir='/home/ubuntu/test_lkh/problem')
    print(f"lkh_input_file = {lkh_input_file}")
    t0 = time.time()
    nn_multi_synapse = asyncio.run(nn_multi_solver_solution(synapse))
    nn_multi_synapse.problem.edges = originl_edges
    t1 = time.time()
    lkh3_mtsp_synapse = asyncio.run(lkh3_mtsp_solver_solution(synapse,num_run=1,input_file=lkh_input_file))
    lkh3_mtsp_synapse.problem.edges = originl_edges
    t2 = time.time()
    print(f'nn_multi_synapse.solution = {nn_multi_synapse.solution}\n\n')
    print(f'lkh3_mtsp_synapse = {lkh3_mtsp_synapse.solution}')
    print(f"time baseline  = {t1 - t0}, time kh3_mtsp = {t2-t1}, num node = {synapse.problem.n_nodes}")

    list_synapse = [nn_multi_synapse,lkh3_mtsp_synapse]
    scores = [scoring_solution(synapse) for synapse in list_synapse]
    min_score = min(scores)
    scores.append(min_score)
    print(f'scores = {scores}')


def compare(gen_func=None, min_node=2000, max_node=5000):
    if gen_func == 'V2':
        synapse_request = generate_problem_from_dataset(min_node=min_node, max_node=max_node)
        start = time.time()
        edges = recreate_edges(synapse_request.problem).tolist()
        end = time.time()
        print(f'time calculating edges: {end - start}')
        synapse_request.problem.edges = edges
        # body_dict = {
        #     'route':edges
        # }
        # json_string = json.dumps(body_dict)
        # print(f'body_dict = {json_string}')
        #
        # with open(f'test_{time.time()}.txt', 'w') as file:
        #     # Write the string to the file
        #     file.write(str(edges.tolist()))
    else:
        synapse_request = generate_problem()
    print(f'Number of node: {synapse_request.problem.n_nodes}')

    t1 = time.time()
    lkh_input_file = build_lkh_input_file(synapse_request)
    print(f"lkh_input_file = {lkh_input_file}")
    # beam_synapse = asyncio.run(beam_solver_solution(synapse_request))
    t2 = time.time()
    print(f'time build input file: {t2 - t1}')
    baseline_synapse = asyncio.run(baseline_solution(synapse_request))
    t3 = time.time()
    # nns_vali_synapse = asyncio.run(nns_vali_solver_solution(synapse_request))
    t4 = time.time()
    lkh_synapse = asyncio.run(lkh_solver_solution(synapse_request, input_file=lkh_input_file))
    t5 = time.time()
    # new_synapse = asyncio.run(new_solver_solution(synapse_request))
    t6 = time.time()
    lkh_synapse_3 = asyncio.run(lkh_solver_solution(synapse_request, num_run=3, input_file=lkh_input_file))
    t7 = time.time()

    # time_point = [t1, t2, t3, t4, t5, t6, t7]
    # time_processing_list = []
    # for i in range(1, len(time_point)):
    #     time_processing_list.append(time_point[i] - time_point[i - 1])
    #
    # for i in range(len(time_processing_list)):
    #     if time_processing_list[i] > 20:
    #         print(f'time process of {i} > 20')
    #         exit(0)

    print(
        f"time baseline = {t3 - t2}, time lkh = {t5 - t4}, time lkh3 = {t7 - t6} ,num node = {synapse_request.problem.n_nodes}")
    list_synapse = [baseline_synapse, lkh_synapse, lkh_synapse_3]
    scores = [scoring_solution(synapse) for synapse in list_synapse]
    # scores = [1e+20, 1e+20, 1e+20]
    #
    min_score = min(scores)
    scores.append(min_score)
    #
    # # scores.append(scoring_solution(new_synapse))
    # scores.append(1e+20)
    #
    # min_score1 = min(scores)
    # scores.append(min_score1)
    #
    # scores.append(scoring_solution(lkh_synapse))
    #
    # min_score2 = min(scores)
    # scores.append(min_score2)
    #
    # # scores.append(scoring_solution(simulated_annealing_synapse))
    # scores.append(1e+20)
    #
    # min_score3 = min(scores)
    # scores.append(min_score3)
    print(f'score = {scores}')
    return scores


def recreate_edges(problem: Union[GraphV2Problem, GraphV2ProblemMulti], factor=1):
    node_coords_np = loaded_datasets[problem.dataset_ref]["data"]
    node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
    if problem.cost_function == "Geom":
        return geom_edges(node_coords, factor=factor)
    elif problem.cost_function == "Euclidean2D":
        return euc_2d_edges(node_coords)
    elif problem.cost_function == "Manhatten2D":
        return man_2d_edges(node_coords)
    else:
        return "Only Geom, Euclidean2D, and Manhatten2D supported for now."


def review_solution_new_meta():
    synapse_request = generate_problem_from_dataset(min_node=200, max_node=500)
    # synapse_request = generate_problem()
    print(f'Number of node: {synapse_request.problem.n_nodes}')
    t1 = time.time_ns()
    edges = recreate_edges(synapse_request.problem)
    synapse_request.problem.edges = edges
    synapse = asyncio.run(hpn_solver_solution(synapse_request))
    t2 = time.time_ns()
    print(f'time processing: {(t2 - t1) / 1e6} ms')
    score = scoring_solution(synapse)
    print(f'score = {score}')


def calculate_raw_data(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    dataset_ref = config['dataset_ref']['value']
    selected_ids = config['selected_ids']['value']
    cost_function = config['cost_function']['value']
    objective_function = config['objective_function']['value']
    problem_type = config['problem_type']['value']
    n_nodes = config['n_nodes']['value']
    test_graph_problem = GraphV2Problem(problem_type=problem_type, objective_function=objective_function,
                                        n_nodes=n_nodes, selected_ids=selected_ids,
                                        cost_function="Geom", dataset_ref=dataset_ref)

    synapse_request = GraphV2Synapse(problem=test_graph_problem)
    print(f'synapse_request = {synapse_request}')
    edges = recreate_edges(synapse_request.problem)
    synapse_request.problem.edges = edges
    t1 = time.time()
    baseline_synapse = asyncio.run(baseline_solution(synapse_request))
    t2 = time.time()
    lkh_synapse = asyncio.run(lkh_solver_solution(synapse_request))
    t3 = time.time()
    print(f"time baseline  = {t2 - t1}, time lkh = {t3 - t2}, num node = {synapse_request.problem.n_nodes}")
    list_synapse = [baseline_synapse, lkh_synapse]
    scores = [scoring_solution(synapse) for synapse in list_synapse]

    min_score = min(scores)
    scores.append(min_score)
    print(f'score = {scores}')
    return scores


if __name__ == '__main__':
    # config_file = 'raw_data/data.json'
    # calculate_raw_data(config_file)

    synapse_request = generate_problem_for_mTSP(min_node=500, max_node=2000)
    problem_dict = synapse_request.problem.dict()
    # json_problem = json.dumps(problem_dict)
    payload = json.dumps({
        "problem": problem_dict,
        "hash": 'abcdef',
        "config_file_path": 'config.json'
    })
    print(f'payload = {payload}')
    #
    # data = json.loads(payload)
    # problem = data['problem']
    # dataset_ref = problem.get('dataset_ref')
    # print(f'dataset_ref: {dataset_ref}')
    # graph_problem = GraphV2Problem.parse_obj(problem)
    # graphsynapse_req = GraphV2Synapse(problem=graph_problem)
    #
    # t1 = time.time_ns()
    # edges = recreate_edges(graphsynapse_req.problem)
    # graphsynapse_req.problem.edges = edges
    # synapse = asyncio.run(baseline_solution(graphsynapse_req))
    # t2 = time.time_ns()
    # print(f'time processing: {(t2-t1)/1e6} ms')
    # score = scoring_solution(synapse)
    # print(f'score = {score}')

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
