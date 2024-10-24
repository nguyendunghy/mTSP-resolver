import asyncio
import os
import random
import time
from typing import Union
import bittensor as bt
import numpy as np
from pydantic import ValidationError
from graphite.data.constants import ASIA_MSB_DETAILS, WORLD_TSP_DETAILS
from graphite.data.dataset_utils import load_dataset
from graphite.data.distance import geom_edges, euc_2d_edges, man_2d_edges
from graphite.protocol import GraphV2Synapse, GraphV2Problem, GraphV2ProblemMulti
from neurons.call_method import scoring_solution, build_lkh_input_file, nn_multi_solver_solution, \
    lkh3_mtsp_solver_solution

loaded_datasets = {
    ASIA_MSB_DETAILS['ref_id']: load_dataset(ASIA_MSB_DETAILS['ref_id']),
    WORLD_TSP_DETAILS['ref_id']: load_dataset(WORLD_TSP_DETAILS['ref_id'])
}


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


def solve_mTSP(min_node, max_node, min_salesman=2, max_salesman=10):
    synapse = generate_problem_for_mTSP(min_node=min_node, max_node=max_node, min_salesman=min_salesman,
                                        max_salesman=max_salesman)
    if synapse.problem.dataset_ref == 'World_TSP':
        edges = recreate_edges(synapse.problem, factor=10).tolist()
    else:
        edges = recreate_edges(synapse.problem, factor=100).tolist()

    original_edges = recreate_edges(synapse.problem, factor=1).tolist()
    synapse.problem.edges = edges
    print(f'n_node = {synapse.problem.n_nodes}, n_salesmen = {synapse.problem.n_salesmen}')

    lkh_input_file = build_lkh_input_file(synapse, dir=f'{os.getcwd()}/problem')
    print(f"lkh_input_file = {lkh_input_file}")

    t0 = time.time()
    nn_multi_synapse = asyncio.run(nn_multi_solver_solution(synapse))
    nn_multi_synapse.problem.edges = original_edges
    t1 = time.time()
    lkh3_mtsp_synapse = asyncio.run(lkh3_mtsp_solver_solution(synapse, num_run=1, input_file=lkh_input_file))
    lkh3_mtsp_synapse.problem.edges = original_edges
    t2 = time.time()

    print(f'nn_multi_synapse.solution = {nn_multi_synapse.solution}\n\n')
    print(f'lkh3_mtsp_synapse = {lkh3_mtsp_synapse.solution}')
    print(f"time baseline  = {t1 - t0}, time kh3_mtsp = {t2 - t1}, num node = {synapse.problem.n_nodes}")

    list_synapse = [nn_multi_synapse, lkh3_mtsp_synapse]
    scores = [scoring_solution(synapse) for synapse in list_synapse]
    min_score = min(scores)
    scores.append(min_score)
    print(f'scores = {scores}')
    return scores
