import random
import time

from graphite.data.distance import euc_2d_edges, geom_edges, man_2d_edges
from graphite.protocol import GraphV2Problem, GraphV2Synapse
import numpy as np


def recreate_edges(problem: GraphV2Problem):
    with np.load('dataset/Asia_MSB.npz') as f: # "dataset/Asia_MSB.npz"
        node_coords_np = f['data']
    # loaded_datasets["Asia_MSB"] = np.array(node_coords_np)
    # node_coords_np = self.loaded_datasets[problem.dataset_ref]["data"]
    node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
    if problem.cost_function == "Geom":
        return geom_edges(node_coords)
    elif problem.cost_function == "Euclidean2D":
        return euc_2d_edges(node_coords)
    elif problem.cost_function == "Manhatten2D":
        return man_2d_edges(node_coords)
    else:
        return "Only Geom, Euclidean2D, and Manhatten2D supported for now."


def get_lat_long(problem: GraphV2Problem):
    with np.load('dataset/Asia_MSB.npz') as f: # "dataset/Asia_MSB.npz"
        node_coords_np = f['data']
    node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
    return node_coords

n_nodes = 3000
# randomly select n_nodes indexes from the selected graph
selected_node_idxs = random.sample(range(26000000), n_nodes)
test_problem = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref="Asia_MSB")
from python_tsp.heuristics import solve_tsp_lin_kernighan
start = time.time()
edges = recreate_edges(test_problem)
result = solve_tsp_lin_kernighan(edges)
print(result, time.time() - start)