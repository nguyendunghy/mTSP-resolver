import time

import numpy as np
import tqdm
import asyncio
from neurons.call_api import call_apis, load_config
from neurons.compare_solutions import compare, recreate_edges, generate_problem_from_dataset, mTSP_solve


def check_solutions():
    data = [compare(gen_func='V2', min_node=4500, max_node=5000) for i in tqdm.tqdm(range(20))]
    data = np.array(data)

    print("BASELINE:", data[:, 0].mean())
    print("LKH:", data[:, 1].mean())
    print("LKH 3 :", data[:, 2].mean())
    # # print("HPN:", data[:, 3].mean())
    print("MIN:", data[:, 3].mean())
    # print("NEW:", data[:, 4].mean())
    # print("MIN 1:", data[:, 5].mean())
    # print("lkh_annealer :", data[:, 6].mean())
    # print("MIN 2:", data[:, 7].mean())
    # print("simulated_annealing :", data[:, 8].mean())
    # print("MIN 3:", data[:, 9].mean())


def check_call_api():
    start = time.time()
    synapse = generate_problem_from_dataset(min_node=2000, max_node=5000)
    config = load_config()
    edges = recreate_edges(synapse.problem).tolist()
    synapse.problem.edges = edges
    route = asyncio.run(call_apis(synapse, config))
    synapse.solution = route
    end = time.time()
    print(f'time processing total : {end-start}')

def check_mtsp():
    mTSP_solve(min_node=1500, max_node=1500, min_salesman=2, max_salesman=2, dataset_ref='World_TSP')


if __name__ == '__main__':
    check_mtsp()
    # count  = 0
    # while count < 20:
    #     check_call_api()
    #     count = count + 1
    #     time.sleep(2)
