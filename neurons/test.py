import numpy as np
import tqdm

from neurons.compare_solutions import compare

if __name__ == '__main__':
    data = [compare(gen_func='V2', min_node=2000, max_node=5000) for i in tqdm.tqdm(range(10))]
    data = np.array(data)

    print("BEAM:", data[:, 0].mean())
    print("BASELINE:", data[:, 1].mean())
    print("NNS_VALI :", data[:, 2].mean())
    print("HPN:", data[:, 3].mean())
    print("MIN:", data[:, 4].mean())
    print("NEW:", data[:, 5].mean())
    print("MIN 1:", data[:, 6].mean())
    # print("tsp_annealer :", data[:, 7].mean())
    # print("MIN 2:", data[:, 8].mean())
    print("simulated_annealing :", data[:, 7].mean())
    print("MIN 3:", data[:, 8].mean())