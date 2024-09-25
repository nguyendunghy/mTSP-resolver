import numpy as np
import tqdm

from neurons.compare_solutions import compare

if __name__ == '__main__':
    data = [compare(gen_func='V2', min_node=5, max_node=10) for i in tqdm.tqdm(range(20))]
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