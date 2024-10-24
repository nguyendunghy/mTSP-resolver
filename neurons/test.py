import numpy as np
import tqdm

from neurons.compare_solutions import solve_mTSP


def check_mTSP():
    data = [solve_mTSP(min_node=31, max_node=35, min_salesman=2, max_salesman=10) for i in tqdm.tqdm(range(20))]
    data = np.array(data)

    print("BASELINE:", data[:, 0].mean())
    print("LKH:", data[:, 1].mean())
    print("MIN:", data[:, 2].mean())


if __name__ == '__main__':
    check_mTSP()
