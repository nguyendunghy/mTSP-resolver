import numpy as np
import tqdm

from neurons.compare_solutions import compare

if __name__ == '__main__':
    data = [compare() for i in tqdm.tqdm(range(2000))]
    data = np.array(data)

    print("BEAM:", data[:, 0].mean())
    print("BASELINE:", data[:, 1].mean())
    print("NNS_VALI :", data[:, 2].mean())
    print("HPN:", data[:, 3].mean())
    print("MIN:", data[:, 4].mean())
    print("NEW:", data[:, 5].mean())
    print("MIN 1:", data[:, 6].mean())
