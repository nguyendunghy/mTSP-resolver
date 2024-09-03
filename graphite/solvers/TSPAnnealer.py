import numpy as np
from simanneal import Annealer


class TSPAnnealer(Annealer):
    def __init__(self, state, edges):
        self.edges = edges
        super(TSPAnnealer, self).__init__(state)

    def move(self):
        """Make a small change in the tour, ensuring the first city is fixed."""
        a = np.random.randint(1, len(self.state))  # start index from 1 to keep the first city fixed
        b = np.random.randint(1, len(self.state))
        self.state[a], self.state[b] = self.state[b], self.state[a]

    def energy(self):
        """Calculate the total weight of the current tour."""
        total_distance = 0
        for i in range(len(self.state) - 1):
            total_distance += self.edges[self.state[i]][self.state[i + 1]]
        total_distance += self.edges[self.state[-1]][self.state[0]]  # Add distance to return to the start
        return total_distance


# # Input: Ma trận trọng số của các cạnh (2D array)
# edges = [
#     [30, 22, 32, 71, 2],
#     [47, 12, 71, 57, 5],
#     [48, 5, 2, 10, 20],
#     [65, 78, 63, 75, 80],
#     [13, 29, 54, 81, 47]
# ]

# # # Tạo trạng thái ban đầu cho Simulated Annealing
# # init_state = list(range(1, len(edges)))  # Khởi tạo trạng thái ban đầu không bao gồm đỉnh 0
# # init_state.insert(0, 0)  # Đảm bảo đỉnh 0 là điểm xuất phát

# # # Khởi tạo Simulated Annealing với trạng thái ban đầu
# # tsp_annealer = TSPAnnealer(init_state, edges)

# # # Thực hiện Simulated Annealing
# # best_state, best_fitness = tsp_annealer.anneal()

# # # In kết quả
# # print("Lộ trình tốt nhất:", best_state)
# # print("Tổng quãng đường:", best_fitness)