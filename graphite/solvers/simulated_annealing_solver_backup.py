import random
import math
import networkx as nx
import time
from concurrent.futures import ThreadPoolExecutor


class SimulatedAnnealingSolver:
    def __init__(self, edges, initial_temperature=1000, cooling_rate=0.995, min_temperature=1e-5, max_iterations=10000, num_restarts=3):
        self.edges = edges
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.num_restarts = num_restarts
        self.graph = self.create_graph(edges)

    def create_graph(self, edges):
        graph = nx.Graph()
        for u, v, weight in edges:
            graph.add_edge(u, v, weight=weight)
        return graph

    def solve(self):
        best_solution = None
        best_cost = float('inf')

        for _ in range(self.num_restarts):
            current_solution = self.create_initial_solution()
            current_cost = self.calculate_cost(current_solution)

            temperature = self.initial_temperature
            iteration = 0

            start_time = time.time()

            while temperature > self.min_temperature and iteration < self.max_iterations:
                new_solution = self.get_optimized_neighbor_solution(current_solution)
                new_cost = self.calculate_cost(new_solution)

                if self.acceptance_probability(current_cost, new_cost, temperature) > random.random():
                    current_solution = new_solution
                    current_cost = new_cost

                    if current_cost < best_cost:
                        best_solution = current_solution
                        best_cost = current_cost

                temperature *= self.cooling_rate
                iteration += 1

                if time.time() - start_time > 5:  # Limit runtime
                    break

        return best_solution, best_cost

    def create_initial_solution(self):
        # Using Christofides to create a high-quality initial solution
        mst = nx.minimum_spanning_tree(self.graph)
        odd_degree_nodes = [v for v, d in mst.degree() if d % 2 == 1]
        subgraph = self.graph.subgraph(odd_degree_nodes)
        matching = nx.algorithms.matching.max_weight_matching(subgraph, maxcardinality=True)
        multigraph = nx.MultiGraph(mst)
        multigraph.add_edges_from(matching)
        eulerian_circuit = list(nx.eulerian_circuit(multigraph))
        tsp_tour = []
        visited = set()
        for u, v in eulerian_circuit:
            if u not in visited:
                visited.add(u)
                tsp_tour.append(u)
        tsp_tour.append(tsp_tour[0])  # Return to starting node

        return tsp_tour

    def calculate_cost(self, solution):
        cost = 0
        for i in range(len(solution) - 1):
            u = solution[i]
            v = solution[i + 1]
            edge_cost = self.get_edge_cost(u, v)
            cost += edge_cost
        return cost

    def get_edge_cost(self, u, v):
        return next((weight for (x, y, weight) in self.edges if (x == u and y == v) or (x == v and y == u)), float('inf'))

    def get_optimized_neighbor_solution(self, solution):
        # Apply 3-Opt for improved local search
        neighbor = self.get_3opt_solution(solution)
        return neighbor

    def acceptance_probability(self, current_cost, new_cost, temperature):
        if new_cost < current_cost:
            return 1.0
        else:
            return math.exp((current_cost - new_cost) / temperature)

    def get_3opt_solution(self, solution):
        best = solution[:]
        best_cost = self.calculate_cost(best)
        improved = True

        while improved:
            improved = False
            n = len(best)
            for i in range(n - 2):
                for j in range(i + 2, n - 1):
                    for k in range(j + 2, n):
                        new_solution = self.apply_3opt_swap(best, i, j, k)
                        new_cost = self.calculate_cost(new_solution)

                        if new_cost < best_cost:
                            best = new_solution
                            best_cost = new_cost
                            improved = True

        return best

    def apply_3opt_swap(self, tour, i, j, k):
        # Perform a 3-Opt swap
        new_tour = tour[:]
        if j > i + 1 and k > j + 1:
            new_tour = tour[:i+1] + tour[j:k+1][::-1] + tour[k+1:]
        return new_tour

# Example usage:
if __name__ == "__main__":
    # Replace with actual edges from your problem instance
    edges = [(0, 1, 10), (1, 2, 20), (2, 3, 30), (3, 0, 40)]
    solver = SimulatedAnnealingSolver(edges)
    best_solution, best_cost = solver.solve()
    print(f"Best Solution: {best_solution}")
    print(f"Best Cost: {best_cost}")
