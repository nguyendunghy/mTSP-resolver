# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import List
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
from graphite.utils.graph_utils import timeout
import asyncio
import time



def nearest_neighbor(distance_matrix):
    num_cities = len(distance_matrix)
    best_path = [0]  # Start from city 0
    visited = [False] * num_cities
    visited[0] = True

    for _ in range(num_cities - 1):
        current_city = best_path[-1]
        next_city = -1
        min_distance = float('inf')

        for city in range(num_cities):
            if not visited[city] and distance_matrix[current_city][city] < min_distance:
                min_distance = distance_matrix[current_city][city]
                next_city = city

        best_path.append(next_city)
        visited[next_city] = True

    best_path.append(0)  # Return to the starting city
    return best_path

def calculate_total_distance(path, distance_matrix):
    return sum(distance_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))

def two_opt(path, distance_matrix):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path) - 1):
                if j - i == 1: 
                    continue  # No change if the same edge
                new_path = path[:i] + path[i:j][::-1] + path[j:]
                if calculate_total_distance(new_path, distance_matrix) < calculate_total_distance(path, distance_matrix):
                    path = new_path
                    improved = True
    return path

def tsp_with_time_limit(distance_matrix, time_limit=5.0):
    start_time = time.time()
    # Step 1: Get the initial path using Nearest Neighbor
    best_path = nearest_neighbor(distance_matrix)
    
    # Step 2: Improve the path using 2-opt until time runs out
    while time.time() - start_time < time_limit:
        new_path = two_opt(best_path, distance_matrix)
        if calculate_total_distance(new_path, distance_matrix) < calculate_total_distance(best_path, distance_matrix):
            best_path = new_path
        else:
            break  # Stop if no improvement
    
    return best_path

class NewSearchSolver(BaseSolver):
    def __init__(self, problem_types:List[GraphProblem]=[GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem, future_id:int, beam_width:int=3)->List[int]:
        distance_matrix = formatted_problem
        #best_path=nearest_neighbor(distance_matrix)
        #best_path = simulated_annealing(distance_matrix, initial_temp, cooling_rate, max_iterations)
        best_path = nearest_neighbor(distance_matrix)
        start=time.time()
        itercount=0
        best_path=tsp_with_time_limit(distance_matrix)
        return best_path

    def problem_transformations(self, problem: GraphProblem):
        return problem.edges
    
if __name__=='__main__':
    # runs the solver on a test MetricTSP
    n_nodes = 100
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = NewSearchSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
    
