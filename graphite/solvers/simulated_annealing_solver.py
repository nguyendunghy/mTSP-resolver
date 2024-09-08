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

import asyncio
import math
import random
import time
from typing import List

from graphite.protocol import GraphV1Problem,GraphV2Problem
from graphite.solvers.base_solver import BaseSolver
from typing import Union

def simulated_annealing(path, distance_matrix, initial_temp=1000, cooling_rate=0.995, time_limit=4.9):
    start_time = time.time()
    current_path = path
    current_distance = calculate_total_distance(current_path, distance_matrix)
    best_path = list(current_path)
    best_distance = current_distance
    temperature = initial_temp

    while time.time() - start_time < time_limit and temperature > 1:
        # Randomly swap two cities to create a new path
        i, j = random.sample(range(1, len(current_path) - 1), 2)
        new_path = list(current_path)
        new_path[i], new_path[j] = new_path[j], new_path[i]
        
        new_distance = calculate_total_distance(new_path, distance_matrix)
        delta_distance = new_distance - current_distance

        if delta_distance < 0 or math.exp(-delta_distance / temperature) > random.random():
            current_path = new_path
            current_distance = new_distance
            if current_distance < best_distance:
                best_path = list(current_path)
                best_distance = current_distance

        temperature *= cooling_rate

    return best_path

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
    best_path = nearest_neighbor(distance_matrix)
    
    # Step 2: Improve the path using 2-opt
    best_path = two_opt(best_path, distance_matrix)
    remaining_time = time_limit - (time.time() - start_time)
    best_path = simulated_annealing(best_path, distance_matrix, time_limit=remaining_time)

    return best_path

class SimulatedAnnealingSolver(BaseSolver):
    def __init__(self, problem_types:List[Union[GraphV1Problem, GraphV2Problem]]=[GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem, future_id:int, beam_width:int=3)->List[int]:
        distance_matrix = formatted_problem
        # best_path=nearest_neighbor(distance_matrix)
        # best_path = simulated_annealing(distance_matrix, initial_temp, cooling_rate, max_iterations)
        # best_path = nearest_neighbor(distance_matrix)
        # start=time.time()
        # itercount=0
        best_path=tsp_with_time_limit(distance_matrix)
        return best_path

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges
    
if __name__=='__main__':
    # runs the solver on a test MetricTSP
    n_nodes = 100
    test_problem = GraphV1Problem(n_nodes=n_nodes)
    solver = SimulatedAnnealingSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
    
