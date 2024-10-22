import os
import random
import subprocess
import traceback
from typing import List
from typing import Union

from graphite.protocol import GraphV1Problem, GraphV2Problem, GraphV2ProblemMulti
from graphite.solvers.base_solver import BaseSolver


class LKH3_MTSP_Solver(BaseSolver):
    def __init__(self, problem_types: List[GraphV2Problem] = [GraphV2ProblemMulti()],
                 num_run=1,
                 init_tour_algo='NE',
                 max_trial=1,
                 input_file=None):
        super().__init__(problem_types=problem_types)
        self.lkh_path = 'LKH-3.0.11/LKH'
        # self.lkh_path= 'mTSP/run_MTSP'# Update with the actual path to LKH
        self.num_run = num_run
        self.init_tour_algo = init_tour_algo
        self.max_trial = max_trial
        self.input_file = input_file

    def write_tsplib_file(self, distance_matrix: List[List[int]], filename: str, directed):
        """Writes a distance matrix to a TSPLIB formatted file."""
        problem_type = "ATSP" if directed else "TSP"
        n = len(distance_matrix)
        # if(directed == False):
        # with open(filename, 'w') as f:
        #     f.write(f"NAME: {problem_type}\n")
        #     f.write(f"TYPE: {problem_type}\n")
        #     f.write(f"DIMENSION: {n}\n")
        #     f.write(f"EDGE_WEIGHT_TYPE: EUC_2D\n")
        #     f.write("NODE_COORD_SECTION\n")
        #     for i, (x, y) in enumerate(distance_matrix, start=1):
        #         f.write(f"{i} {x * 100} {y * 100}\n")
        #     f.write("EOF\n")
        # else:
        with open(filename, 'w') as f:
            f.write(f"NAME: {problem_type}\n")
            f.write(f"TYPE: {problem_type}\n")
            f.write(f"DIMENSION: {n}\n")
            f.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\n")
            f.write(f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
            f.write(f"EDGE_WEIGHT_SECTION\n")
            for row in distance_matrix:
                f.write(" ".join(map(str, row)) + "\n")
            f.write("EOF\n")

    def write_lkh_parameters(self, filename: str, problem_filename: str, tour_filename: str):
        """Writes the parameter file for LKH."""
        with open(filename, 'w') as f:
            f.write(f"PROBLEM_FILE = {problem_filename}\n")
            # f.write(f"SINTEF_SOLUTION_FILE = {tour_filename}\n")
            f.write(f"MTSP_SOLUTION_FILE = {tour_filename}\n")
            f.write(f"MTSP_MIN_SIZE = 1\n")
            f.write(f"SALESMEN = {self.n_salesmen}\n")
            # f.write(f"VEHICLES = 3\n")
            f.write(f"DEPOT = 1\n")
            f.write(f"MTSP_OBJECTIVE = MINMAX\n")
            f.write(f"CANDIDATE_SET_TYPE = ALPHA\n")
            f.write(f"INITIAL_PERIOD = 5\n")
            f.write(f"MAX_TRIALS = {self.max_trial}\n")
            f.write(f"INITIAL_TOUR_ALGORITHM = {self.init_tour_algo}\n")
            f.write(f"MAX_CANDIDATES = 3\n")
            f.write(f"RUNS = {self.num_run}\n")
            f.write("EOF\n")

    def run_lkh(self, parameter_file: str):
        """Runs the LKH solver using a given parameter file."""
        result = subprocess.run([self.lkh_path, parameter_file], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"LKH failed: {result.stderr}")
        # print(f'result = {result}')
        # print(f'result.stdout = {result.stdout}')
        return result.stdout

    def read_lkh_solution(self, tour_filename: str):
        """Reads the solution produced by LKH."""
        print(f'start read read_lkh_solution')
        paths = []
        with open(tour_filename, 'r') as f:
            lines = f.readlines()
            for line in lines[2:]:
                line = line.strip()
                path = line.split(' ')
                path = path[:len(path) - 4]
                path = [(int(ele) - 1) for ele in path]
                paths.append(path)
        return paths

    def build_scaled_distance_matrix(self, problem):
        distance_matrix = problem.edges
        is_float = isinstance(distance_matrix[0][0], float)

        scale_factor = 1 if is_float else 1

        scaled_distance_matrix = [
            [int(round(distance * scale_factor)) for distance in row]
            for row in distance_matrix
        ]
        return scaled_distance_matrix

    async def solve(self, problem, future_id: int) -> List[int]:
        try:
            directed = problem.directed
            self.n_salesmen = problem.n_salesmen

            random_number = random.randint(10000, 999999)
            problem_filename = f"{random_number}_problem.tsp" if self.input_file is None else self.input_file
            parameter_filename = f"{random_number}_params.par"
            tour_filename = f"{random_number}_solution.tour"

            if self.input_file is None:
                scaled_distance_matrix = self.build_scaled_distance_matrix(problem)
                self.write_tsplib_file(scaled_distance_matrix, problem_filename, directed)
            print(f'start run write params')
            self.write_lkh_parameters(parameter_filename, problem_filename, tour_filename)
            print(f'start run lkh')
            self.run_lkh(parameter_filename)

            tour = self.read_lkh_solution(tour_filename)

            # if self.input_file is None:
                # os.remove(problem_filename)

            # os.remove(parameter_filename)
            # os.remove(tour_filename)
            return tour
        except Exception as e:
            print(f'error = {e}')
            traceback.print_stack()

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem


if __name__ == '__main__':
    n_nodes = 100  # Adjust as needed

