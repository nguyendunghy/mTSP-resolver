import asyncio
import time
from argparse import ArgumentParser
from pydantic import BaseModel
from graphite.protocol import GraphV2Problem
from graphite.solvers.lkh_solver import LKHSolver
from flask import Flask, request, jsonify

app = Flask(__name__)


def parse():
    parser = ArgumentParser()
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on.')
    parser.add_argument('--num_run', type=str, default='', help='Number of running')
    parser.add_argument('--max_trial', type=str, default='', help='Maximum of trial')
    parser.add_argument('--max_node', type=str, default=5000, help='Maximum of trial')
    return parser.parse_args()


args = parse()


class InputProblem(BaseModel):
    input_file_path: str
    n_nodes: int
    dataset_ref: str
    timeout: int


@app.route('/', methods=['GET'])
def hello_world():
    return "Hello! I am resolver service"


@app.route('/lkh_resolve', methods=['POST'])
def lkh_resolve():
    if request.is_json:
        start_time = time.time_ns()
        problem = request.get_json()
        input_file = problem['input_file_path']
        n_nodes = problem['n_nodes']
        dataset_ref = problem['dataset_ref']
        timeout = problem['timeout']
        if n_nodes > args.max_node:
            print(f'problem too large. Num node = {n_nodes}')
            return jsonify({"message": f"PROBLEM TOO LARGE", "result": None, "score": 0}), 200

        lkh_solver = LKHSolver(num_run=args.num_run, max_trial=args.max_trial, input_file=input_file)
        graph_problem = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=[0],
                                       cost_function="Geom",
                                       dataset_ref=dataset_ref, directed=False)

        solution = asyncio.run(lkh_solver.solve_problem(graph_problem, timeout))

        print(f"time loading {int(time.time_ns() - start_time):,} nanosecond")
        return jsonify({"message": f"SUCCESS", "result": solution, "score": 0}), 200
    else:
        return jsonify({"error": "Request must be JSON"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=args.port)
