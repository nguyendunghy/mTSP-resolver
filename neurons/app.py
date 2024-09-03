import asyncio
import time
from argparse import ArgumentParser

from flask import Flask, request, jsonify

from graphite.protocol import GraphProblem, GraphSynapse
from neurons.call_method import (beam_solver_solution, baseline_solution, nns_vali_solver_solution,
                                 hpn_solver_solution, scoring_solution, tsp_annealer_solver, new_solver_solution,
                                 simulated_annealing_solver, or_solver_solution)


def parse():
    parser = ArgumentParser()
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on.')
    parser.add_argument('--method', type=str, default='', help='Resolver to generate route')
    return parser.parse_args()


args = parse()
app = Flask(__name__)


def run_resolver(method, synapse_request):
    if method == 'BEAM':
        return asyncio.run(beam_solver_solution(synapse_request))
    elif method == 'BASE_LINE':
        return asyncio.run(baseline_solution(synapse_request))
    elif method == 'NNS_VALI':
        return asyncio.run(nns_vali_solver_solution(synapse_request))
    elif method == 'HPN':
        return asyncio.run(hpn_solver_solution(synapse_request))
    elif method == 'LEAF_SA':
        return asyncio.run(tsp_annealer_solver(synapse_request))
    elif method == 'JACKIE_SA':
        return asyncio.run(simulated_annealing_solver(synapse_request))
    elif method == 'NEW':
        return asyncio.run(new_solver_solution(synapse_request))
    elif method == 'OR':
        return asyncio.run(or_solver_solution(synapse_request))
    else:
        print(f"method not in accept list")
        resolver = None

    return resolver


@app.route("/")
def hello_world():
    return "Hello! I am resolver service"


@app.route('/resolve', methods=['POST'])
def register():
    start_time = time.time_ns()
    if request.is_json:
        data = request.get_json()
        problem = data['problem']
        graph_problem = GraphProblem.parse_obj(problem)
        synapse = run_resolver(args.method, GraphSynapse(problem=graph_problem))
        print(f"synapse = {synapse}")
        score = scoring_solution(synapse)
        print(f"score = {score}")
        print(f"time loading {int(time.time_ns() - start_time):,} nanosecond")
        return jsonify({"message": "SUCCESS", "result":  synapse.solution, "score": score}), 200
    else:
        return jsonify({"error": "Request must be JSON"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=args.port)
