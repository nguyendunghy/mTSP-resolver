import time
from argparse import ArgumentParser
from graphite.protocol import GraphV2Problem
from graphite.solvers.lkh_solver import LKHSolver
import fastapi
from fastapi import status
from fastapi.responses import JSONResponse
from typing import Dict
app = fastapi.FastAPI()


def parse():
    parser = ArgumentParser()
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on.')
    parser.add_argument('--num_run', type=str, default='', help='Number of running')
    parser.add_argument('--max_trial', type=str, default='', help='Maximum of trial')
    return parser.parse_args()


args = parse()


@app.get("/")
def hello_world():
    return "Hello! I am resolver service"


@app.post('/lkh_resolve')
async def lkh_resolve(data: Dict):
    start_time = time.time_ns()
    input_file = data['input_file_path']
    n_nodes = data['n_nodes']
    dataset_ref = data['dataset_ref']
    timeout = data['timeout']

    lkh_solver = LKHSolver(num_run=args.num_run, max_trial=args.max_trial, input_file=input_file)
    problem = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=[0], cost_function="Geom",
                             dataset_ref=dataset_ref, directed=False)
    solution = lkh_solver.solve_problem(problem,timeout)

    print(f"time loading {int(time.time_ns() - start_time):,} nanosecond")
    return {
        "message": "SUCCESS",
        "result": solution,
        "score": 0
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
