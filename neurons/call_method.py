import copy

from graphite.solvers import DPSolver, NearestNeighbourSolver, BeamSearchSolver, HPNSolver
from graphite.solvers.TSPAnnealer import TSPAnnealer
from graphite.solvers.greedy_solver_vali import NearestNeighbourSolverVali
from graphite.solvers.new_solver import NewSearchSolver
from graphite.solvers.or_solver import ORToolsSolver
from graphite.solvers.simulated_annealing_solver import SimulatedAnnealingSolver
from graphite.validator.reward import ScoreResponse
from graphite.solvers.lkh_solver import LKHSolver

solvers = {
    'small': DPSolver(),
    'large': NearestNeighbourSolver()
}
beam_solver = BeamSearchSolver()
nearest_neighbour_solver_vali = NearestNeighbourSolverVali()
hpn_solver = HPNSolver()
new_solver = NewSearchSolver()
sa_solver = SimulatedAnnealingSolver()
or_solver = ORToolsSolver()



async def baseline_solution(synapse):
    new_synapse = copy.deepcopy(synapse)
    if new_synapse.problem.n_nodes < 15:
        # Solves the problem to optimality but is very computationally intensive
        route = await solvers['small'].solve_problem(new_synapse.problem)
    else:
        # Simple heuristic that does not guarantee optimality.
        route = await  solvers['large'].solve_problem(new_synapse.problem)
    new_synapse.solution = route
    # print(
    #     f"Miner returned value {synapse.solution}   length =  {len(synapse.solution) if isinstance(synapse.solution, list) else synapse.solution}"
    # )
    return new_synapse


async def beam_solver_solution(synapse):
    new_synapse = copy.deepcopy(synapse)
    route = await  beam_solver.solve_problem(new_synapse.problem)
    new_synapse.solution = route
    return new_synapse


async def nns_vali_solver_solution(synapse):
    new_synapse = copy.deepcopy(synapse)
    route = await  nearest_neighbour_solver_vali.solve_problem(new_synapse.problem)
    new_synapse.solution = route
    return new_synapse


async def hpn_solver_solution(synapse):
    new_synapse = copy.deepcopy(synapse)
    route = await  hpn_solver.solve_problem(new_synapse.problem)
    new_synapse.solution = route
    return new_synapse


def scoring_solution(synapse_req):
    score_response_obj = ScoreResponse(synapse_req)
    miner_scores = score_response_obj.get_score(synapse_req)
    return miner_scores


async def new_solver_solution(synapse):
    new_synapse = copy.deepcopy(synapse)
    route = await  new_solver.solve_problem(new_synapse.problem)
    new_synapse.solution = route
    return new_synapse


async def simulated_annealing_solver(synapse):
    new_synapse = copy.deepcopy(synapse)
    route = await sa_solver.solve_problem(new_synapse.problem)
    new_synapse.solution = route
    return new_synapse


async def or_solver_solution(synapse):
    new_synapse = copy.deepcopy(synapse)
    route = await  or_solver.solve_problem(new_synapse.problem)
    new_synapse.solution = route
    return new_synapse


def build_lkh_input_file(synapse):
    lkh_solver = LKHSolver()
    import random
    random_number = random.randint(10000, 999999)
    problem_filename = f"{random_number}_prebuild_problem.tsp"
    scaled_distance_matrix = lkh_solver.build_scaled_distance_matrix(synapse.problem)
    # scaled_distance_matrix = synapse.problem.edges
    lkh_solver.write_tsplib_file(scaled_distance_matrix, problem_filename, synapse.problem.directed)
    return problem_filename

async def lkh_solver_solution(synapse, num_run=1, input_file=None):
    new_synapse = copy.deepcopy(synapse)
    if num_run == 1:
        lkh_solver = LKHSolver(input_file=input_file)
        route =  await  lkh_solver.solve_problem(new_synapse.problem)
    elif num_run == 3:
        lkh_solver_3 = LKHSolver(num_run=4,max_trial=30,input_file=input_file)
        route = await  lkh_solver_3.solve_problem(new_synapse.problem)
    else:
        lkh_solver_5000 = LKHSolver(num_run=1, max_trial=2,input_file=input_file)
        route = await  lkh_solver_5000.solve_problem(new_synapse.problem)
    new_synapse.solution = route
    return new_synapse

async def tsp_annealer_solver(synapse):
    new_synapse = copy.deepcopy(synapse)
    coords = new_synapse.problem.edges
    init_state = list(range(1, new_synapse.problem.n_nodes))
    init_state.insert(0, 0)  # start from 0
    tsp_annealer = TSPAnnealer(init_state, coords)
    tsp_annealer.steps = 200000
    best_state, best_fitness = tsp_annealer.anneal()
    best_state.append(best_state[0])
    new_synapse.solution = best_state
    return new_synapse

# async def lin_kernighan_solution(synapse):
#     new_synapse = copy.deepcopy(synapse)
#     from python_tsp.heuristics import solve_tsp_lin_kernighan
#     result = solve_tsp_lin_kernighan(new_synapse.problem.edges)
#     new_synapse.solution = result
#     return new_synapse