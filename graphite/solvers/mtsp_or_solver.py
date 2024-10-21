import traceback
from typing import List
from typing import Union

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from graphite.protocol import GraphV2Problem, GraphV2ProblemMulti
from graphite.solvers.base_solver import BaseSolver


class MTSP_ORToolsSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphV2Problem] = [GraphV2ProblemMulti()]):
        super().__init__(problem_types=problem_types)

    async def solve(self, problem, future_id: int) -> List[int]:
        try:

            print(f'start solve. problem in mtsp_or_solver')
            data = self.create_data_model(distance_matrix=problem.edges,num_vehicles=problem.n_salesmen,depot=0)
            manager = pywrapcp.RoutingIndexManager(
                len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
            )
            routing = pywrapcp.RoutingModel(manager)

            # Create and register a transit callback.
            def distance_callback(from_index, to_index):
                """Returns the distance between the two nodes."""
                # Convert from routing variable Index to distance matrix NodeIndex.
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return data["distance_matrix"][from_node][to_node]

            transit_callback_index = routing.RegisterTransitCallback(distance_callback)

            # Define cost of each arc.
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

            # Add Distance constraint.
            dimension_name = "Distance"
            routing.AddDimension(
                transit_callback_index,
                0,  # no slack
                3000,  # vehicle maximum travel distance
                True,  # start cumul to zero
                dimension_name,
            )

            distance_dimension = routing.GetDimensionOrDie(dimension_name)
            distance_dimension.SetGlobalSpanCostCoefficient(100)
            # Setting first solution heuristic.
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )
            # Solve the problem.
            solution = routing.SolveWithParameters(search_parameters)

            # Print solution on console.
            if solution:
                self.print_solution(data, manager, routing, solution)
            else:
                print("No solution found !")

        except Exception as e:
            print(f'Exception {e}')
            traceback.print_exc()
        return []

    def problem_transformations(self, problem: Union[GraphV2ProblemMulti]):
        return problem.edges


    def create_data_model(self,distance_matrix, num_vehicles, depot=0):
        data = {}
        data["distance_matrix"] = distance_matrix
        data["num_vehicles"] = num_vehicles
        data["depot"] = depot
        return data

    def print_solution(self, data, manager, routing, solution):
        """Prints solution on console."""
        print(f"Objective: {solution.ObjectiveValue()}")
        max_route_distance = 0
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            plan_output = f"Route for vehicle {vehicle_id}:\n"
            route_distance = 0
            while not routing.IsEnd(index):
                plan_output += f" {manager.IndexToNode(index)} -> "
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            plan_output += f"{manager.IndexToNode(index)}\n"
            plan_output += f"Distance of the route: {route_distance}m\n"
            print(plan_output)
            max_route_distance = max(route_distance, max_route_distance)
        print(f"Maximum of the route distances: {max_route_distance}m")


if __name__ == '__main__':
    ...
