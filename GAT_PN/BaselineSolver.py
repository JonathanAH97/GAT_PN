#Create a class for the baseline solver, used for other solvers to inherit from
import numpy as np
import matplotlib.pyplot as plt
import tsplib95
import os
import networkx as nx
import torch
import random
from scipy.spatial import distance_matrix
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from networkx import approximation as nx_app
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing, solve_tsp_local_search
import sys
from scipy.spatial import distance
from GAT_PN import *
import time

class BaselineTSP(object):
    def __init__(self, TSP_instance=None, TSPLIB=None):
        self.method = None
        self.solution = None
        self.solution_cost = None
        self.solution_time = None
        self.solution_order = None
        self.optimal_solution_order = None
        self.optimal_solution = None
        self.problem_name = None
        self.problem_size = None
        self.problem_description = None
        self.optimal_solution_cost = None
        if TSPLIB is not None:
            file_path = "../TSP/TSPData/SYMTSP/" + TSPLIB + ".tsp"
            TSP_DICT = tsplib95.load(file_path).as_keyword_dict()
            self.TSP_instance = TSP_DICT['NODE_COORD_SECTION']
            self.TSP_instance = np.array(list(self.TSP_instance.values()))
            self.TSP_instance = self.TSP_instance.astype(np.float32)
            self.problem_name = TSP_DICT['NAME']
            self.problem_size = TSP_DICT['DIMENSION']
            self.problem_description = TSP_DICT['COMMENT']

            #If optimal solution is provided, load it
            opt_file_path = "../TSP/TSPData/SYMTSP/" + TSPLIB + ".opt.tour"
            if os.path.exists(opt_file_path):
                OPT_DICT = tsplib95.load(opt_file_path).as_keyword_dict()
                self.optimal_solution_order = np.array(OPT_DICT['TOUR_SECTION'])[0]
                self.optimal_solution_order = np.concatenate((self.optimal_solution_order, [self.optimal_solution_order[0]]))
                #Convert the solution order to the euclidean coordinates
                self.optimal_solution = self.TSP_instance[self.optimal_solution_order-1]
            else:
                print("No optimal solution provided")

        elif TSP_instance is not None:
            self.TSP_instance = TSP_instance
        else:
            print("No instance provided")

        self.dist_matrix = distance_matrix(self.TSP_instance, self.TSP_instance)
    
    def get_problem_info(self):
        problem_info = {}
        problem_info['problem_name'] = self.problem_name
        problem_info['problem_size'] = self.problem_size
        problem_info['problem_description'] = self.problem_description
        return problem_info

    def get_solution(self):
        return self.solution
    
    def get_optimal_solution(self):
        return self.optimal_solution

    def opt_solution_gap(self):
        if self.optimal_solution is None:
            print("No optimal solution provided")
            return
        else:
            self.optimal_solution_cost = self.get_solution_cost(opt=True)
            self.solution_cost = self.get_solution_cost()
            return self.solution_cost/self.optimal_solution_cost


    def get_solution_cost(self, opt=False):
        if opt:
            solution = self.optimal_solution
        else:
            solution = self.solution
        tour = np.concatenate((solution, np.expand_dims(solution[0],0))) # sequence to tour (end=start)
        inter_city_distances = np.sqrt(np.sum(np.square(tour[:-1,:2]-tour[1:,:2]),axis=1)) # tour length
        solution_cost = np.sum(inter_city_distances) # reward
        return solution_cost
    
    def get_permutation_cost(self, permutation):
        solution = self.TSP_instance[permutation]
        tour = np.concatenate((solution, np.expand_dims(solution[0],0)))
        inter_city_distances = np.sqrt(np.sum(np.square(tour[:-1,:2]-tour[1:,:2]),axis=1))
        solution_cost = np.sum(inter_city_distances)
        return solution_cost
    
    
    def two_opt(self):

        def calculate_tour_distance(tour):
            distance = 0
            num_cities = len(tour)

            for i in range(num_cities - 1):
                city1 = tour[i]
                city2 = tour[i + 1]
                distance += self.dist_matrix[city1][city2]

            # Add distance from last city back to the first city
            distance += self.dist_matrix[tour[num_cities - 1]][tour[0]]

            return distance


        # Initialize the best tour
        tour = self.solution_order
        best_tour = self.solution_order
        best_distance = calculate_tour_distance(best_tour)

        # Flag to indicate if any improvement is made
        improvement = True

        while improvement:
            improvement = False

            for i in range(1, len(tour) - 2):
                for j in range(i + 1, len(tour)):
                    if j - i == 1:
                        continue

                    new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                    new_distance = calculate_tour_distance(new_tour)

                    if new_distance < best_distance:
                        best_tour = new_tour
                        best_distance = new_distance
                        improvement = True

            tour = best_tour
        self.solution_order = best_tour
        self.solution = self.TSP_instance[self.solution_order]

    def plot_solution(self, opt=False, show_plot = True):
        if opt:
            solution = self.optimal_solution
        else:
            solution = self.solution

        assert len(solution) == len(self.TSP_instance) + 1

        if solution is None:
            print("No solution to plot")
            return
        else:
            plt.figure(1)
            colors = ['red'] # First and last city in red (Tour start and end)
            for i in range(len(solution)-2):
                colors.append('blue')
            colors.append('red')
                
            plt.scatter(solution[:,0], solution[:,1],  color=colors) # Plot cities
            tour=np.array(list(range(len(solution))) + [0]) # Plot tour
            X = solution[tour, 0]
            Y = solution[tour, 1]
            plt.plot(X, Y,"--")

            x_min = np.min(solution[:, 0])
            x_max = np.max(solution[:, 0])
            y_min = np.min(solution[:, 1])
            y_max = np.max(solution[:, 1])

            # Add some padding to the limits
            padding = 0.1
            x_padding = (x_max - x_min) * padding
            y_padding = (y_max - y_min) * padding

            # Set the plot limits based on the calculated values
            plt.xlim(x_min - x_padding, x_max + x_padding)
            plt.ylim(y_min - y_padding, y_max + y_padding)
            plt.xlabel('X')
            plt.ylabel('Y')
            if opt:
                plt.title("Optimal solution")
            else:
                plt.title(self.method)
            
            solution_cost = self.get_solution_cost(opt=opt)
            computation_time = self.solution_time
            main_legend = plt.legend([f"Len: {solution_cost:.3f}"], loc='upper left')
            if not opt:
                time_legend = plt.legend([f"Time: {computation_time:.3f}"], loc='upper right')
                plt.gca().add_artist(main_legend)
            if show_plot == True:
                plt.show()

class TSP_Christofides(BaselineTSP):
    def __init__(self, TSP_instance=None, TSPLIB=None):
        super().__init__(TSP_instance=TSP_instance, TSPLIB=TSPLIB)
        self.method = "Christofides"
  
    def solve(self):
        dist_matrix = distance_matrix(self.TSP_instance, self.TSP_instance)
        G = nx.Graph()
        num_points = len(self.TSP_instance)
        for i in range(num_points):
            for j in range(i + 1, num_points):
                G.add_edge(i, j, weight=dist_matrix[i, j])
        start_time = time.time()
        christofides_order = np.array(nx_app.christofides(G, weight="weight"))
        self.solution_time = time.time() - start_time 
        self.solution = self.TSP_instance[christofides_order]

    def get_computation_time(self):
        return self.solution_time
    


class TSP_SA(BaselineTSP):
    def __init__(self, TSP_instance=None, TSPLIB=None):
        super().__init__(TSP_instance=TSP_instance, TSPLIB=TSPLIB)
        self.method = "Simulated Annealing"
    
    def solve(self, LS = False):
        dist_matrix = distance_matrix(self.TSP_instance, self.TSP_instance)
        start_time = time.time()
        SA_Sol, SA_Dist = solve_tsp_simulated_annealing(dist_matrix)
        if LS:
            SA_Sol, LS_Dist = solve_tsp_local_search(
                                                     dist_matrix, x0 = SA_Sol, perturbation_scheme='ps3')
        SA_Sol.append(SA_Sol[0])
        self.solution_time = time.time() - start_time
        self.solution = self.TSP_instance[SA_Sol]
        
    def get_computation_time(self):
        return self.solution_time

class TSP_RI(BaselineTSP):
    def __init__(self, TSP_instance=None, TSPLIB=None, iters = 1000):
        super().__init__(TSP_instance=TSP_instance, TSPLIB=TSPLIB)
        self.method = "Random Insertion"
        self.iters = iters
    
    def solve(self):
        start_time = time.time()
        num_cities = len(self.TSP_instance)
        best_tour = list(range(num_cities))  # Initialize with a baseline tour
        
        # Calculate the total distance of the initial tour
        best_distance = self.get_permutation_cost(best_tour)
        
        for _ in range(self.iters):
            tour = best_tour[:]
            city = random.choice(range(num_cities))
            pos1, pos2 = random.sample(range(num_cities), 2)
            pos1, pos2 = min(pos1, pos2), max(pos1, pos2)
            tour.insert(pos1, city)
            tour.insert(pos2, city)
            distance = self.get_permutation_cost(tour)
            
            if distance < best_distance:
                best_tour = tour
                best_distance = distance
        
        self.solution = self.TSP_instance[best_tour]
        self.solution_time = time.time() - start_time

    def get_computation_time(self):
        return self.solution_time


class TSP_GAT(BaselineTSP):
    def __init__(self, GAT_path, TSP_instance=None, TSPLIB=None):
        super().__init__(TSP_instance=TSP_instance, TSPLIB=TSPLIB)
        model = torch.load(GAT_path)
        self.model = model

        self.method = "GAT-PN"
    
    def solve(self, search = 'greedy', two_opt = True):
        model = self.model
        model.eval()
        start_time = time.time()
        n_nodes = len(self.TSP_instance)
        mask = torch.zeros(1, n_nodes)

        Y = torch.from_numpy(self.TSP_instance).float().view(1, n_nodes, 2)
        x = Y[:, 0, :]
        h = None
        c = None

        GPN_Sol = []
        GPN_idx = []
        for k in range(len(self.TSP_instance)):
            
            output, h, c, _ = model(x=x, X_all=Y, h=h, c=c, mask=mask)
            
            if search == 'greedy':
                idx = torch.argmax(output, dim = 1)
            elif search == 'sampling':
                sampler = torch.distributions.Categorical(output)
                idx = sampler.sample()
            
            GPN_idx.append(idx.data.item())
            x = Y[[0], idx.data]
            GPN_Sol.append(x.cpu().numpy())

            mask[[0], idx.data] += -np.inf

        GPN_Sol.append(GPN_Sol[0])
        GPN_Sol = np.array(GPN_Sol).squeeze()
        self.solution = GPN_Sol
        GPN_idx.append(GPN_idx[0])
        self.solution_order = GPN_idx
        if two_opt:
            self.two_opt()
        self.solution_time = time.time() - start_time

    def get_computation_time(self):
        return self.solution_time


class TSP_ORTools(BaselineTSP):
    def __init__(self, time_limit = 5, TSP_instance=None, TSPLIB=None):
        super().__init__(TSP_instance=TSP_instance, TSPLIB=TSPLIB)
        self.method = "OR-Tools"
        self.time_limit = time_limit
    
    def create_data_model(self, graph, vehicles, depot):
            data = dict()
            dist_matrix = (distance.cdist(graph, graph, 'euclidean') * (10 ** 9)).tolist()
            data['distance_matrix'] = [[int(entry) for entry in row] for row in dist_matrix]
            data['num_vehicles'] = vehicles
            data['depot'] = depot
            return data
    

    def solve(self, first_solution_strategy = 'Christofides'):
        start_time = time.time()
        data = self.create_data_model(self.TSP_instance, 1, 0)
        # Create the routing index manager and the routing model
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)
        
        # Define the distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        def get_routes(solution, routing, manager):
            """Get vehicle routes from a solution and store them in an array."""
            # Get vehicle routes and store them in a two dimensional array whose
            # i,j entry is the jth location visited by vehicle i along its route.
            routes = []
            for route_nbr in range(routing.vehicles()):
                index = routing.Start(route_nbr)
                route = [manager.IndexToNode(index)]
                while not routing.IsEnd(index):
                    index = solution.Value(routing.NextVar(index))
                    route.append(manager.IndexToNode(index))
                routes.append(route)
            return routes

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        # Set the solver parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        if first_solution_strategy == 'Christofides':
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES)
        else:
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.SAVINGS)
            
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = self.time_limit  # Set a time limit for the solver

        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)
        # Get the best tour
        if solution:
            routes = get_routes(solution, routing, manager)
        
        self.solution_order = routes[0]
        self.solution = self.TSP_instance[self.solution_order]
        self.solution_time = time.time() - start_time

    def get_computation_time(self):
        return self.solution_time
    

class TSP_DP(BaselineTSP):
    def __init__(self, TSP_instance=None, TSPLIB=None):
        super().__init__(TSP_instance=TSP_instance, TSPLIB=TSPLIB)
        self.method = "Held Karp"

    def solve(self):
        dist_matrix = distance_matrix(self.TSP_instance, self.TSP_instance)
        start_time = time.time()
        HK_Sol, HK_Dist = solve_tsp_dynamic_programming(dist_matrix)
        HK_Sol.append(HK_Sol[0])
        self.solution_time = time.time() - start_time
        self.solution = self.TSP_instance[HK_Sol]
        
    def get_computation_time(self):
        return self.solution_time