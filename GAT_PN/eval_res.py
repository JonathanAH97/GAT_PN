import tsplib95
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle
import pandas as pd
from DataGenerator import DataGenerator
from createplots import *
from BaselineSolver import *
DG = DataGenerator()
plt.style.use('bmh')

GAT20PATH = 'RL_Models/GPN_20_critic_VC-1_GE-GAT.pt'
GAT50PATH = 'RL_Models/GPN_50_critic_VC-1_GE-GAT.pt'
GAT75PATH = 'RL_Models/GPN_75_critic_VC-1_GE-GAT.pt'

TSPLIB_INSTANCES = ['berlin52','att48', 
                    'gr96',
                    'ch130','ch150',
                    'eil51','eil76','eil101',
                    'pr76','rd100', 'st70',
                    'kroA100','kroC100',
                    'kroD100','lin105']
                    

def solve_tsplib(TSP_INSTANCES):
    problem_dicts = {}
    
    for problem in TSP_INSTANCES:
        GATPATH = GAT75PATH
        if problem in ['berlin52','eil51', 'att48', 'gr24', 'gr48']:
            GATPATH = GAT50PATH
        print(f'Solving {problem}')
        SA_RES = {'cost':[], 'time':[]}
        #GAT50_GR_RES = {'cost':[], 'time':[]}
        GAT75_GR_RES = {'cost':[], 'time':[]}
        OR_CHR_RES = {'cost':[], 'time':[]}
        OR_SAV_RES = {'cost':[], 'time':[]}
        CHR_RES = {'cost':[], 'time':[]}

        for i in tqdm(range(5)):
            SA = TSP_SA(TSPLIB = problem)
            SA.solve()
            SA_RES['cost'].append(SA.get_solution_cost())
            SA_RES['time'].append(SA.get_computation_time())


            GAT75 = TSP_GAT(GATPATH,TSPLIB = problem)
            GAT75.solve()
            GAT75_GR_RES['cost'].append(GAT75.get_solution_cost())
            GAT75_GR_RES['time'].append(GAT75.get_solution_cost())

            OR = TSP_ORTools(TSPLIB = problem)
            OR.solve('christofides')
            OR_CHR_RES['cost'].append(OR.get_solution_cost())
            OR_CHR_RES['time'].append(OR.get_computation_time())
            OR.solve('savings')
            OR_SAV_RES['cost'].append(OR.get_solution_cost())
            OR_SAV_RES['time'].append(OR.get_computation_time())

            CHR = TSP_Christofides(TSPLIB = problem)
            CHR.solve()
            CHR_RES['cost'].append(CHR.get_solution_cost())
            CHR_RES['time'].append(CHR.get_computation_time())

        #PLOT LAST SOLUTION
        solution1 = (SA.solution, SA.get_solution_cost(), SA.get_computation_time(), False, 'SA')
        solution2 = (GAT75.solution, GAT75.get_solution_cost(), GAT75.get_computation_time(), False, 'GAT-PN')
        solution3 = (CHR.solution, CHR.get_solution_cost(), CHR.get_computation_time(), False, 'Christofides')
        solution4 = (CHR.optimal_solution, CHR.get_solution_cost(opt = True), CHR.get_computation_time(), True, 'Optimal')
        combine_plot_solutions([solution1, solution2, solution3, solution4], problem)
        SA_RES['cost'] = np.mean(SA_RES['cost'])
        SA_RES['time'] = np.mean(SA_RES['time'])
        GAT75_GR_RES['cost'] = np.mean(GAT75_GR_RES['cost'])
        GAT75_GR_RES['time'] = np.mean(GAT75_GR_RES['time'])
        OR_CHR_RES['cost'] = np.mean(OR_CHR_RES['cost'])
        OR_CHR_RES['time'] = np.mean(OR_CHR_RES['time'])
        OR_SAV_RES['cost'] = np.mean(OR_SAV_RES['cost'])
        OR_SAV_RES['time'] = np.mean(OR_SAV_RES['time'])
        CHR_RES['cost'] = np.mean(CHR_RES['cost'])
        CHR_RES['time'] = np.mean(CHR_RES['time'])

        problem_dicts[problem] = {'SA':SA_RES, 'GAT75':GAT75_GR_RES,
                                'OR_CHR':OR_CHR_RES, 'OR_SAV':OR_SAV_RES,
                                'CHR':CHR_RES}

    return problem_dicts
    
def solve_sample(n_iters, node_size, or_limit):
    SA_RES = {'cost':[], 'time':[]}
    GAT20_GR_RES = {'cost':[], 'time':[]}
    GAT50_GR_RES = {'cost':[], 'time':[]}
    GAT75_GR_RES = {'cost':[], 'time':[]}
    GAT20_SA_RES = {'cost':[], 'time':[]}
    GAT50_SA_RES = {'cost':[], 'time':[]}
    GAT75_SA_RES = {'cost':[], 'time':[]}
    OR_CHR_RES = {'cost':[], 'time':[]}
    OR_SAV_RES = {'cost':[], 'time':[]}
    CHR_RES = {'cost':[], 'time':[]}
    for i in tqdm(range(n_iters)):
        tsp_inst = DG.gen_instance(node_size, 2)

        SA = TSP_SA(tsp_inst)
        SA.solve()
        SA_RES['cost'].append(SA.get_solution_cost())
        SA_RES['time'].append(SA.get_computation_time())

        GAT20 = TSP_GAT(GAT20PATH,tsp_inst)
        GAT20.solve()
        GAT20_GR_RES['cost'].append(GAT20.get_solution_cost())
        GAT20_GR_RES['time'].append(GAT20.get_computation_time())
        """ GAT20.solve('sampling')
        GAT20_SA_RES['cost'].append(GAT20.get_solution_cost())
        GAT20_SA_RES['time'].append(GAT20.get_computation_time()) """

        GAT50 = TSP_GAT(GAT50PATH,tsp_inst)
        GAT50.solve()
        GAT50_GR_RES['cost'].append(GAT50.get_solution_cost())
        GAT50_GR_RES['time'].append(GAT50.get_computation_time())
        """ GAT50.solve('sampling')
        GAT50_SA_RES['cost'].append(GAT50.get_solution_cost())
        GAT50_SA_RES['time'].append(GAT50.get_computation_time()) """

        GAT75 = TSP_GAT(GAT75PATH,tsp_inst)
        GAT75.solve()
        GAT75_GR_RES['cost'].append(GAT75.get_solution_cost())
        GAT75_GR_RES['time'].append(GAT75.get_computation_time())
        """ GAT75.solve('sampling')
        GAT75_SA_RES['cost'].append(GAT75.get_solution_cost())
        GAT75_SA_RES['time'].append(GAT75.get_computation_time()) """

        OR = TSP_ORTools(time_limit = or_limit, TSP_instance = tsp_inst)
        OR.solve('christofides')
        OR_CHR_RES['cost'].append(OR.get_solution_cost())
        OR_CHR_RES['time'].append(OR.get_computation_time())
        OR.solve('savings')
        OR_SAV_RES['cost'].append(OR.get_solution_cost())
        OR_SAV_RES['time'].append(OR.get_computation_time())

        CHR = TSP_Christofides(tsp_inst)
        CHR.solve()
        CHR_RES['cost'].append(CHR.get_solution_cost())
        CHR_RES['time'].append(CHR.get_computation_time())

        res_dict = {
            'SA':SA_RES,
            'GAT20_GR':GAT20_GR_RES,
            'GAT50_GR':GAT50_GR_RES,
            'GAT75_GR':GAT75_GR_RES,
            'OR_CHR':OR_CHR_RES,
            'OR_SAV':OR_SAV_RES,
            'CHR':CHR_RES
        }
    return res_dict


if __name__ == '__main__':
    """ TSP100 = solve_sample(1000, 100, or_limit=3)
    with open('RESULTS/TSP100.pkl', 'wb') as f:
        pickle.dump(TSP100, f) 
    
    TSPLIB = solve_tsplib(TSPLIB_INSTANCES)
    with open('RESULTS/TSPLIB.pkl', 'wb') as f:
        pickle.dump(TSPLIB, f) """