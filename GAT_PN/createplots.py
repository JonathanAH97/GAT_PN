import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from DataGenerator import DataGenerator
from BaselineSolver import *
plt.style.use('bmh')

def load_val_data(n_nodes, baseline, vc):
    path = 'VAL_COSTS/GPN_' + str(n_nodes) + '_' + str(baseline) + '_VC-' + str(vc) + '_GE-GAT_val_costs' + '.pkl'
    print(path)
    with open(path, 'rb') as f:
        val_costs = pickle.load(f)
    return val_costs

def load_tsp_data(n_nodes):
    path = 'RESULTS/TSP' + str(n_nodes) +'.pkl'
    with open(path, 'rb') as f:
        tsp_data = pickle.load(f)
    return tsp_data

def plot_validation(n_nodes, vc):
    gat_n_exp = load_val_data(n_nodes, 'exponential', vc)
    gat_n_critic = load_val_data(n_nodes, 'critic', vc)
    exp_val = np.stack([tensor.numpy() for tensor in gat_n_exp[:-1]])
    critic_val = np.stack([tensor.numpy() for tensor in gat_n_critic[:-1]])
    exp_time = gat_n_exp[-1][0]
    critic_time = gat_n_critic[-1][0]
    x_exp = np.arange(1, exp_val.shape[0] + 1)
    x_critic = np.arange(1, critic_val.shape[0] + 1)
    print(f'Exp time: {exp_time}, critic time: {critic_time}')
    plt.figure(figsize=(7, 5))
    plt.plot(x_exp, exp_val, label='Exponential baseline')
    plt.plot(x_critic, critic_val, label='Critic baseline')
    plt.xlabel('Epoch')
    plt.ylabel('Average tour length')
    plt.title('Validation cost for GAT-PN{}'.format(n_nodes))
    main_legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')
    
    # Add secondary legend for training time
    time_legend = plt.legend(['Time: {:.0f}s'.format(exp_time), 'Time: {:.0f}s'.format(critic_time)],
                             loc='center right', shadow = True, fontsize='large', bbox_to_anchor=(1, 0.70))
    plt.gca().add_artist(main_legend)
    plt.show()

def plot_tsp_sol(tsp_data):
    
    cost_keys = ['SA', 'GAT20_GR', 'GAT50_GR', 'GAT75_GR', 'OR_CHR', 'OR_SAV', 'CHR']
    time_keys = ['GAT20_GR', 'GAT50_GR', 'GAT75_GR', 'CHR']
    
    plt.figure(figsize=(8, 6))
    plt.boxplot([tsp_data[key]['cost'] for key in cost_keys], labels=['SA', 'GAT-PN20', 'GAT-PN50', 'GAT-PN75', 'OR-CHR', 'OR-SAV', 'CHR'])
    plt.ylabel('Solution Cost')
    plt.xlabel('Solver')
    plt.title('TSP100')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.boxplot([tsp_data[key]['time'] for key in time_keys], labels=['GAT-PN20', 'GAT-PN50', 'GAT-PN75','CHR'])
    plt.ylabel('Solution Time')
    plt.xlabel('Solver')
    plt.show()

def plot_tsp_sol_single(n_nodes, GAT):
    DG = DataGenerator()
    tsp_inst = DG.gen_instance(n_nodes, 2)
    GATPATH = 'RL_Models/GPN_{}_critic_VC-1_GE-GAT.pt'.format(GAT)

    GAT = TSP_GAT(GATPATH,TSP_instance = tsp_inst)
    GAT.solve(two_opt = True)
    GAT.plot_solution()

    CHR = TSP_Christofides(TSP_instance = tsp_inst)
    CHR.solve()
    CHR.plot_solution()

    SA = TSP_SA(TSP_instance = tsp_inst)
    SA.solve()
    SA.plot_solution()

    ORT = TSP_ORTOOLS(TSP_instance = tsp_inst)
    ORT.solve(first_solution_strategy = 'SAVINGS')
    ORT.plot_solution()


def plot_solution2(solution, cost, time, opt, method):
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
        plt.title(method)
    
    solution_cost = cost
    main_legend = plt.legend([f"Len: {solution_cost:.3f}"], loc='upper left')
    if not opt:
        time_legend = plt.legend([f"Time: {time:.3f}"], loc='upper right')
        plt.gca().add_artist(main_legend)
    plt.show()



def summarize_tsp(n_nodes):
    tsp_data = load_tsp_data(n_nodes)
    result_dict = {}
    for key in tsp_data:
        result_dict[key] = {}
        result_dict[key]['avg_cost'] = round(np.mean(tsp_data[key]['cost']),3)
        result_dict[key]['avg_time'] = round(np.mean(tsp_data[key]['time']),3)
        result_dict[key]['std_cost'] = round(np.std(tsp_data[key]['cost']), 3)
        result_dict[key]['std_time'] = round(np.std(tsp_data[key]['time']), 3)
    return result_dict


def plot_solution(solution, cost, time, opt, method, ax=None):
    if ax is None:
        ax = plt.gca()

    colors = ['red'] # First and last city in red (Tour start and end)
    for i in range(len(solution)-2):
        colors.append('blue')
    colors.append('red')
        
    ax.scatter(solution[:,0], solution[:,1], color=colors) # Plot cities
    tour=np.array(list(range(len(solution))) + [0]) # Plot tour
    X = solution[tour, 0]
    Y = solution[tour, 1]
    ax.plot(X, Y, "--")

    x_min = np.min(solution[:, 0])
    x_max = np.max(solution[:, 0])
    y_min = np.min(solution[:, 1])
    y_max = np.max(solution[:, 1])

    # Add some padding to the limits
    padding = 0.1
    x_padding = (x_max - x_min) * padding
    y_padding = (y_max - y_min) * padding

    # Set the plot limits based on the calculated values
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if opt:
        ax.set_title("Optimal solution")
    else:
        ax.set_title(method)
    
    solution_cost = cost
    main_legend = ax.legend([f"Len: {solution_cost:.3f}"], loc='upper left')
    if not opt:
        time_legend = ax.legend([f"Time: {time:.3f}"], loc='upper right')
        ax.add_artist(main_legend)


def combine_plot_solutions(solution_list, problem):
    fig, axes = plt.subplots(2, 2, figsize = (12,8))

    for i, (solution, cost, time, opt, method) in enumerate(solution_list):
        ax = axes[i // 2, i % 2]
        plot_solution(solution, cost, time, opt, method, ax)

    # Remove the tick labels for all subplots
    for ax in axes.flatten():
        ax.set_aspect('auto')
        ax.autoscale()

    plt.suptitle(problem, fontsize=18, fontweight='bold', y=0.95,
                 )
    plt.tight_layout()
    plt.savefig(f"Results/{problem}.png")
    # Display the combined image
    plt.clf()






def tsplib_plot(problem, gat = 75):
    GATPATH = 'RL_Models/GPN_{}_critic_VC-1_GE-GAT.pt'.format(gat)

    GAT = TSP_GAT(GATPATH, TSPLIB = problem)
    CHR = TSP_Christofides(TSPLIB = problem)
    SA = TSP_SA(TSPLIB = problem)
    SA.solve()
    GAT.solve()
    CHR.solve()

    solution1 = (SA.solution, SA.get_solution_cost(), SA.get_computation_time(), False, 'SA')
    solution2 = (GAT.solution, GAT.get_solution_cost(), GAT.get_computation_time(), False, 'GAT-PN')
    solution3 = (CHR.solution, CHR.get_solution_cost(), CHR.get_computation_time(), False, 'Christofides')
    solution4 = (CHR.optimal_solution, CHR.get_solution_cost(opt = True), CHR.get_computation_time(), True, 'Optimal')
    combine_plot_solutions([solution1, solution2, solution3, solution4], problem)

def get_opt_gap():
    with open('RESULTS/TSPLIB.pkl', 'rb') as f:
        tsplib_res = pickle.load(f)
    
    gap_dict = {}
    for instance, method in tsplib_res.items():
        CHR = TSP_Christofides(TSPLIB = instance)
        opt_cost = CHR.get_solution_cost(opt = True)
        print(instance, opt_cost)
        for method_name, method_res in method.items():
            gap_dict[(instance, method_name)] = round((method_res['cost'] - opt_cost) / opt_cost, 3)
    return gap_dict

