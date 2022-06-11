import json
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from numpy import int64

from climbing import ClimbSearch
from algorithms.neighbourhood import get_neighbours_insert, LK_swap_neighbours, get_neighbours_to_climb, \
    get_neighbours_swap
# from notebooks.algorithms.utils import read_data
from search import Search
from algorithms.annealing import SimmulatedAnnealingSearch
import pandas as pd


# from genetic import GeneticSearch, Selection, Crossover


def run_grid_search(config, json_path, nr_tries=2):
    with open(json_path, 'w') as f:
        f.write('[]')

    for algorithm in config:
        algorithm_type = algorithm['algorithm']
        grid = product(*[v for v in algorithm['params'].values()])

        for combination in grid:

            best_result = 10000000
            search_summary = None
            for r in range(nr_tries):
                search = algorithm_type(*combination)
                result, plot = search.run()
                if result['best cost'] < best_result:
                    best_result = result['best cost']
                    search_summary = search.__dict__

                    try:
                        search_summary['selection_method'] = str(search_summary['selection_method'])
                        # print(search_summary['cx_method'])
                        del search_summary['cx_method'] #= str(search_summary['cx_method'].__str__())
                        search_summary['_best_fintess'] = int(search_summary['_best_fintess'])
                        search_summary['_generation'] = int(search_summary['_generation'])
                        del search_summary['_cx_mapping']
                        del search_summary['_selection_mapping']
                        del search_summary['history']
                        del search_summary['_best_genotype']
                    except KeyError:
                        pass
                    print(search_summary)
                    search_summary['result'] = result
                    search_summary['plot'] = plot
                    del search_summary['data']
                    try:
                        search_summary['get_neighbours'] = str(search_summary['get_neighbours'])
                    except KeyError:
                        search_summary['get_neighbours'] = 'NO nGBH'
                    search_summary['algorithm'] = str(search)
                    search_summary['data'] = algorithm['dataset_name']
            with open(json_path, 'r') as f:
                grid_search_history = json.load(f)
            grid_search_history.append(search_summary)
            grid_search_history.sort(key=lambda x: x['result']['best cost'])
            with open(json_path, 'w') as f:
                json.dump(grid_search_history, f)
    return grid_search_history


if __name__ == '__main__':
    config = [
        # {'algorithm': SimmulatedAnnealingSearch,
        #  'dataset_name': 'Dane_TSP_48.xlsx',
        #  'params': {
        #      'data': [pd.read_excel('../data/Dane_TSP_48.xlsx')],
        #      'decreasing_factor': [0.99, 0.9],
        #      'max_same_iters': [20, 100, 200],
        #      'temperature_min': [0.1],
        #      'first_nr_of_neighbors': [20, 50, 100, 200],
        #      'temperature': [1, 0.9, 0.8],
        #      'get_neighbours': [get_neighbours_swap, LK_swap_neighbours, get_neighbours_insert]
        #  }
        #  }
        # ,
        # {'algorithm': GeneticSearch,
        #  'dataset_name': 'Dane_S2_100_20.xlsx',
        #  'params': {
        #      'data': [read_data('../../data/Dane_S2_100_20.xlsx')],
        #      'selection_method': [Selection.TOUR, Selection.RANK, Selection.ROULETTE],
        #      'cx_method': [Crossover.MKX, Crossover.OX],
        #      'generations': [100, 500,100],
        #      'pop_size': [10],
        #      'mutation_threshold': [0.01],
        #      'crossover_mask_size': [20],
        #      'mut_mask_size': [3]
        #  }
        #  }
        # ,
        # {'algorithm': ClimbSearch,
        #  'dataset_name': 'Dane_TSP_76.xlsx',
        #  'params': {
        #      'data': [pd.read_excel('../data/Dane_TSP_76.xlsx')],
        #      'nr_of_neighbours': [20,50,100,400,500,1000],
        #      'get_neighbours': [get_neighbours_to_climb,get_neighbours_swap, get_neighbours_insert, LK_swap_neighbours],
        #      'nr_of_same_costs': [20, 50, 100, 800]
        #  }
        #  },
        {'algorithm': SimmulatedAnnealingSearch,
         'dataset_name': 'Dane_TSP_127.xlsx',
         'params': {
             'data': [pd.read_excel('../data/Dane_TSP_127.xlsx')],
             'decreasing_factor': [0.99, 0.997, 0.9997],
             'max_same_iters': [20, 50, 100, 200],
             'temperature_min': [0.05],
             'first_nr_of_neighbors': [200, 500, 600, 800],
             'temperature': [1, 0.99, 0.9, 0.8],
             'get_neighbours': [get_neighbours_swap, LK_swap_neighbours, get_neighbours_insert, get_neighbours_to_climb]
         }
         }

    ]



    """
    Poniżej immplementacja wspinaczki z mulitstartem (iteracyjnej) 
    -czyli puszczamy config dla wspinaczki number_of_multistarts_climbing razy
    """
    filename = 'results/SA/results_SA_big2.json'


    results_climb = run_grid_search(config, filename, nr_tries=1)
    print(results_climb)


# for experiment in results_climb[:10]:
#     try:
#         fig, ax = plt.subplots(len(experiment['plot'].keys()), figsize=(5, 10))
#         for i, (parameter_name, parameter) in enumerate(experiment['plot'].items()):
#             if parameter_name == 'step length':
#                 ax[i].set_title('diagram Gantta')
#                 t_before = 0
#                 t_after = 0
#
#                 for idx, steps in enumerate(parameter):
#                     t_after = t_before + steps
#                     ax[i].broken_barh([(t_before, steps)], (idx * -2, 2), facecolors=('tab:red'))
#                     t_before = t_after
#                 # Setting labels for x-axis and y-axis
#                 ax[i].set_xlabel('epochs(neighbourhood)')
#                 ax[i].set_ylabel('temp')
#             else:
#                 ax[i].plot(parameter)
#                 ax[i].set_title(f"{parameter_name} ")
#         fig.show()
#     except ValueError:
#         pass
