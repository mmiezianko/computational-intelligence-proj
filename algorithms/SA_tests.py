import json
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from numpy import int64

from algorithms.climbing import ClimbSearch
from algorithms.neighbourhood import get_neighbours_insert, LK_swap_neighbours, get_neighbours_to_climb, get_neighbours_swap
# from notebooks.algorithms.utils import read_data
from search import Search
from algorithms.annealing import SimmulatedAnnealingSearch
import pandas as pd
# from genetic import GeneticSearch, Selection, Crossover


def run_grid_search(config):
    grid_search_history = []
    for algorithm in config:
        algorithm_type = algorithm['algorithm']
        grid = product(*[v for v in algorithm['params'].values()])
        for combination in grid:
            search: Search = algorithm_type(*combination)
            result, plot = search.run()
            search_summary = search.__dict__

            try:
                search_summary['selection_method'] = str(search_summary['selection_method'])
                search_summary['cx_method'] = str(search_summary['cx_method'])
                search_summary['_best_fintess'] = int(search_summary['_best_fintess'])
                search_summary['_generation'] = int(search_summary['_generation'])
                del search_summary['_cx_mapping']
                del search_summary['_selection_mapping']
                del search_summary['history']
                del search_summary['_best_genotype']
            except KeyError:
                pass
            search_summary['result'] = result
            search_summary['plot'] = plot
            del search_summary['data']
            try:
                search_summary['get_neighbours'] = str(search_summary['get_neighbours'])
            except KeyError:
                search_summary['get_neighbours'] = 'NO nGBH'
            search_summary['algorithm'] = str(search)
            search_summary['data'] = algorithm['dataset_name']
            grid_search_history.append(search_summary)
    grid_search_history.sort(key=lambda x: x['result']['best cost'])
    return grid_search_history



if __name__ == '__main__':

    config = [
        {'algorithm': SimmulatedAnnealingSearch,
         'dataset_name': 'Dane_TSP_127.xlsx',
         'params': {
             'data': [pd.read_excel('../data/Dane_TSP_127.xlsx')],
             'decreasing_factor': [0.99, 0.997, 0.9997],
             'max_same_iters': [20, 100, 200,400],
             'temperature_min': [0.05],
             'first_nr_of_neighbors': [200, 400, 500, 600],
             'temperature': [1, 0.99, 0.9, 0.8],
             'get_neighbours': [get_neighbours_swap, LK_swap_neighbours, get_neighbours_insert, get_neighbours_to_climb]
         }
         }
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
        #   'dataset_name': 'Dane_TSP_48.xlsx',
        #   'params': {
        #       'data': [pd.read_excel('../data/Dane_TSP_48.xlsx')],
        #       'nr_of_neighbours': [20,50, 100, 400, 500,1000],
        #       'get_neighbours': [get_neighbours_to_climb, get_neighbours_swap, get_neighbours_insert],
        #       'nr_of_same_costs': [20, 50, 100,800]
        #  }
        #  }

    ]

    # results_SA = run_grid_search(config)
    # print(results_SA)
    # json.dump(results_SA, open('results/SA/results_SA_big.json', 'w'))

    with open('results/SA/results_SA_small3.json') as f:
        results_SA = json.load(f)


    print(results_SA)
    for experiment in results_SA[:10]:
        try:
            fig, ax = plt.subplots(len(experiment['plot'].keys()), figsize=(5, 10))
            for i, (parameter_name, parameter) in enumerate(experiment['plot'].items()):
                if parameter_name == 'step length':
                    ax[i].set_title('Spadek temperatury')
                    t_before = 0
                    t_after = 0

                    for idx, steps in enumerate(parameter):
                        t_after = t_before + steps
                        ax[i].broken_barh([(t_before, steps)], (idx * -2, 2), facecolors=('tab:red'))
                        t_before = t_after
                    # Setting labels for x-axis and y-axis
                    ax[i].set_xlabel('epochs(neighbourhood)')
                    ax[i].set_ylabel('temp')
                else:
                    ax[i].plot(parameter)
                    ax[i].set_title(f"{parameter_name} ")
            fig.show()
        except ValueError:
            pass



