
import pprint
import sys

sys.path.append('/Users/majkamiezianko/PycharmProjects/IO_TSP')

from copy import deepcopy, copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Simmulated annealing
"""

from search import *


class SimmulatedAnnealingSearch(Search):
    def __init__(self,
                 data: pd.DataFrame,
                 decreasing_factor=0.99,
                 max_same_iters=1000,
                 temperature_min=1e-11,
                 first_nr_of_neighbors=500,
                 temperature=0.99,
                 get_neighbours=LK_swap_neighbours):
        self.data = data
        self.decreasing_factor = decreasing_factor
        self.max_same_iters = max_same_iters
        self.temperature_min = temperature_min
        self.first_nr_of_neighbors = first_nr_of_neighbors
        self.temperature = temperature
        self.get_neighbours = get_neighbours

    def probability(self, temperature, old_cost, new_cost):
        if new_cost < old_cost:
            return 1.0
        else:
            return np.exp((old_cost - new_cost) / temperature)


    def run(self):
        # ////////Initial /////////
        cost_table = Cost(self.data)
        current_idx_list: List[int] = cost_table.multistart()

      #   current_idx_list: List[int]  = [116,83,80,125,81,82,74,75,77,79,78,76,17,20,16,21,3,22,23,5,105,
      # 14,107,19,18,71,7,8,10,113,104,6,0,15,1,50,56,53,44,102,43,34,35,36,40,13,
      # 11,30,26,29,42,33,38,37,25,24,32,121,27,28,31,41,39,120,4,55,123,51,49,12,
      # 114,9,119,2,89,115,59,61,60,90,57,63,99,112,65,54,46,48,52,117,47,45,93,111,
      # 110,106,126,92,94,122,96,97,100,101,62,118,95,108,86,85,84,87,109,70,69,68,
      # 67,72,73,66,58,124,88,91,98,64,103]

        best_solution = current_idx_list.copy()
        best_cost = cost_table.cost(current_idx_list)
        last_best_cost = best_cost
        counter = 0
        iter = 0

        history_of_cost = []
        history_of_iter = []
        history_of_step_length = []
        current_nr_of_neighbours = self.first_nr_of_neighbors
        # ////////Initial END/////////
        # //////Do Until/////////////
        while self.temperature > self.temperature_min:

            # długość schodka (nr_neighbors) zależna od temperatury
            current_nr_of_neighbours = int(current_nr_of_neighbours * self.temperature)
            history_of_step_length.append(current_nr_of_neighbours)
            neighbours = self.get_neighbours(current_nr_of_neighbours, deepcopy(current_idx_list), T=self.temperature)
            iter += 1
            # wybieranie wyniku spośród zestawu sąsiadów
            for neighbour in neighbours:
                for i in list(range(48)):
                    if i not in set(neighbour):
                        print('false')
                cost = cost_table.cost(neighbour)
                prob_condition = self.probability(self.temperature, best_cost, cost)
                # jesli przyjmujemy rozwiazanie gorsze to best cost sie nie zmienia ale zmienia sie rozwiazanie
                if prob_condition > np.random.uniform(0, 1, 1)[0]:  # wtedy przyjmujemy gorsze rozwiazanie
                    current_idx_list = neighbour
                    if cost < best_cost:
                        # jesli przyjmujemy lepsze rozwiazanie to best_cost bedzie nowym costem
                        best_solution = current_idx_list.copy()
                        best_cost = cost
            #update temperatury
            self.temperature = self.temperature * self.decreasing_factor
            counter = counter + 1 if last_best_cost == best_cost else 1
            last_best_cost = best_cost
            history_of_cost.append(int(best_cost))
            print(best_cost)
            if counter >= self.max_same_iters:
                break
        # //////Do Until END/////////////
        # //////Do Until END/////////////

        return {'best cost': float(best_cost), 'solution': [int(i) for i in list(best_solution)]}, {
                                                                'last best cost': history_of_cost,
                                                                'step length': history_of_step_length}










    # info, history = simulated_annealing(data=df, decreasing_factor=0.999, max_same_iters=50, get_neighbours=LK_swap_neighbours)
    # historyDF = pd.DataFrame(history)
    # historyDF.plot(y=['last best cost'])
    # # historyDF.plot.scatter(x='number of iteration', y=['step length'])
    # plt.show()
    # print(info)
    #

    #
    # cost_table = Cost(data=df)
    # pprint.pprint(cost_table.data)
    # indices = list(cost_table.data.index)
    # np.random.shuffle(indices)
    # print(indices)
    # print('first', cost_table.calcTime())
    # cost_table.reindex(indices)
    # pprint.pprint(cost_table.data)
    # print('after_reindex', cost_table.calcTime())
    #
    # for n in get_neighbours(5, [i for i in range(5)]):
    #     print(n)
if __name__ == '__main__':
    df = pd.read_excel("/Users/majkamiezianko/PycharmProjects/IO_TSP/data/Dane_TSP_76.xlsx")
    c = SimmulatedAnnealingSearch(data=df, get_neighbours=get_neighbours_swap)
    print(c.run())
