from annealing import Cost

from search import *


"""
Hill climbing with multistart (multistart version is implemented in main.py)
"""


class ClimbSearch(Search):
    def __init__(self,
                 data: pd.DataFrame,
                 nr_of_neighbours=50,
                 get_neighbours=get_neighbours_to_climb,
                 nr_of_same_costs=10):

        self.data = data
        self.nr_of_neighbours = nr_of_neighbours
        self.get_neighbours = get_neighbours
        self.nr_of_same_costs = nr_of_same_costs



    def run(self):
        cost_table = Cost(self.data)
        # generate first random solution
        current_idx_list: List[int] = (list(cost_table.multistart()))
        best_solution = current_idx_list.copy()
        best_cost = cost_table.cost(best_solution)
        last_best_cost = best_cost

        counter = 0
        iter = 0

        while True:
            # new solution
            new_solution_idx = best_solution.copy()
            neighbours = self.get_neighbours(self.nr_of_neighbours, deepcopy(new_solution_idx))

            for neighbour in neighbours:
                iter += 1
                print(iter, best_cost)
                new_cost = cost_table.cost(neighbour)

                if new_cost < last_best_cost:
                    best_solution = neighbour
                    best_cost = new_cost

                counter = counter + 1 if last_best_cost == best_cost else 1
                last_best_cost = best_cost
            if counter >= self.nr_of_same_costs:
                break

        return {'best cost': float(best_cost), 'solution': best_solution }, {}


                                                                # 'last best cost': history_of_cost,
                                                                # 'step length': history_of_step_length}


if __name__ == '__main__':
    df = pd.read_excel("/Users/majkamiezianko/PycharmProjects/IO_TSP/data/Dane_TSP_48.xlsx")
    c = ClimbSearch(data=df, get_neighbours=get_neighbours_swap)
    print(c.run())
