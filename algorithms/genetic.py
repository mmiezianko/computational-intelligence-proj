from enum import Enum
import numpy as np
from math import factorial
from utils import read_data
import routines as rt 
from typing import Callable


class Crossover(Enum):
    OX = "Order"
    MKX = "Masked"

    def __str__(self):
        return f"Crossover {self.value}"


class Selection(Enum):
    TOUR = "Tournament"
    ROULETTE = "Roulette"
    RANK = "Rangking"



class Mutation(Enum):
    MASK = 1
    INSERT = 2
    ARBITRARY = 3
    CONSEQUTIVE = 4


class GeneticSearch:
    def __init__(
        self, data: np.ndarray,
        selection_method: Selection, 
        cx_method: Crossover, 
        mut_method: Mutation,
        cost_fun: Callable[[np.ndarray, np.ndarray], float],
        generations, 
        pop_size, 
        mutation_threshold,
        crossover_mask_size,
        mut_mask_size,
        verbose=True
        ):
        self.data = data.copy()
        self.n_jobs = self.data.shape[0]
        self.selection_method = selection_method
        self.cx_method = cx_method
        self._cx_mapping = {
            Crossover.OX: rt.ox,
            Crossover.MKX: rt.mkx
        }
        self._selection_mapping = {
            Selection.TOUR: rt.tournament_selection,
            Selection.RANK: rt.rank_selection,
            Selection.ROULETTE: rt.roulette_selection
        }

        self.cost_fun = lambda genotype: cost_fun(genotype, self.data)
        self.max_generations = generations 
        self.pop_size = pop_size 
        self.mut_threshold = mutation_threshold
        self.cx_mask_size = crossover_mask_size
        self.mut_mask_size = mut_mask_size
        self._mutation_mapping = {
            Mutation.MASK: lambda pop: rt.mask_permutation_mutation(pop, self.mut_mask_size, self.mut_threshold),
            Mutation.INSERT: lambda pop: rt.random_insert(pop, self.mut_threshold),
            Mutation.ARBITRARY: lambda pop: rt.arbitrary_swap(pop, self.mut_threshold),
            Mutation.CONSEQUTIVE: lambda pop: rt.consequtive_swap(pop, self.mut_threshold)
        }
        self.mutation_fun = self._mutation_mapping[mut_method]
        self.history = dict()
        self._best_genotype = None 
        self._best_fitness = np.inf
        self.verbose = verbose


    def get_initial_population(self, size: int=None) -> np.ndarray:
        size = size or self.pop_size
        if factorial(self.n_jobs) < size: #factorial Raise a ValueError if x is negative or non-integral.
            raise ValueError("Expected initial population size is too large") 
        population_set = set()
        while len(population_set) < size:
            population_set.add(tuple(np.random.permutation(self.n_jobs)))
        return np.array([np.array(genotype) for genotype in population_set])


    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        return np.array(list(map(self.cost_fun, population)))
    

    def select_next_population(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        #funkcja zawiera mapowanie metod delekcji
        selection_fun = self._selection_mapping[self.selection_method]
        return selection_fun(population, fitness, self.pop_size)

    #operacje krzyżowania
    def crossover(self, population: np.ndarray) -> np.ndarray:
        cx_fun = self._cx_mapping[self.cx_method]
        return rt.apply_crossover(population, cx_fun, self.cx_mask_size)

    #mutacje
    def mutate(self, population) -> np.ndarray:
        return self.mutation_fun(population)


    def stop_criterion(self, population: np.ndarray, fitness: np.ndarray) -> bool:
        return self._generation > self.max_generations


    def _best_individual(self, population, fitness) -> np.ndarray:
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    
    def _log(self):
        if self._generation % 50 == 0 and self.verbose: 
            print(f"Generation {self._generation}")
    

    def _update_stats(self, population, fitness):
        best_genotype, best_fitness = self._best_individual(population, fitness)
        self._best_genotype = best_genotype if best_fitness < self._best_fitness else self._best_genotype 
        self._best_fintess = min(self._best_fitness, best_fitness)
        if self._generation % 50 == 0: 
            self.history[self._generation] = fitness.copy() 
            print(len(set([tuple(list(arr)) for arr in population])))


    def run(self):

        population = self.get_initial_population() #inicjujemy początkową populację osobników
        fitness = self.evaluate_population(population) #poddajemy każdego z nich ocenie
        self._generation = 0
        while not self.stop_criterion(population, fitness):
            self._update_stats(population, fitness)
            population = self.select_next_population(population, fitness) # z populacji wybieramy osobniki najlepiej do tego przystosowane
            population = self.crossover(population)  #tworzenie dzieci
            population = self.mutate(population) #wprowadzenie różnorodnosci w populacji
            fitness = self.evaluate_population(population)
            self._generation += 1
            self._log()
        return {'best cost': int(self._best_fintess), 'solution': list(int(i) for i in self._best_genotype)}, {}


if __name__ == '__main__':
    # TEST
    import matplotlib.pyplot as plt
    data = read_data("/Users/majkamiezianko/PycharmProjects/IO_TSP/data/Dane_TSP_127.xlsx")
    solver = GeneticSearch(
        data,
        Selection.ROULETTE,
        Crossover.OX,
        Mutation.CONSEQUTIVE,
        rt.path_length,
        generations=1000,
        pop_size=200,
        mutation_threshold=0.01, 
        crossover_mask_size=10,
        mut_mask_size=3
    )
    print(solver.run())
    mins = [min(fitness) for fitness in solver.history.values()]
    maxs = [max(fitness) for fitness in solver.history.values()]
    mean = [np.mean(f) for f in solver.history.values()]
    plt.plot(range(len(mins)), mins, color='r', label = "mins")
    plt.plot(range(len(maxs)), maxs, color='g', label = "maxs")
    plt.legend(loc="upper right")

    plt.show()





