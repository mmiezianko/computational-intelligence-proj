import numpy as np 
from numba import njit
# from numba.types import bool_
from typing import Tuple

def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

# @njit
def path_length(genotype:np.ndarray, adj_matrix:np.ndarray) -> float:
    dist = 0
    for i, city1 in enumerate(genotype):
        city2 = genotype[(i + 1) % genotype.size]
        dist += adj_matrix[city1, city2]
    return dist


@njit
def calc_time(indexes: np.ndarray, data: np.ndarray):
    """
    Function calculates makespan of the given 'data', assuming that task are in order dictated by 'indexes'
    :param indexes: 1d numpy array contatining information about order of the tasks. eg. [1, 0, 2] means that task 1 goes first, than 0 and 2.
    :param data: 2d numpy array with information about processing time of every task on every machine. 
        eg. data[i][j] = t means that task i takes time t to process on machine j
    :returns int: makespan
    """
    result = data.copy()
    # reindexing data 
    result[:] = result[indexes]
    result[0, :] = np.cumsum(result[0, :])
    result[:, 0] = np.cumsum(result[:, 0])
    for i in range(1, data.shape[0]):
        for j in range(1, data.shape[1]):
            result[i, j] += max(result[i - 1, j], result[i, j - 1])
    return result[-1, -1]


# @njit
def apply_crossover(population: np.ndarray, cx_fun, mask_size: int) -> np.ndarray:
    """
    Function apply given crossover method on the given population. 
    :param population: 2d numpy array in which rows are genotypes indicating order of processing tasks
    :param cx_fun: Crossover function.
    :param mask_size: number of elements between crossover points (in Order Crossover), or number of masked elements (in Masked Crossover)
    :returns 2d numpy array: New population of children. Note that every 2 parents produce 2 children so the size of the population doesn't change
    """
    new_population = np.zeros(population.shape, dtype=np.uint64) 
    for i in range(0, population.shape[0], 2):
        p1 = population[i, :]
        p2 = population[i + 1,:]
        child1, child2 = cx_fun(p1, p2, mask_size)
        new_population[i] = child1
        new_population[i+1] = child2 
    return new_population


@njit
def _ox_one_child(p1: np.ndarray, p2: np.ndarray, cx1: int, cx2: int) -> np.ndarray:
    """
    Function implements 'Order Crossover' to produce 1 child from 2 parents p1, p2. 
    This is only helper function, usually 'ox' should be used instead.
    :param p1: 1d numpy array - parent1 genotype 
    :param p2: 1d numpy - parent2 genotype 
    :param cx1: index] of the first crossover point. 
    :param cx2: ]index of the second crossover point. 
    :returns 1d numpy array: Child genotype 
    """
    size = p1.shape[0]
    child = np.zeros(p1.shape, dtype=p1.dtype)
    child[cx1: cx2] = p2[cx1: cx2]
    copied_to_c = np.zeros(size) # helper array indicating whether number has been already copied (copied[number] == 1) or not (copied[number] == 0)
    copied_to_c[p2[cx1:cx2]] = 1 
    parent_pos = cx2 # keeps track of the position we are in parent 
    for i in range(size - cx2 + cx1): #  there are cx2-cx1 elements that have been already copied
        offset = (cx2 + i) % size # if we exceed array size, go to the begining
        while copied_to_c[p1[parent_pos]] == 1:
            parent_pos = (parent_pos + 1) % size 
        child[offset] = p1[parent_pos]
        copied_to_c[p1[parent_pos]] = 1
    return child


@njit
def ox(p1: np.ndarray, p2: np.ndarray, mask_size: int) -> Tuple:
    """
    Function implements 'Order Crossover' to produce 2 children from parents p1 and p2
    :param p1: 1d numpy array - parent1 genotype 
    :param p2: 1d numpy - parent2 genotype 
    :param mask_size: number of ellements between cx1 and cx2. 
        In Alternative implementation this could be exchanged for two random crossover points 
    :returns Tuple: 2 children genotypes
    """
    size = len(p1)
    # TODO: it is worth considering whether to keep crossover points constant for every simulation
    cx1 = np.random.randint(0, size - mask_size)# included index
    cx2 = cx1 + mask_size# excluded index
    child1 = _ox_one_child(p1, p2, cx1, cx2)
    child2 = _ox_one_child(p2, p1, cx1, cx2)
    return child1, child2


@njit
def _mkx_one_child(p1: np.ndarray, p2: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Function implements 'Order Crossover' to produce 1 child from parents p1 and p2.
    This is only helper function, usually 'mkx' should be used instead.
    :param p1: 1d numpy array - parent1 genotype 
    :param p2: 1d numpy - parent2 genotype 
    :param mask: 1d numpy array indicating witch elements in parent1 are copied to the child. Format:
        array of False's and True's. eg. mask[i] == True means that element at index i in parent1 genotype need to be copied to te child
    :returns Tuple: 2 children genotypes
    """
    size = p1.shape[0]
    child = np.zeros(p1.shape, dtype=p1.dtype)
    child[mask] = p1[mask]
    copied_to_c = np.zeros(size) # helper array indicating whether number has been already copied (copied[number] == 1) or not (copied[number] == 0)
    copied_to_c[p1[mask]] = 1 
    parent_pos = 0 # keeps track of the position we are in parent2
    # assert len(set(list(p1))) == len(p1)
    # assert len(set(list(p2))) == len(p2)
    for i in range(size): # iterate over child, and fill missing values
        if mask[i]:
            continue 
        while copied_to_c[p2[parent_pos]] == 1:
            # print(parent_pos, p2[parent_pos], copied_to_c[p2[parent_pos]])
            parent_pos = (parent_pos + 1) % size 
        child[i] = p2[parent_pos]
        copied_to_c[p2[parent_pos]] = 1
    # assert len(set(list(child))) == len(child), child 
    return child


# @njit # could make it decorated
def mkx(p1: np.ndarray, p2: np.ndarray, mask_size: int) -> Tuple:
    """
    Function implements 'Order Crossover' to produce 2 children from parents p1 and p2.
    :param p1: 1d numpy array - parent1 genotype 
    :param p2: 1d numpy - parent2 genotype 
    :param mask-size: Number of elements in a mask
    :returns Tuple: 2 children genotypes
    """
    size = p1.size
    mask = np.random.choice(size, mask_size)
    temp = np.zeros(size)
    bool_mask = np.zeros_like(temp, bool)
    bool_mask[mask] = True
    child1 = _mkx_one_child(p1, p2, bool_mask)
    child2 = _mkx_one_child(p2, p1, bool_mask)
    return child1, child2


@njit
def tournament_selection(population: np.ndarray, fitness: np.ndarray, size: int) -> np.ndarray:
    """
    Function implements 'Tournament Selection' to select candidates for reproduction.
    :param population: 2d numpy array in which rows are genotypes indicating order of processing tasks
    :param fitness: 1d array indicating fitness of each individual in population. eg. fitness[i] = 10 means that population[i] has makespan of 100
    :param size: Number of selected individuas
    :returns 2d array of length = 'size': Selected individuals.  Each row represent genotype of selected individuals
    """
    # stack fitness as first column 
    population = population.copy()
    size = min(size, population.shape[0])
    #inicjujemy nową populację
    new_population = np.zeros((size, population.shape[1]), dtype="uint8")
    for i in range(0, size, 2):
        #losujemy 4 rodziców spośród populacji
        parents_indicies = np.random.choice(population.shape[0], replace=False, size=4)
        #bierzemy dwóch rodziców od końca, ponieważ lista fitness jest posortowana malejąco
        best_2_inidicies = parents_indicies[np.argsort(fitness[parents_indicies])[:2]]
        parents = population[best_2_inidicies]
        new_population[i, :] = parents[0,:]
        new_population[i + 1, :] = parents[1,:]
    return new_population


# @njit # chould not decorate it due to np.random.choice function with 'p' optipon
def roulette_selection(population: np.ndarray, fitness: np.ndarray, size: int)-> np.ndarray:
    """
    Function implements 'Roulette Selection' to select candidates for reproduction. 
    It usess 1 - scaled_fitness(min max scaling) as real fitness, because Roullete selection promote individuals with the
    highest fitness (which is the oposite of what we need)
    :param population: 2d numpy array in which rows are genotypes indicating order of processing tasks
    :param fitness: 1d array indicating fitness of each individual in population. eg. fitness[i] = 10 means that population[i] has makespan of 100
    :param size: Number of selected individuas
    :returns 2d array of length = 'size': Selected individuals.  Each row represent genotype of selected individuals
    """
    population = population.copy()
    size = min(size, population.shape[0])
    new_population = np.zeros((size, population.shape[1]), dtype=population.dtype)
    min_ = min(fitness)
    max_ = max(fitness)
    #skalowanie zeby sie sumowalo do 100%
    reversed_fitness = 1 - (fitness - min_)/(max_ - min_) # min max scaling; 1- () beacause we need to minimize our cost
    try:
        probas = reversed_fitness / np.sum(reversed_fitness)
        print(probas.shape)
    except Exception:
        #jesli jest dzielenie przez 0 to dajemy prob 0.5
        probas = np.full_like(reversed_fitness,0.5)

    for i in range(0, size):
        mask = np.random.choice(population.shape[0], p=probas)
        selected = population[mask]
        new_population[i, :] = selected[:]
    return new_population


# @njit # chould not decorate it due to np.random.choice function with 'p' optipon
def rank_selection(population: np.ndarray, fitness: np.ndarray, size: int) -> np.ndarray:
    """
    Function implements 'Ranking Selection' to select candidates for reproduction.
    :param population: 2d numpy array in which rows are genotypes indicating order of processing tasks
    :param fitness: 1d array indicating fitness of each individual in population. eg. fitness[i] = 10 means that population[i] has makespan of 100
    :param size: Number of selected individuas
    :returns 2d array of length = 'size': Selected individuals.  Each row represent genotype of selected individuals
    """
    population = population.copy()
    size = min(size, population.shape[0])
    # sorting in descending order. The higher the fitness the lower the rank 
    # the smaller probability it receives
    ranked = population[np.flip(np.argsort(fitness))]
    new_population = np.zeros((size, population.shape[1]), dtype="uint8")
    indexes = np.arange(1, ranked.shape[0] + 1)
    probas = indexes / sum(indexes)
    for i in range(0, size):
        selected = ranked[np.random.choice(ranked.shape[0], p=probas)]
        new_population[i, :] = selected[:]
    return new_population

@njit
def mask_permutation_mutation(population: np.ndarray, mask_size: int, threshold: float) -> np.ndarray: 
    """
    Function implements custom mutation wihch changes the order of randomly selected tasks.
    :param population: 2d numpy array in which rows are genotypes indicating order of processing tasks
    :param mask_size: Number of items we want to permutate
    "param threshold: mutation probability
    :returns 2d array: New population which is the same size as that in input
    """
    for i in range(population.shape[0]):
        if np.random.rand() > threshold:
            continue 
        genotype = population[i]
        mask_size = min(mask_size, genotype.shape[0])
        mask = np.random.choice(genotype.shape[0], mask_size, replace=False)
        # Randomly change order of the selected elements 
        order = np.random.permutation(mask.shape[0])
        # apply mutation to the genotype 
        new_genotype = genotype.copy() 
        new_genotype[mask] = new_genotype[mask[order]]
        population[i] = new_genotype
    return population

@njit
def random_insert(population, threshold):
    """
    function iterates over population and for each individual pick one element from its genotype, end insert it in random spot
    """
    size, n = population.shape
    for i in range(size):
        if np.random.rand() > threshold:
            continue 
        indexes = np.random.choice(n, replace=False, size=2)
        old_idx = indexes[0]
        new_idx = indexes[1]
        direction = 1 if old_idx > new_idx else -1 
        left = min(old_idx, new_idx)
        right = max(old_idx, new_idx)
        population[i][left: right + 1] = np.roll(population[i][left: right+1], direction)
    return population

# @njit
def arbitrary_swap(population, threshold):
    """
    Swaps two random elements 
    """
    size, n = population.shape
    for i in range(size):
        if np.random.rand() > threshold:
            continue 
        indexes = np.random.choice(n, replace=False, size=2)
        a = indexes[0]
        b = indexes[1]
        population[i][[a, b]] = population[i][[b, a]]
    return population 

# @njit
def consequtive_swap(population, threshold):
    """
    Zamienia dwa sąsiednie, następujący (jeden po drugim) elementy
    """
    size, n = population.shape
    for i in range(size):
        if np.random.rand() > threshold:
            continue 
        a = np.random.randint(0, n-1)
        population[i][[a, a+1]] = population[i][[a+1, a]]
    return population
