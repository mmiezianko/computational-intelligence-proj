from copy import deepcopy, copy
import numpy as np
import pandas as pd
from typing import List
import random

"""
Neighbourhood functions used in: climbing hill & SA
"""


def LK_swap(source, target, bool_table):
    """
    Applies LK-swap of the source having target as reference
    :param source: array to modify
    :param target: array to compare
    :param bool_table: array determining weather to compare given index of the arrays or not
    :return: modified array
    """

    target_to_swap = np.argwhere(np.array(bool_table) > 0).T[0]
    print('taget_to_swap', target_to_swap)
    for t in target_to_swap:
        for i, x in enumerate(source):
            if x == target[t]:
                a, b = source[t], source[i]
                source[t], source[i] = b, a

    return source


def LK_swap_neighbours(nr_of_neighbours, current_solution, T=None):
    if T == None:
        T=1
    for n in range(nr_of_neighbours):
        # prawdopodobienstwo zrobienia swapa zalezy od temperatury!
        # temp determinuje prawdopodobienstwo
        # czyli  losujemy kazda liczbe z przedzialu 0 - 1 i jesli ona jest mniejsza od T * 0.5 to przyjmujemy zamiane, czyli 1
        # wiec prawd zamiany maleje wraz ze spatkiem temp - jesli T * 0.5 = 0.3 to mamy tylko 30% szans na wylosowanie 1
        bool_table = [int(np.random.uniform(0, 1, 1)[0] < T * .5) for i in range(len(current_solution))]
        target = deepcopy(current_solution)
        print(target)
        np.random.shuffle(target)
        print(target)
        print(bool_table)
        neighbour = LK_swap(source=current_solution, target=target, bool_table=bool_table)
        print(neighbour)
        print()
        yield neighbour


# TODO:

### INSERT METHOD ###
def get_neighbours_insert(nr_neighbors, current_result, T=None):  # ||
    '''
    ta funkcja zamienia dowolny ciąg elementow elementy kolejnoscia - dla jednego sasiada
    czyli kazda zamiana to jest 1 sasiad

    :param nr_neighbors:
    :param current_result:
    :param T:
    :return: liste indeksow
    1 2 3 4 5 6
    '''
    # lista indeksow ktore bedziemy zamieniac. wybieramy element i od tego idzie ciąg n elementow na prawo od niego
    # TODO: wiecej zamianek

    tasks_to_swap: List[int] = np.random.choice([i for i in range(len(current_result))],
                                                nr_neighbors)
    for task in tasks_to_swap:  # iteracja po liscie indeksow
        neighbour = deepcopy(current_result)

        friends_to_make = [i for i in
                           range(len(current_result))]  # iterujemy po indeksach aby wybrac w ktore miejsce wstawic task
        friends_to_make.pop(task)

        friend_to_swap: int = np.random.choice(friends_to_make, 1)[
            0]  # 1 elementowa arrayka wiec bierzemy pierwszy element

        # neighbour2 = neighbour[:friend_to_swap] + neighbour[task:task+neighbours_to_insert] + neighbour[friend_to_swap:task] + neighbour[task+neighbours_to_insert:]
        task = neighbour.pop(task)
        neighbour.insert(friend_to_swap, task)

        yield neighbour  # generate neighbor
        # ta funkcja zamienia 2 elementy kolejnoscia dla jednego sasiada
        # czyli kazda zamiana to jest 1 sasiad
        # zwraca liste indeksow


# list_of_neighbours = get_neighbours()
# next(list_of_neighbours)


### SWAP METHOD ###

def get_neighbours_swap(nr_neighbors, current_result, T=None):  # ||
    '''
    ta funkcja zamienia 2 elementy kolejnoscia dla jednego sasiada
    czyli kazda zamiana to jest 1 sasiad

    :param nr_neighbors:
    :param current_result:
    :param T:
    :return: liste indeksow
    '''
    # lista indeksow ktore bedziemy zamieniac
    tasks_to_swap: List[int] = np.random.choice([i for i in range(len(current_result))], nr_neighbors)
    for task in tasks_to_swap:
        neighbour = copy(current_result)

        friends_to_make = [i for i in range(len(current_result))]
        friends_to_make.pop(task)
        friend_to_swap: int = np.random.choice(friends_to_make, 1)[0]  # 1 elementowa arrayka wiec bierzemy pierwszy element
        a, b = neighbour[task], neighbour[friend_to_swap]
        neighbour[task], neighbour[friend_to_swap] = b, a
        yield neighbour  # generate neighbor
        # ta funkcja zamienia 2 elementy kolejnoscia dla jednego sasiada
        # czyli kazda zamiana to jest 1 sasiad
        # zwraca liste indeksow


# list_of_neighbours = get_neighbours()
# next(list_of_neighbours)


def get_neighbours_to_climb(numberofneighbours, solution, T=None):
    neighbours = []

    for i in range(numberofneighbours):

        index = random.randrange(0, len(solution))

        firsthalf = solution[:index]
        secondhalf = solution[index:]

        secondhalf.reverse()

        neighbour = firsthalf + secondhalf

        if neighbour not in neighbours:
            neighbours.append(neighbour)

    return neighbours


# def neighborhood_insert(list, subset_length):
#     left_cut = random.randrange(0, len(list) - subset_length + 1)
#     right_cut = left_cut + subset_length
#
#     left_part = list[:left_cut]
#     subset = list[left_cut:right_cut]
#     right_part = list[right_cut:]
#
#     list1 = left_part + right_part
#
#     cut_index = random.randrange(0, len(list1) + 1)
#
#     left_part = list1[:cut_index]
#     right_part = list1[cut_index:]
#
#     list1 = left_part + subset + right_part
#
#     return list1

if __name__ == '__main__':

    list = list(range(48))

