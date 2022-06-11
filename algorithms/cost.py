import itertools
import random
from copy import deepcopy

import pandas as pd
import openpyxl
import numpy as np


class Cost:
    def __init__(self, data: pd.DataFrame):
        self.data = data.to_numpy()[:, 1:]

    def cost(self, actual_list):

        path_sum = 0
        for i in range(len(actual_list)): #OD ZERA ITERACJA
            list = deepcopy(actual_list)

            point = list[i]
            # TUTAJ LICZYMY ODLEGLOSC JUZ OD PIERWSZEGO DO OSTATNIEGO
            # ZATEM POWROT NA POCZATEK JEST LICZONY LECZ NIE PRINTUJEMY GO!!!
            point_prev = list[i-1]
            path_sum += self.data[point, point_prev]
        # print(actual_list)
        return float(path_sum)


    def multistart(self):
        solution = []

        first_column = [x for x in range(0, self.data.shape[0])]

        while len(solution) < len(first_column):

            task = random.choice(first_column)

            if task not in solution:
                solution.append(task)

        return solution

if __name__ == '__main__':
    df = pd.read_excel("/Users/majkamiezianko/PycharmProjects/IO_TSP/data/Dane_TSP_127.xlsx")
    list = [109, 84, 85, 86, 87, 108, 95, 118, 62, 101, 100, 82, 81, 125, 80, 83, 116, 77, 75, 74, 68, 69, 70, 67, 66, 72, 73, 76, 17, 71, 7, 18, 22, 23, 8, 10, 113, 104, 14, 105, 5, 107, 19, 3, 21, 20, 16, 78, 79, 11, 30, 26, 29, 33, 38, 37, 25, 24, 32, 28, 31, 121, 27, 96, 97, 126, 94, 122, 41, 42, 39, 34, 36, 35, 40, 13, 15, 0, 6, 12, 1, 50, 56, 53, 43, 44, 102, 92, 106, 110, 111, 93, 45, 117, 47, 52, 48, 46, 54, 65, 123, 51, 4, 55, 120, 49, 114, 119, 9, 2, 89, 115, 59, 58, 61, 60, 90, 99, 57, 63, 112, 64, 98, 91, 88, 124, 103]

    # list = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 73, 14, 15, 16, 17, 36, 35, 19, 18, 30, 29, 28, 25, 26, 27, 32, 31, 33, 34, 37, 38, 39, 40, 59, 58, 42, 41, 53, 52, 51, 54, 57, 60, 61, 56, 64, 65, 66, 67, 68, 69, 70, 71, 72, 63, 62, 55, 50, 49, 48, 46, 47, 43, 44, 45, 23, 24, 20, 21, 22, 0, 75, 74, 1, 2, 3]


    a = Cost(data=df)
    print(a.cost(actual_list=list))
    # print(a.multistart())
