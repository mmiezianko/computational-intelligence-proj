import json

from search import *

class NNSearch(Search):
    def __init__(self,
                 data: pd.DataFrame,
                 start):

        self.data_table = Cost(data)
        self.start = start



    def run(self):
        #aktualna lista nieodwiedzonych wierzchołków
        vertices = list(range(self.data_table.data.shape[0]))
        start_point = vertices.pop(self.start) #wyrzucamy punkt startowy
        solution = [start_point] #punkt startowy zostaje dodany do listy ktora bedzie ROZWIAZANIEM
        while len(vertices) > 0: #dopóki lista wierzchołków jest większa od 0
            #bierzemy jako wiersz solution[-1] bo sprawdzamy minimum w kolumnie dla ostatniego wierzcholka w liscie. argmin sprawdza indeks dla minimalnej wartosci
            #vertices jako kolumna poniewaz podajemy miasta ktore mozna odwiedzic
            #z tego bierzemy minimum
            best_solution = np.argmin(self.data_table.data[solution[-1], vertices])
            #wyjmujemy z listy wierzcholkow miasto z najlepszym kosztem
            vertice_to_add = vertices.pop(best_solution)
            #dodajemy do listy miasto z najmniejszym kosztem
            solution.append(vertice_to_add)

        return {'best cost': float(self.data_table.cost(solution)), 'solution': solution}

#RUN ALGORITHM
if __name__ == '__main__':
    df = pd.read_excel("/Users/majkamiezianko/PycharmProjects/IO_TSP/data/Dane_TSP_127.xlsx")
    # c = NNSearch(data=df,start=6)
    # print(c.run())
    grid_search_history_nn = []
    for i in range(df.shape[0]):
        results_nn = NNSearch(data=df, start=i).run()
        results_nn['start']=i
        print(results_nn)

        grid_search_history_nn.append(results_nn)
    grid_search_history_nn.sort(key=lambda x: x['best cost'])
    json.dump(grid_search_history_nn, open('results/NN/results_nn_big.json', 'w'))






