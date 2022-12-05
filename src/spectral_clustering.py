from typing import Tuple

import networkx as nx
import numpy as np
from scipy import linalg
from sklearn.cluster import KMeans

from dataset_reader import read_graph


class Spectral_clustering:
    '''
    TODO
    '''

    def __init__(self, G: nx.Graph) -> None:
            '''
            Initialization of the class function with defined parameters 
            :param M: number of edges in our reservoir sample stored in memory
            :param edge_stream: edge stream being processed 
            '''
            self.G = G
            self.A = nx.to_numpy_matrix(self.G)
            self.D = np.diagflat(np.sum(self.A, axis=1))



    # selection_methods: Dict[str, Callable[[np.ndarray, int], int]] = {
    #     # +2 because 1 accounts for indices starting from 0 and 1 accounts for the fact that k is the index of the NEXT
    #     # eigenvalue
    #     'auto': lambda eigenvalues, k: np.argmin(np.ediff1d(eigenvalues)) + 1,
    #     'manual': lambda eigenvalues, k: k
    # }   

    def compute_Laplacian(self) -> np.matrix:
        D_inv = np.linalg.inv(np.sqrt(self.D))
        # L = D_inv @ self.A @ D_inv           #infix operator that is designated to be used for matrix multiplication

        L = np.dot(D_inv, self.A).dot(D_inv)

        return L

    def spectral_clustering(self, get_optimal_k: bool = True, k: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        TODO (change text)
        This function implements the algorithm described in
        “On Spectral Clustering: Analysis and an algorithm” (Links to an external site.)
        by Andrew Y. Ng, Michael I. Jordan, Yair Weiss
        This function computes k clusters in the graph contained in file with spectral clustering and returns a numpy
        array of shape (number of vertices,) containing the label for each vertex, the Fiedler vector and
        the adjacency matrix of the graph
        :param G: nx graph
        :param number_of_clusters_selection: either auto or manual, determines how k is selected
        :param k: the number of clusters to be identified, works if selection method is manual
        :return: returns a numpy array of shape (number of vertices,) containing the label for each vertex,
                        a numpy array of shape (number of vertices,) containing the Fiedler vector,
                        a numpy matrix of shape (number of vertices, number of vertices) containing the adjacency matrix
        """


        L = self.compute_Laplacian()
    
        # returns eigenvalues and vectors in ascending order
        values, vectors = linalg.eigh(L)
        k_optimal  = np.argmax(np.abs(np.diff(values))) + 1
        print(f"The optimal number of clusters is {(len(values) - k_optimal) -1}.") 

        if get_optimal_k:
            k = k_optimal
            X = vectors[:, k:]
        else:
            X = vectors[:, -k:]
        Y = X / np.linalg.norm(X, axis=1, keepdims=True)
        result = KMeans(n_clusters=k).fit(Y).labels_

       

        return result, self.A

    def compute_Fiedler(self):
        ''' Compute the Fiedler vector of the Laplacian matrix L = D - A'''
        A = nx.to_numpy_matrix(self.G)
        D = np.diagflat(np.sum(A, axis=1))
        laplace_mat = D - A
        eigen_values, eigen_vecs = linalg.eigh(laplace_mat)
        print(eigen_vecs)
        return eigen_values, eigen_vecs[:, 1]

if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    file = 'example1.dat'
    graph = read_graph(file)
    clustering = Spectral_clustering(graph)
    clustering.spectral_clustering(get_optimal_k = False, k = 10)
    _, fiedler = clustering.compute_Fiedler()
    plt.plot(np.sort(fiedler))
    plt.xlabel("Node")
    plt.ylabel("Eigenvector")
    plt.show()
