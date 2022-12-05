from typing import Tuple

import networkx as nx
import numpy as np
from scipy import linalg
from sklearn.cluster import KMeans

from dataset_reader import read_graph


class Spectral_clustering:
    '''
    This class clusters a given graph by considering spectral information. It follows the algorithm proposed in "On Spectral Clustering: Analysis and an algorithm" by Andrew Y. Ng, Michael I. Jordan, and Yair Weiss.
    '''

    def __init__(self, G: nx.Graph) -> None:
            '''
            Initialization of the class function that uses the networkx graph for initialisation
            :param G: NetworkX graph (weighted and unweighted)

            '''
            self.G = G
            self.A = nx.to_numpy_matrix(self.G)
            self.D = np.diagflat(np.sum(self.A, axis=1))


    def compute_Laplacian(self) -> np.matrix:
        """
        This function computes the normalised Laplacian matrix L=D^(-1/2)AD^(-1/2).
    
        :return: returns the normalised Laplacian as a numpy matrix of shape (number of vertices, number of vertices)
        """
        D_inv = np.linalg.inv(np.sqrt(self.D))
        L = np.dot(D_inv, self.A).dot(D_inv)

        return L

    def spectral_clustering(self, get_optimal_k: bool = True, k: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
 
        This function clusters a graph into either:
        (1: get_optimal_k = True) the optimal number of clusters 
        or
        (2: get_optimal_k = False) k clusters.
        It therefore computes the eigenvector of the laplacian and select corresponding eigenvectors which then 
        will be clustered using K-means.
        Th
    
        :param get_optimal_k: Determines whether a flexible or the optimal number of eigenvectors will be selected
        :param k: the number of clusters to be identified, works if selection method is manual
        :return: returns a numpy array of shape (number of vertices,) containing the label for each vertex,
   
        """


        L = self.compute_Laplacian()
    
        # returns eigenvalues and vectors in ascending order
        values, vectors = linalg.eigh(L)
        k_optimal  = np.argmax(np.abs(np.diff(values)))
        k_optimal = len(values) - k_optimal -1
        print(f"The optimal number of clusters is {k_optimal}.") 

        if get_optimal_k:
            k = k_optimal
     
        X = vectors[:, -k:]
        # normalise the matrix formed by the eigvectors
        Y = X / np.linalg.norm(X, axis=1, keepdims=True)

        clusters = KMeans(n_clusters=k).fit(Y).labels_

        return clusters

    def compute_Fiedler(self):
        ''' Computes the Fiedler vector of the Laplacian matrix L = D - A'''
        A = nx.to_numpy_matrix(self.G)
        D = np.diagflat(np.sum(A, axis=1))
        laplace_mat = D - A
        eigen_values, eigen_vecs = linalg.eig(laplace_mat)
        return eigen_values, eigen_vecs[:, 1]