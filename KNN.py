import numpy as np

class KNN:
    @staticmethod
    def __euclidean(u, v):
        """ Calculer la distance (euclidienne) entre deux vecteurs """
        return np.sqrt(np.sum((u - v) * (u - v)))
    
    @staticmethod
    def __distances(u, dataset):
        """ Calculer les distances d'un vecteur à tous les autres vecteurs d'un dataset """
        dist = []
        for d in dataset:
            dist.append(KNN.__euclidean(u, d))
        return dist

    @staticmethod
    def __voisins(u, dataset, k):
        """ Récupèrer la liste des k voisins les plus proches"""
        distances = []
        for d in dataset:
            v = d[0: len(d) - 1]
            distances.append((d, KNN.__euclidean(u, v)))
        
        distances.sort(key=lambda tup: tup[1])
        neighs = []
        for i in range(k):
            neighs.append(distances[i][0])
        return neighs

    @staticmethod
    def classify(u, dataset, k):
        """ Faire des prédictions """
        v = KNN.__voisins(u, dataset, k)
        classes = {0:0, 1:0}
        for e in v:
            classes[e[-1]] += 1
        if classes[0] > classes[1]:
            return 0
        return 1  
