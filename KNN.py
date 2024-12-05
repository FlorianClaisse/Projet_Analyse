import numpy as np

class KNN:
    @staticmethod
    def __euclidean(self, u, v):
        """ Calculer la distance (euclidienne) entre deux vecteurs """
        return np.sqrt(np.sum((u - v) * (u - v)))
    
    @staticmethod
    def __distances(self, u, dataset):
        """ Calculer les distances d'un vecteur à tous les autres vecteurs d'un dataset """
        dist = []
        for d in dataset:
            dist.append(KNN.__euclidean(u, d))
        return dist

    @staticmethod
    def __voisins(u, dataset, k):
        """Récupérer la liste des k voisins les plus proches"""
        distances = []
        
        for d in dataset:
            # Supposant que les caractéristiques sont jusqu'à -1 et l'étiquette est à [-1]
            v = d[:-1]
            distance = KNN.__euclidean(u, v)
            distances.append((d, distance))
        
        # Trier par distance croissante
        distances.sort(key=lambda tup: tup[1])
        
        # Retourner les k plus proches voisins
        return [distances[i][0] for i in range(k)]  

    @staticmethod
    def classify(u, dataset, k):
        """Faire des prédictions pour un seul vecteur"""
        voisins = KNN.__voisins(u, dataset, k)
        classes = {0: 0, 1: 0}  # Assumant 2 classes: 0 et 1
        
        for voisin in voisins:
            label = voisin[-1]  # L'étiquette est la dernière colonne
            classes[label] += 1
        
        # Retourner la classe majoritaire
        return 0 if classes[0] > classes[1] else 1


    @staticmethod
    def predict(X, dataset, k):
        """Faire des prédictions sur tout un dataset"""
        predictions = []
        for u in X:
            predictions.append(KNN.classify(u, dataset, k))
        return predictions
