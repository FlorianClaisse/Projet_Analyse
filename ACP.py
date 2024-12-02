import numpy as np
from numpy.linalg import eig

class ACP:

    @staticmethod
    def fit(dataset):
        return dataset - np.mean(dataset)
    
    @staticmethod
    def scatter(dataset):
        return np.cov(dataset, rowvar=False)
    

    @staticmethod
    def sort(dataset):
        lambdas, vects = eig(dataset)
        idx = np.argsort(lambdas)[::-1]
        lambdas_tries = lambdas[idx]
        vects_tries = vects[:, idx]
        return (lambdas, lambdas_tries, vects, vects_tries)
    
    @staticmethod
    def fixe_k(lambdas, lambdas_sorted, vects_sorted, variance=0.7):
        k = 0
        total_variance_ratio = 0
        total_lambdas = np.sum(lambdas)
        while k < len(lambdas) and total_variance_ratio < variance:
            k += 1
            total_variance_ratio += lambdas_sorted[k] / total_lambdas
        base = vects_sorted[0:k]
        return (k, total_variance_ratio, base)
    
    @staticmethod
    def proj(scatter_dataset, base):
        return np.dot(base, scatter_dataset.T)