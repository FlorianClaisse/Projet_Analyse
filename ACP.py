import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

class ACP:

    @staticmethod
    def fit(dataset):
        return dataset - np.mean(dataset)
    
    @staticmethod
    def standardize(dataset):
        mean = np.mean(dataset, axis=0)
        std = np.std(dataset, axis=0)
        return (dataset - mean) / std, mean, std
    
    @staticmethod
    def scatter(dataset):
        return np.cov(dataset, rowvar=False)
    

    @staticmethod
    def sort(dataset):
        lambdas, vects = eig(dataset)
        idx = np.argsort(lambdas)[::-1]
        lambdas_tries = lambdas[idx]
        vects_tries = vects[:, idx]
        return lambdas, lambdas_tries, vects, vects_tries
    
    @staticmethod
    def fixe_k(lambdas, lambdas_sorted, vects_sorted, variance=0.7):
        k = 0
        total_variance_ratio = 0
        total_lambdas = np.sum(lambdas)
        while k < len(lambdas) and total_variance_ratio < variance:
            
            total_variance_ratio += lambdas_sorted[k] / total_lambdas
            k += 1
        base = vects_sorted[0:k]
        return k, total_variance_ratio, base
    
    @staticmethod
    def proj(scatter_dataset, base):
        return np.dot(base, scatter_dataset.T)
    
    @staticmethod
    def plot_acp(X_reduced, vects_sorted, lambdas_sorted, feature_names, label_colors=None, labels=None):
        plt.figure(figsize=(12, 8))

        # Cercle des corrélations
        plt.axhline(0, color='grey', linestyle='--', linewidth=0.7)
        plt.axvline(0, color='grey', linestyle='--', linewidth=0.7)
        circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--', linewidth=1.5)
        plt.gca().add_artist(circle)

        # Flèches des variables
        for i in range(vects_sorted.shape[1]):
            plt.arrow(0, 0, vects_sorted[0, i], vects_sorted[1, i],
                      color='r', alpha=0.8, head_width=0.03)
            plt.text(vects_sorted[0, i] * 1.1, vects_sorted[1, i] * 1.1,
                     feature_names[i], color='r', ha='center', va='center')

        # Points projetés
        if label_colors is None:
            label_colors = ['blue'] * X_reduced.shape[0]
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                              c=label_colors, alpha=0.6, cmap='viridis', edgecolor='k')
        if labels is not None:
            plt.legend(handles=scatter.legend_elements()[0], labels=np.unique(labels))

        # Configurations du graphique
        plt.title("Cercle des corrélations et projection des observations")
        plt.xlabel(f"Composante principale 1 ({lambdas_sorted[0] / sum(lambdas_sorted) * 100:.2f}%)")
        plt.ylabel(f"Composante principale 2 ({lambdas_sorted[1] / sum(lambdas_sorted) * 100:.2f}%)")
        plt.grid()
        plt.show()