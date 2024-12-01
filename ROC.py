import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import pyreadr

# Charger le fichier RDA
fichier_rda = "./data/spam_data_train.rda"
result = pyreadr.read_r(fichier_rda)

# Extraire le DataFrame
for nom_objet, data in result.items():
    print(f"Objet trouvé : {nom_objet}")
    df = data  # On suppose qu'il y a un seul DataFrame dans le fichier
    break

# Aperçu des données
print(df.head())

df = df.fillna(0)
print(df.head())

X = df.drop(columns=["label"])
y = df["label"]

# Vérifier les valeurs manquantes
print(df.isnull().sum().sum())  # Doit retourner 0 si aucune valeur manquante

# Vérifier les types de données
print(df.dtypes)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

# Créer et entraîner le modèle
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prédire sur le jeu de test
y_pred = model.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score

# Calculer l'exactitude
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Rapport de classification complet
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Probabilités pour la classe positive
y_proba = model.predict_proba(X_test)[:, 1]

# Calculer FPR, TPR et AUC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Tracer la courbe ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig("ROC.png")