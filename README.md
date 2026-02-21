# fraud-detection-machine-learning
# Fraud Detection Project
# Auteur : Moses BALUME

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Création d'un dataset simulé
# -----------------------------

np.random.seed(42)

# 1000 transactions normales
normal_transactions = np.random.normal(50, 10, 1000)

# 50 transactions frauduleuses (montants anormaux)
fraud_transactions = np.random.normal(200, 50, 50)

amounts = np.concatenate([normal_transactions, fraud_transactions])
labels = np.concatenate([np.zeros(1000), np.ones(50)])  # 0 = normal, 1 = fraude

data = pd.DataFrame({
    "amount": amounts,
    "fraud": labels
})

# -----------------------------
#  Séparation des données
# -----------------------------

X = data[["amount"]]
y = data["fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
#  Modèle de régression logistique
# -----------------------------

model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
#  Prédictions
# -----------------------------

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
#  Visualisation
# -----------------------------

plt.scatter(data["amount"], data["fraud"])
plt.title("Distribution des transactions")
plt.xlabel("Montant")
plt.ylabel("Fraude (0 = Non, 1 = Oui)")
plt.show()
