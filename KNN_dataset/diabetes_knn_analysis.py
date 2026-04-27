# KNN Activity - Based on Report Tables

import math
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Part 2: Data Preprocessing
# -------------------------------

medians = {
    "Glucose": 117,
    "BloodPressure": 72,
    "SkinThickness": 29,
    "Insulin": 125,
    "BMI": 32.3
}

ranges = {
    "Glucose": (44, 199),
    "BloodPressure": (24, 122),
    "SkinThickness": (7, 99),
    "Insulin": (14, 846),
    "BMI": (18.2, 67.1)
}

scaled = {}
for feature, median in medians.items():
    min_val, max_val = ranges[feature]
    scaled[feature] = (median - min_val) / (max_val - min_val)

print("=== Part 2: Preprocessing Results ===")
for f, val in scaled.items():
    print(f"{f} scaled median: {val:.3f}")
print()

# -------------------------------
# Part 3: Manual KNN Computation
# -------------------------------

test = {"glucose": 138, "bmi": 36.1}

samples = [
    {"glucose":148, "bmi":33.6, "outcome":1},
    {"glucose":85,  "bmi":26.6, "outcome":0},
    {"glucose":183, "bmi":23.3, "outcome":1},
    {"glucose":89,  "bmi":28.1, "outcome":0},
    {"glucose":137, "bmi":43.1, "outcome":1},
    {"glucose":116, "bmi":25.6, "outcome":0},
    {"glucose":78,  "bmi":31.0, "outcome":1},
    {"glucose":115, "bmi":35.3, "outcome":0},
    {"glucose":197, "bmi":30.5, "outcome":1},
    {"glucose":125, "bmi":32.3, "outcome":1},
]

for s in samples:
    dx = test["glucose"] - s["glucose"]
    dy = test["bmi"] - s["bmi"]
    s["distance"] = math.sqrt(dx**2 + dy**2)

neighbors = sorted(samples, key=lambda x: x["distance"])

print("=== Part 3: Nearest Neighbors ===")
for i, n in enumerate(neighbors, start=1):
    print(f"Rank {i}: Glucose={n['glucose']}, BMI={n['bmi']}, "
          f"Distance={n['distance']:.2f}, Outcome={n['outcome']}")

def predict(k):
    top_k = neighbors[:k]
    outcomes = [n["outcome"] for n in top_k]
    prediction = max(set(outcomes), key=outcomes.count)
    print(f"K={k}, Neighbors outcomes={outcomes}, Prediction={prediction}")

print()
predict(3)
predict(5)
predict(7)
print()

# -------------------------------
# Extra: Manual Distance Table + Graph
# -------------------------------

import pandas as pd

# Data from your manual table (Samples 1–10)
data = {
    "Sample": [1,2,3,4,5,6,7,8,9,10],
    "Glucose": [148,85,183,89,137,116,78,115,197,125],
    "BMI": [33.6,26.6,23.3,28.1,43.1,25.6,31.0,35.3,30.5,32.3],
    "Distance": [10.31,53.84,46.79,49.65,7.07,24.38,60.22,23.01,59.27,13.54],
    "Outcome": [1,0,1,0,1,0,1,0,1,1]
}

df = pd.DataFrame(data)
print("=== Manual Distance Table ===")
print(df)

# Plot distances for each sample
plt.bar(df["Sample"], df["Distance"], color="skyblue")
plt.title("Euclidean Distances to Test Instance (Glucose=138, BMI=36.1)")
plt.xlabel("Sample #")
plt.ylabel("Distance")
plt.grid(True, axis="y")
plt.show()

# -------------------------------
# Part 4: Model Evaluation
# -------------------------------

conf_matrices = {
    3: np.array([[74,25],[27,28]]),
    5: np.array([[82,17],[26,29]]),
    7: np.array([[79,20],[27,28]])
}

total = 154

print("=== Part 4: Evaluation Results ===")
for k, cm in conf_matrices.items():
    tn, fp, fn, tp = cm.ravel()
    correct = tn + tp
    acc = correct / total * 100
    print(f"K={k}, Accuracy={acc:.2f}%, Correct={correct}/{total}")
print()

# -------------------------------
# Bonus: Accuracy vs K Plot
# -------------------------------

# Bonus: Accuracy vs K Plot (full range 1–20)
import matplotlib.pyplot as plt

# Replace with your actual accuracy values from the graph
k_values = list(range(1,21))
accuracies = [
    0.65, 0.66, 0.67, 0.68, 0.70, 0.71, 0.69, 0.70, 0.71, 0.72,
    0.71, 0.72, 0.73, 0.70, 0.71, 0.72, 0.71, 0.72, 0.71, 0.70
]

plt.plot(k_values, accuracies, marker='o', color='blue')
plt.title("KNN Accuracy for different values of K")
plt.xlabel("Value of K")
plt.ylabel("Testing Accuracy")
plt.grid(True)
plt.show()

# -------------------------------
# Bonus: Accuracy vs K Plot only for K=1,3,5,7
# -------------------------------

k_values_subset = [1, 3, 5, 7]
accuracies_subset = [accuracies[i-1] for i in k_values_subset]

plt.plot(k_values_subset, accuracies_subset, marker='o', color='red')
plt.title("KNN Accuracy for different values of K")
plt.xlabel("Value of K")
plt.ylabel("Testing Accuracy")
plt.grid(True)
plt.show()

# -------------------------------
# Bonus: Logistic Regression Comparison
# -------------------------------

print("=== Bonus: Logistic Regression vs KNN ===")

# Values from your report
knn_acc = 72.08
log_acc = 74.03

knn_conf = np.array([[82,17],[26,29]])
log_conf = np.array([[87,12],[27,27]])

print(f"KNN (K=5): Accuracy={knn_acc}%, Confusion Matrix=\n{knn_conf}")
print(f"Logistic Regression: Accuracy={log_acc}%, Confusion Matrix=\n{log_conf}")

if log_acc > knn_acc:
    print("Logistic Regression performed slightly better overall, but KNN was more sensitive to diabetic cases.")
else:
    print("KNN performed better overall.")
