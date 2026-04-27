import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1. DATA SET
# --------------------------
X = np.array([
    [2, 2], [3, 3], [5, 7], [6, 8],
    [7, 6], [10, 2], [11, 3], [12, 2]
])

labels_names = [f'P{i+1}' for i in range(len(X))]

# --------------------------
# 2. INITIAL CENTROIDS
# --------------------------
centroids = np.array([
    [2, 6],
    [6, 3],
    [10, 7]
])

k = 3
history = [centroids.copy()]
all_tables = []

# --------------------------
# 3. K-MEANS ITERATIONS
# --------------------------
for iteration in range(4):
    distances = np.zeros((len(X), k))

    # Compute distances
    for i in range(k):
        distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)

    # Assign clusters
    clusters = np.argmin(distances, axis=1)

    # Save table (NO PANDAS)
    table = []
    for i in range(len(X)):
        table.append([
            labels_names[i],
            X[i][0],
            X[i][1],
            round(distances[i][0], 2),
            round(distances[i][1], 2),
            round(distances[i][2], 2),
            clusters[i] + 1
        ])
    all_tables.append(table)

    # Compute new centroids
    new_centroids = np.array([
        X[clusters == i].mean(axis=0) if len(X[clusters == i]) > 0 else centroids[i]
        for i in range(k)
    ])

    centroids = new_centroids
    history.append(centroids.copy())

# --------------------------
# 4. TABLES
# --------------------------
for i, table in enumerate(all_tables):
    print(f"\n=== ITERATION {i+1} ===")
    print("Point | X | Y | D1 | D2 | D3 | Cluster")
    for row in table:
        print(row)
    print("\nCentroids:", history[i+1])

# --------------------------
# --------------------------
# 5. PLOTTING
# --------------------------
colors = ['red', 'green', 'blue']

fig, axes = plt.subplots(1, len(history), figsize=(24, 5), constrained_layout=True)

for i, ax in enumerate(axes):

    cents = history[i]

    # compute clusters
    distances = np.zeros((len(X), k))
    for j in range(k):
        distances[:, j] = np.linalg.norm(X - cents[j], axis=1)

    clusters = np.argmin(distances, axis=1)

    # --------------------------
    # PLOT CLUSTERS 
    # --------------------------
    for j in range(k):
        ax.scatter(
            X[clusters == j][:, 0],
            X[clusters == j][:, 1],
            color=colors[j],
            s=120,
            edgecolors='black',
            label=f'Cluster {j+1}'
        )

    # --------------------------
    # LABEL POINTS
    # --------------------------
    for idx, point in enumerate(X):
        ax.text(point[0] + 0.15, point[1] + 0.15,
                labels_names[idx],
                fontsize=9)

    # --------------------------
    # CENTROIDS 
    # --------------------------
    ax.scatter(
        cents[:, 0], cents[:, 1],
        c='black',
        marker='*',
        s=350,
        edgecolors='yellow',
        linewidth=1.5,
        label='Centroids'
    )

    # centroid labels
    for j in range(k):
        ax.text(cents[j,0] + 0.2, cents[j,1] + 0.2,
                f'C{j+1}', fontsize=11, fontweight='bold')

    # --------------------------
    # MOVEMENT ARROWS
    # --------------------------
    if i > 0:
        prev = history[i-1]
        for j in range(k):
            ax.arrow(
                prev[j,0], prev[j,1],
                cents[j,0] - prev[j,0],
                cents[j,1] - prev[j,1],
                head_width=0.25,
                length_includes_head=True,
                alpha=0.6,
                linestyle='--',
                color='black'
            )

    # --------------------------
    # STYLE I
    # --------------------------
    ax.set_title(f'Iteration {i}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')  # IMPORTANT FIX
    ax.grid(True, linestyle='--', alpha=0.3)

    ax.legend(loc='upper right', fontsize=8)

plt.show()