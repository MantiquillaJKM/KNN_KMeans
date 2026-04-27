import matplotlib.pyplot as plt
import math

# Dataset
data = [
    (150, 5, "Healthy"),
    (300, 25, "Unhealthy"),
    (350, 28, "Unhealthy"),
    (200, 10, "Healthy"),
    (400, 30, "Unhealthy"),
    (180, 8, "Healthy"),
    (220, 12, "Healthy"),
    (420, 35, "Unhealthy"),
    (250, 15, "Healthy")
]

# New point
new_point = (330, 20)

# Separate data
healthy_x = []
healthy_y = []
unhealthy_x = []
unhealthy_y = []

for cal, sugar, label in data:
    if label == "Healthy":
        healthy_x.append(cal)
        healthy_y.append(sugar)
    else:
        unhealthy_x.append(cal)
        unhealthy_y.append(sugar)

# Plot points
plt.scatter(healthy_x, healthy_y, label="Healthy")
plt.scatter(unhealthy_x, unhealthy_y, label="Unhealthy")

# Plot new point
plt.scatter(new_point[0], new_point[1], marker='x', s=100, label="New Point")

# Draw lines to nearest neighbors (K=3 example)
distances = []
for cal, sugar, label in data:
    dist = math.sqrt((cal - new_point[0])**2 + (sugar - new_point[1])**2)
    distances.append((dist, cal, sugar, label))

# Sort by distance
distances.sort()

# Choose K = 3
k = 3
for i in range(k):
    _, cal, sugar, _ = distances[i]
    plt.plot([new_point[0], cal], [new_point[1], sugar])

# Labels and legend
plt.xlabel("Calories")
plt.ylabel("Sugar (g)")
plt.title("KNN Classification Graph")
plt.legend()

plt.show()