import numpy as np
import pandas as pd
from metrics import TotalEncodingCost, GraphPartitioning, OutlierDetection, Visualize, Transform, ClusterDistance

# Import the Adjacency Matrix

inputExcel = pd.read_excel("Datasets/AdjacencyMatrix-1.xlsx", header=None)
D = inputExcel.to_numpy()

Visualize(D)
print("Initial Matrix:\n", D)
print("Adjacency matrix shape: ", D.shape)

A = np.array([len(D)])
G = np.ones((len(D))).astype(np.uint16)

print("Initial Total encoding cost = ",TotalEncodingCost(D, A, G))

# Process to map node to k-cluster

print("\n\n------------- Auto-Partitioning Graph -------------\n")
K, G_, A_ = GraphPartitioning(D, A, G)
Visualize(Transform(D, G_), 1)
print("After Auto-partitioning : ")
print("K = ", K)
print("Final Node to Cluster mapping", G_)
print("Elements in each cluster : ", A_)
print("Total encoding cost (after partitioning) = ", TotalEncodingCost(D, A_, G_))



# Outlier Detection

print("\n\n------------- Outlier Detection -------------\n")
D_ = D.copy()
D_ = OutlierDetection(D_, A_, G_)
outlier = []
n = G.shape[0]
for i in range(n):
	for j in range(n):
		if D_[i][j] == 0 and D[i][j] == 1:
			outlier.append((i, j))
		j = j + 1

if len(outlier) == 0:
	print("\nNo outlier detected!")

else:
	print("\nFollowing are the detected outliers\n")
	no = 1
	for edge in outlier:
		(x, y) = edge
		print("Edge %-6d : %6d    <-> %6d" % (no, x+1, y+1))
		no = no + 1
Visualize(Transform(D_, G_), 2)
print("Matrix after removing outliers:\n", D_)
print("Total encoding cost :")
print("Before removing outlier = ", TotalEncodingCost(D, A_, G_))
print("After removing outlier = ", TotalEncodingCost(D_, A_, G_))



# Cluster Distance

print("\n\n------------- Cluster Distance -------------\n")
distance = ClusterDistance(D, A_, G_)
print("Following is the matrix representing the Cluster distance")
print(distance)
print("\n")

df = pd.DataFrame (D_)
filepath = 'Datasets/FinalAdjacencyMatrix-1.xlsx'
df.to_excel(filepath, index=False, header = False)
