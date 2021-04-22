import numpy as np
from math import log2, ceil, log
import pandas as pd
import matplotlib.pyplot as pl


'''
D : Adjacency Matrix
A : Number of nodes in each groups
G : Mapping of nodes to cluster index

'''

def log_n(n):
	val = 0
	n = log2(n)
	while n>0:
		val = val + n
		n = log2(n)

	return val


def LOG(val):
	if val == 0:
		return 0
	else:
		if val <= 0:
			print(val)
			exit(0)
		return np.log2(val)


def calculateA(G, K):
	A = np.zeros((K))
	for k in G:
		A[k-1] += 1
	return A


def DescriptionCost(A):
	k = len(A)
	cost1 = log_n(k)
	cost2 = 0

	i = 0
	while i < k-1:
		j = i
		cost_sum = 0
		while j<k:
			cost_sum = cost_sum + A[j]
			j = j+1
		cost_sum += i+1-k
		cost2 = cost2 + ceil(LOG(cost_sum))
		i = i + 1

	cost3 = 0

	for i in range(k):
		for j in range(k):
			cost3 = cost3 + ceil(LOG((A[i]*A[j] + 1)))

	return cost1 + cost2 + cost3


def CodeCost(D, G, k):
	n = G.shape[0]
	cost = 0

	# Iterate for cluster ID
	for i in range(k):
		for j in range(k):
			total = 0
			ones = 0
			x = 0

			while x<n:
				while x<n and G[x] != i+1:
					x = x + 1
				if x == n:
					break

				y = 0
				while y<n:
					while y<n and G[y] != j+1:
						y = y + 1
					if y == n:
						break

					total = total + 1
					if D[x][y] == 1:
						ones = ones + 1

					y = y + 1
				x = x + 1

			if total > 0:
				P = ones / total

				if ones > 0 and ones < total:
					cost = cost - ones*np.log2(P) - (total - ones)*np.log2(1 - P)

	return cost


def TotalEncodingCost(D, A, G):
	description_cost = DescriptionCost(A)
	code_cost = CodeCost(D, G, len(A))
	return description_cost + code_cost


def GraphPartitioning(D, A, G):
	K = 3
	n = len(D)

	# Outer loop
	while True:
		t = 0
		
		cost = TotalEncodingCost(D, A, G)

		# Inner loop
		while True:
			cost_new = cost.copy()
			G_ = G_new.copy()
			Weight = np.zeros((K_new,K_new))

			for x in range(n):
				kx = G_new[x]-1
				for y in range(n):
					ky = G_new[y]-1
					Weight[kx][ky] += D[x][y]


			for x in range(n):
				curr_k = G_new[x]-1
				Xrow = np.zeros((K_new, 2))
				Xcol = np.zeros((K_new, 2))
				for itr in range(n):
					k = G_new[itr]-1
					Xrow[k][D[x][itr]] += 1
					Xcol[k][D[itr][x]] += 1

				arr = np.zeros((K_new))
				for i in range(K_new):
					AiAi = A_new[i]*A_new[i]
					AiAk = A_new[i]*A_new[curr_k]

					if D[x][x] == 1:
						arr[i] = LOG(Weight[i][curr_k]/AiAk) + LOG(Weight[curr_k][i]/AiAk) - LOG(Weight[i][i]/AiAi)
					else:
						arr[i] = LOG((1-Weight[i][curr_k])/AiAk) + LOG((1-Weight[curr_k][i])/AiAk) - LOG((1-Weight[i][i])/AiAi)

					for j in range(K_new):
						AiAj = A_new[i]*A_new[j]
						arr[i] -= Xrow[j][1]*LOG(Weight[i][j]/AiAj) + Xrow[j][0]*LOG((AiAj - Weight[i][j])/AiAj)
						arr[i] -= Xcol[j][1]*LOG(Weight[j][i]/AiAj) + Xcol[j][0]*LOG((AiAj - Weight[j][i])/AiAj)
				
				G_[x] = np.argmin(arr) + 1
			
			A_ = calculateA(G_, K_new)
			cost_ = TotalEncodingCost(D, A_, G_)
			if 0 in A_ or cost_new == cost_:
				return K, G, A
			G_new = G_
			A_new = A_
			cost_new = cost_

		return K, G, A
		break

	return K, G, A


def OutlierDetection(D, A, G):
	n = G.shape[0]
	k = len(A)
	diff = 0.5
	outlier =[]
	D_ = D.copy()
	k_group = {}

	for i in range(n):
		for j in range(n):
			if i == j or D[i][j] == 0:
				continue
			(x, y) = (G[i], G[j])
			if x == y:
				continue
			k_group[(x, y)] = (i, j)


	for key in k_group.keys():
		(x, y) = k_group[key]
		D_[x][y] -= 1
		cost_ = TotalEncodingCost(D_, A, G) - TotalEncodingCost(D, A, G)
		D_[x][y] += 1
		# print(key, cost_)
		if abs(cost_) > diff:
			outlier.append(key)

	for KEY in outlier:
		(X , Y) = KEY
		RemoveOutliers(D, G, X, Y)

	return D


def RemoveOutliers(D, G, x, y):
	n = G.shape[0]
	for i in range(n):
		if G[i] == x:
			for j in range(n):
				if G[j] == y and D[i][j] == 1:	
					D[i][j] -= 1
					D[j][i] -= 1


def Visualize(D):
	pl.figure()
	DD = []
	for i in range(len(D)):
		DD_row = []
		for j in range(len(D[i])):
			if(D[i][j] == 1):
				DD_row += [(0, 0, 0)]
			else:
				DD_row += [(1, 1, 1)]

		DD += [DD_row]
	tb = pl.table(cellColours=DD,  loc=(0, 0), cellLoc='center')
	tc = tb.properties()['children']
	for cell in tc:
		cell.set_height(1/2)
		cell.set_width(1/2)
		cell.set_linewidth(0)
    # cell.set_line_width(0)
	tb.scale(3/len(D), 3/len(D))
	ax = pl.gca()
	ax.axis('off')
	ax.set_xticks([])
	ax.set_yticks([])


def Transform(D, G):
	DD = D.copy()
	mmap = {}
	curr=0
	for i in range(len(D+1)):
		for j in range(len(D)):
			if G[j] == i:
				mmap[j]=curr
				curr+=1
	
	for i in range(len(D)):
		for j in range(len(D)):
			DD[mmap[i]][mmap[j]] = D[i][j]
	return DD
