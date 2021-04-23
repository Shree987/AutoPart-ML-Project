import numpy as np
from math import log2, ceil, log
import copy
import matplotlib.pyplot as pl


def log_n(n):
	val = 0
	n = log2(n)
	while n > 0:
		val += n
		n = log2(n)
	return val

def LOG(val):
	if val == 0:
		return 0
	elif val < 0:
		print(val)
		exit(0)
	return np.log2(val)


def BinaryShannonEntropy(prob):
	if prob == 0 or prob == 1:
		return 0
	else:
		return -prob*np.log2(prob) - (1-prob)*np.log2(1-prob)


def DescriptionCost(A):
	K = len(A)

	A_ = A.copy()
	A_[::-1].sort()
	
	cost1 = log_n(K)
	cost2 = 0
	for i in range(K-1):
		cost_sum = 0
		for j in range(i, K):
			cost_sum += A[j]
		cost_sum -= K - i - 1
		cost2 = cost2 + ceil(LOG(cost_sum))

	cost3 = 0
	for i in range(K):
		for j in range(K):
			cost3 += ceil(LOG(A_[i]*A_[j] + 1))

	return cost1 + cost2 + cost3


def CodeCost(D, A, G):
	K = len(A)
	N = len(D)
	Dij = np.zeros((K, K))

	for i in range(N):
		for j in range(N):
			Dij[G[i]-1][G[j]-1] += D[i][j]

	cost = 0

	for i in range(K):
		for j in range(K):
			cost += (A[i]*A[j])*BinaryShannonEntropy(Dij[i][j] / (A[i]*A[j]))

	return cost


def TotalEncodingCost(D, A, G):
	description_cost = DescriptionCost(A)
	code_cost = CodeCost(D, A, G)
	return description_cost + code_cost


def calculateA(G, K):
	A = np.zeros((K))
	for k in G:
		A[k-1] += 1
	return A


def CalculateDij(D, G, i, j, x = -2):
	Dij, Dji = 0, 0
	N = len(D)
	for I  in range(N):
		if I == x:
			continue
		if G[I] == i:
			for J in range(N):
				if J == x:
					continue
				if J == j:
					Dij += D[I][J]

		if G[I] == j:
			for J in range(N):
				if J == x:
					continue
				if J == i:
					Dji += D[J][I]

	return Dij, Dji

def GraphPartitioning(D, A, G):
	K = 1
	N = len(D)
	cost = TotalEncodingCost(D, A, G)

	# Outer loop
	while True:
		A_new = A.copy()
		G_new = G.copy()
		K_new = K + 1

		Dij = np.zeros((K, K))
		for i in range(N):
			for j in range(N):
				Dij[G[i]-1][G[j]-1] += D[i][j]

		Entropy = np.zeros((K))

		for i in range(K):
			for j in range(K):
				AiAj = A_new[i]*A_new[j]
				Entropy[i] += A_new[j]*(BinaryShannonEntropy(Dij[i][j] / AiAj) + BinaryShannonEntropy(Dij[j][i] / AiAj))

		r = np.argmax(Entropy) + 1
		A_new = np.append(A_new, 0)
		for x in range(N):
			if G_new[x] != r:
				continue
			with_ = 0
			without_ = 0
			for j in range(K):
				Drj, Djr = CalculateDij(D, G_new, r, j+1)
				Arj = A_new[r-1]*A_new[j]
				with_ += BinaryShannonEntropy(Drj / Arj) + BinaryShannonEntropy(Djr / Arj)

				Drj, Djr = CalculateDij(D, G_new, r, j+1, x = x)
				Arj = (A_new[r-1] - 1)*A_new[j]
				if Arj > 0:
					without_ += BinaryShannonEntropy(Drj / Arj) + BinaryShannonEntropy(Djr / Arj)

			if with_ > without_:
				G_new[x] = K+1
				A_new[r-1] -= 1
				A_new[K] += 1

		if 0 in A_new:
			break

		# Inner loop
		while True:
			cost_new = copy.copy(cost)
			G_ = G_new.copy()
			A_ = A_new.copy()
			Weight = np.zeros((K_new,K_new))
			AiAj = np.zeros((K_new, K_new))
			for i in range(K_new):
				for j in range(K_new):
					AiAj[i][j] = A_[i]*A_[j]

			for x in range(N):
				kx = G_new[x]-1
				for y in range(N):
					ky = G_new[y]-1
					Weight[kx][ky] += D[x][y]

			for x in range(N):
				curr_k = G_new[x]-1
				Xrow = np.zeros((K_new, 2))
				Xcol = np.zeros((K_new, 2))
				for itr in range(N):
					k = G_new[itr]-1
					Xrow[k][D[x][itr]] += 1
					Xcol[k][D[itr][x]] += 1

				arr = np.zeros((K_new))
				for i in range(K_new):
					if A_[i]>0 and A_[curr_k]>0:
						PiPk = Weight[i][curr_k]/(A_[i]*A_[curr_k])
						PkPi = Weight[curr_k][i]/(A_[i]*A_[curr_k])
						PiPi = Weight[i][i]/A_[i]*A_[i]

						if D[x][x] == 1:
							arr[i] = LOG(PiPk) + LOG(PkPi) - LOG(PiPi)
						else:
							arr[i] = LOG(1-PiPk) + LOG(1-PkPi) - LOG(1-PiPi)

					for j in range(K_new):
						AiAj = A_[i]*A_[j]
						if AiAj > 0:				
							arr[i] -= Xrow[j][1]*LOG(Weight[i][j]/AiAj) + Xrow[j][0]*LOG((AiAj - Weight[i][j])/AiAj)
							arr[i] -= Xcol[j][1]*LOG(Weight[j][i]/AiAj) + Xcol[j][0]*LOG((AiAj - Weight[j][i])/AiAj)
				
				G_[x] = np.argmin(arr) + 1

			if np.sum((G_ - G_new)**2) == 0:
				break

			A_ = calculateA(G_, K_new)
			if 0 in A_:
				break
			cost_ = TotalEncodingCost(D, A_, G_)

			if cost_new < cost_:
				break
			elif cost_new == cost_:
				G_new = G_.copy()
				A_new = A_.copy()
				cost_new = cost_
				break				

			G_new = G_.copy()
			A_new = A_.copy()
			cost_new = cost_

		if 0 in A_new:
			return K, G, A

		cost_new = TotalEncodingCost(D, A_new, G_new)
		if cost == cost_new:
			return K, G, A

		K = K + 1
		A = A_new.copy()
		G = G_new.copy()
		cost = copy.copy(cost_new)

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


def ClusterDistance(D, A, G):
	n = len(D)
	K = len(A)

	cost_orig = TotalEncodingCost(D, A, G)

	distance = np.zeros((K, K))
	for i in range(K-1):
		A_ = np.zeros((K-1))
		for j in range(i+1, K):
			for a in range(K-1):
				A_[a] = A[a]
			A_[i] += A[j]
			if j!= K-1:
				A_[j] = A[K-1]

			G_ = G.copy()
			G_ = np.where(G_ == j+1, i+1, G_)
			G_ = np.where(G_ == K, j+1, G_)
			cost_ = TotalEncodingCost(D, A_, G_)

			distance[i][j] = distance[j][i] = (cost_ - cost_orig)/cost_orig

	return distance



def Visualize(D, index = 0):
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
	if index == 0:
		pl.savefig('Initial Matrix.png')
	elif index == 1:
		pl.savefig('Initial Matrix after permuting.png')
	elif index == 2:
		pl.savefig('Matrix removing outliers.png')



def Transform(D, G):
	n = len(D)
	DD = D.copy()
	mmap = {}
	curr=0
	for i in range(n):
		for j in range(n):
			if G[j] == i+1:
				mmap[j]=curr
				curr+=1
	
	for i in range(n):
		for j in range(n):
			DD[mmap[i]][mmap[j]] = D[i][j]

	return DD

