# covariance implementation for large feature matrices
# Shan Lu, Renmin University of China

import numpy as np
import math


# X is a n*d feature matrix, where each row represents an observation 
# and each column represents a feature dimention


def cov(X, verbose=False):
	if X is None:
		print '[covarianceCCA]X is null. Error unexpectedly.'
		return
	n = len(X)
	d = len(X[0])
	colSum = map(sum,zip(*X)) # mu[j] j = 1,2,...,d
	mu = [elem/float(n) for elem in colSum]
	#print mu
	covMatrix = [[0 for j in range(d)] for i in range(d)]
	for i in range(d):
		for j in range(0,i+1):
			col_i = [X[k][i] for k in range(n)]
			col_j = [X[k][j] for k in range(n)]
			#print col_i, col_j
			covMatrix[i][j] = sum([(col_i[k] - mu[i])*(col_j[k] - mu[j]) for k in range(n)])/float(n-1)
			covMatrix[j][i] = covMatrix[i][j]
		if i%d == 2000 and verbose:
			print '%d dimensions computed, %f work done'%(i, i/float(d))
	return np.array(covMatrix)


def saveCovMatrix(X, outputFile='bin/feature_cov.npy'):
	print '[saveCovMatrix] Saving feature covariance matrix into %s...'%outputFile
	np.save(outputFile, X)
	print 'Done'

def loadCovMatrix(inputFile='bin/feature_cov.npy'):
	print '[loadCovMatrix] Loading feature covariance matrix from %s'%(inputFile)
	covMatrix = np.load(inputFile)
	print 'Done'
	return covMatrix



if __name__ == '__main__':
	A = [[-2.1,3],[-1,1.1],[4.3,0.12]]

	covMat = cov(A)
	saveCovMatrix(covMat)
	covMat = loadCovMatrix('feature_cov.npy')
	print covMat

