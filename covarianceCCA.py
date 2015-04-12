# covariance implementation for large feature matrices
# Shan Lu, Renmin University of China

import numpy as np
import math


# X is a n*d feature matrix, where each row represents an observation 
# and each column represents a feature dimention


def cov(X, verbose=True):
	if X is None:
		print '[covarianceCCA]X is null. Error unexpectedly.'
		return
	n = X.shape[0]
	d = X.shape[1]
	print 'compute colSum...'
	colSum = map(sum,zip(*X)) # mu[j] j = 1,2,...,d
	print 'compute mu'
	mu = [elem/float(n) for elem in colSum]
	#print mu
	covMatrix = [[0 for j in range(d)] for j in range(d)]
	print 'cov here...'
	for i in range(d):
		col_i = [X[k][i] for k in range(n)]
		for j in range(0,i+1):
			print "(i,j)=(%d,%d)"%(i,j)	
			col_j = [X[k][j] for k in range(n)]
			#print col_i, col_j
			covMatrix[i][j] = sum([(col_i[k] - mu[i])*(col_j[k] - mu[j]) for k in range(n)])/float(n-1)
			covMatrix[j][i] = covMatrix[i][j]
		if i%d == 5 and verbose:
			print '%d dimensions computed, %f work done'%(i, i/float(d))
	return np.array(covMatrix)


def cov2(X, verbose=True):
	n = X.shape[0]
	d = X.shape[1]
	print 'compute mu...'
	mu = X.sum(axis=0)/float(n) # mu[j] j = 1,2,...,d
	
	covMatrix = np.zeros((d,d))
	X_minus_mu = X
	print 'compute X-mu...'
	for j in range(d):
		X_minus_mu[:,j] = X_minus_mu[:,j] - mu[j] * np.ones((n,))


	print 'cov here...'
	for i in range(d):
		col_i = X_minus_mu[:,i]
		for j in range(0,i+1):
			print "(i,j)=(%d,%d)"%(i,j)	
			col_j = X_minus_mu[:,j]
			#print col_i, col_j
			covMatrix[i][j] = sum(col_i * col_j)/float(n-1)
			covMatrix[j][i] = covMatrix[i][j]
		if i%d == 5 and verbose:
			print '%d dimensions computed, %f work done'%(i, i/float(d))
	return np.array(covMatrix)


def saveCovMatrix(X, outputFile='bin/feature_cov.npy'):
	print '[saveCovMatrix] Saving feature covariance matrix into %s...'%outputFile
	np.save(outputFile, X)
	print 'Done'





if __name__ == '__main__':
	X = np.array([[-2.1,3],[-1,1.1],[4.3,0.12]])
	covMat = cov2(X)
	saveCovMatrix(covMat)
	#covMat = loadCovMatrix('feature_cov.npy')
	print covMat

