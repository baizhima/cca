# multiview CCA

import numpy as np
import time
import scipy as Sci
import scipy.linalg


def multiviewCCA(X, index, reg):
    print "get covariance matrix 1..."
    s = time.time()
    C_all = np.cov(X)
    print "done. Time elapsed %f seconds"%(time.time() - s)
    C_diag = np.zeros(C_all.shape)
    print "get covariance matrix 2..."
    s = time.time()
    for i in range(1, max(index)+1):
        index_f = np.nonzero(index == i)[0].tolist() 
        C_diag[np.ix_(index_f,index_f)] = C_all[np.ix_(index_f,index_f)] + reg*np.eye(len(index_f))
        C_all[np.ix_(index_f,index_f)] = C_all[np.ix_(index_f,index_f)] + reg*np.eye(len(index_f))
    print "done. Time elapsed %f seconds"%(time.time()-s)
    s = time.time()
    print "start eigin decomposition..."
    [V,D] = Sci.linalg.eig(C_all, C_diag)
    print "done. Time elapsed %f seconds"%(time.time()-s)
    diagVal = [(D[i][i],i) for i in range(D.shape[0])]
    diagVal.sort(key=lambda x:x[0], reverse=True)
    a = [diagVal[i][0] for i in range(len(diagVal))]
    index = [diagVal[i][1] for i in range(len(diagVal))]
    D = np.diag(a)
    V2 = V.reshape(1,V.shape[0])
    V = V2[np.ix_([i for i in range(V2.shape[0])], [idx for idx in index])]
    
    return [V, D]



if __name__ == "__main__":
    sys.exit(main())