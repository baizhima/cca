# multiview CCA

import numpy as np
import time
import scipy as Sci
import scipy.linalg


def multiviewCCA(C_all, index, reg):
    
    C_diag = np.zeros(C_all.shape)
    print "get covariance matrix 2..."
    s = time.time()
    for i in range(1, max(index)+1):
        index_f = np.nonzero(index == i)[0].tolist() 
        C_diag[np.ix_(index_f,index_f)] = C_all[np.ix_(index_f,index_f)] + reg*np.eye(len(index_f))
        C_all[np.ix_(index_f,index_f)] = C_all[np.ix_(index_f,index_f)] + reg*np.eye(len(index_f))
    print "done. Time elapsed %f seconds"%(time.time()-s)
    s = time.time()
    print "start eigin decomposition(est. 20min)..."
    # [D,V] = EIG(A,B) produces a diagonal matrix D of generalized
    # eigenvalues and a full matrix V whose columns are the corresponding
    # eigenvectors so that A*V = B*V*D.
    [D,V] = Sci.linalg.eig(C_all, C_diag)
    print "done. Time elapsed %f seconds"%(time.time()-s)
    diagVal = [(D[i],i) for i in range(V.shape[0])]
    diagVal.sort(key=lambda x:x[0], reverse=True)
    a = [diagVal[i][0] for i in range(len(diagVal))]
    index = [diagVal[i][1] for i in range(len(diagVal))]
    D = np.diag(a)  
    V2 = V[np.ix_([i for i in range(V.shape[0])], [idx for idx in index])]
    return [V2, D]



if __name__ == "__main__":
    sys.exit(main())