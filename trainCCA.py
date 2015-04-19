# Shan Lu's CCA implementation


import sys, os, time, pickle, random
import numpy as np
#from basic.common import ROOT_PATH, checkToSkip, makedirsforfile, printStatus
#from basic.data import FEATURE_TO_DIM, COLLECTION_TO_CONCEPTSET
from bigfile import BigFile
from common import ROOT_PATH
from multiviewCCA import *
import utilCCA
import covarianceCCA


INFO = 'cca.trainCCA'

# part 1, read in data

class ccaLearner:
    def __init__(self, collection, feature):
        with open('bin/idx_info.bin','rb') as f:
            self.idx_mapping = pickle.load(f)
        print 'loading features(est.1min) from npy files...'
        self.visual_feature = np.load('bin/filtered_visual_feature.npy')
        self.textual_feature = np.load('bin/filtered_textual_feature.npy')
        self.semantic_feature = np.load('bin/filtered_semantic_feature.npy')
        print 'done'

    def save_sparsed_T(self):
        sparsed_T = self.svd(self.textual_feature)
        np.save('bin/sparsed_textual_feature.npy',sparsed_T)



    def svd(self, textual_feature):
        print "applying Singular Value Decomposition.."
        U, s, V = np.linalg.svd(textual_feature, full_matrices=False)
        return np.dot(U, np.diag(s))


    def test_sky_cov(self,X):
        sky_textual = X[:,1026]
        sky_semantic = X[:,2079]
        np.column_stack((sky_textual,sky_semantic))
        print 'sky correlation before computing:'
        print np.corrcoef(sky_textual, sky_semantic)[0][1]

    def compute_feature_cov(self, nviews=3,sampleRatio=0.02):
        self.nviews = nviews
        
        X = self.visual_feature
        T = self.textual_feature
        assert(T.shape[0] == X.shape[0])
        print "[compute_feature_cov]Concatenating features..."
        view1, view2 = np.ones((X.shape[1],1)), 2 * np.ones((T.shape[1],1)) 
        index = np.concatenate((view1, view2))
        XX = np.concatenate((X,T),axis=1)
        if nviews == 3:
            S = self.semantic_feature
            assert(S.shape[0] == T.shape[0])
            view3 = 3 * np.ones((S.shape[1],1))
            index = np.concatenate((index, view3))
            XX = np.concatenate((XX,S),axis=1)
        print 'Done. Starting computing covariance...'
        nr_used = int(len(self.visual_feature) * sampleRatio)
        random.seed(20)
        decision = [True] * (nr_used) + [False] * (len(self.visual_feature)-nr_used)
        random.shuffle(decision)
        if sampleRatio != 1:
            XX = XX[np.ix_([i for i in range(XX.shape[0]) if decision[i]],[i for i in range(XX.shape[1])])]
        self.test_sky_cov(XX)
        covMat = covarianceCCA.cov2(XX,verbose=True)
        covarianceCCA.saveCovMatrix(covMat)
        print 'Done computing and saving feature cov matrix'
        return covMat


    def cca_training(self, nviews=3):
        self.nviews = nviews
        view1 = np.ones((self.visual_feature.shape[1],1))
        view2 = 2 * np.ones((self.textual_feature.shape[1],1))
        index =  np.concatenate((view1, view2))
        if nviews == 3:
            view3 = 3 * np.ones((self.semantic_feature.shape[1],1))
            index = np.concatenate((index, view3))

        covMat = np.load('bin/feature_cov.npy')
        [V,D] = multiviewCCA(covMat, index, 0.0001)
        Wx = V
        index_f1 = np.nonzero(index == 1)[0].tolist() 
        index_f2 = np.nonzero(index == 2)[0].tolist()
        d = 81 # currently set as min(X.shape[1],T.shape[1],S.shape[1])
        W1,W2,W3 = Wx[np.ix_(index_f1,index_f1)],Wx[np.ix_(index_f2,index_f2)], None
        D1,D2,D3 = D[np.ix_(index_f1,index_f1)],D[np.ix_(index_f2,index_f2)], None
        W1 = W1[np.ix_([i for i in range(W1.shape[0])],[i for i in range(d)])]
        W2 = W2[np.ix_([i for i in range(W2.shape[0])],[i for i in range(d)])]
        D1 = D1[np.ix_([i for i in range(d)],[i for i in range(d)])]
        D2 = D2[np.ix_([i for i in range(d)],[i for i in range(d)])]
        if nviews == 3:
            index_f3 = np.nonzero(index == 3)[0].tolist()
            W3 = Wx[np.ix_(index_f3,index_f3)]
            W3 = W3[np.ix_([i for i in range(W3.shape[0])],[i for i in range(d)])]
            D3 = D[np.ix_(index_f3,index_f3)]
            D3 = D3[np.ix_([i for i in range(d)],[i for i in range(d)])]

        print "done training model matrices W and D"
        self.W1, self.W2, self.W3 = W1, W2, W3
        self.D1, self.D2, self.D3 = D1, D2, D3
        return [W1,W2,W3,D1,D2,D3]  
        

    def save_cca_model(self):
        print "[save_cca_model]writing CCA model into binary files...",
        with open('cca.model','wb') as f:
            pickle.dump(self.idx_mapping, f)
            pickle.dump(self.nviews, f)
            pickle.dump([self.W1, self.W2, self.W3], f)
            pickle.dump([self.D1, self.D2, self.D3], f)
        print 'Done'
    
        


if __name__ == "__main__":
    mycca = ccaLearner('flickr81train','dsift')
    #mycca.compute_feature_cov()
    
    mycca.cca_training(3)
    mycca.save_cca_model()
    
    
