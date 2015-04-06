# Shan Lu's CCA implementation


import sys, os, time, pickle, random
import numpy as np
#from basic.common import ROOT_PATH, checkToSkip, makedirsforfile, printStatus
#from basic.data import FEATURE_TO_DIM, COLLECTION_TO_CONCEPTSET
from bigfile import BigFile
from common import ROOT_PATH
from multiviewCCA import *
import utilCCA


INFO = 'cca.trainCCA'

# part 1, read in data

class ccaLearner:
    def __init__(self, collection, feature, rootpath=ROOT_PATH, sampleRatio = 0.25, featureFromRaw = False, loadFromSample=False):
        self.W1, self.W2, self.W3 = None, None, None
        self.D1, self.D2, self.D3 = None, None, None
        if featureFromRaw:
            #utilCCA.buildAnnotations(collection, feature, rootpath)
            utilCCA.buildFeatures(collection,feature,rootpath)
        
        if loadFromSample:
            self.idx_mapping, self.visual_feature, self.textual_feature, self.semantic_feature = utilCCA.readSampleFeatures('sample_feature.bin')
        else:
            self.idx_mapping, self.visual_feature, self.textual_feature, self.semantic_feature  = utilCCA.readFeatures('feature.bin',0.25)

    

    def get_img_tags(self, requested, isImgId=True, freqTagFile='freqtags.txt'):
        tagIdxMapping = utilCCA.getTagIdxMapping(freqTagFile)
        tags = []
        if isImgId:
            currFeature = self.textual_feature[self.idx_mapping[requested]]
        else:
            currFeature = self.textual_feature[requested]
        for i in range(len(currFeature)):
            if currFeature[i]==1:
                tags.append(tagIdxMapping[i])  
        return tags  
    

    def svd(self, textual_feature):
        print "applying Singular Value Decomposition.."
        U, s, V = np.linalg.svd(textual_feature, full_matrices=False)
        return np.dot(U, np.diag(s))


    def cca_training(self, nviews=2, trainRatio = 0.3):
        self.nviews = nviews
        print "training a %d-view CCA model."%(nviews)
        print " Loading textual and visual feature into np arrays..."
        X = np.array(self.visual_feature)
        T = np.array(self.svd(self.textual_feature))
        assert(T.shape[0] == X.shape[0])
        #nr_train = int(X.shape[0]*trainRatio)
        nr_train = 2105
        assert(nr_train >= X.shape[1] + T.shape[1])
        print "The number of training examples: %d"%nr_train
        view1, view2 = np.ones((X.shape[1],1)), 2 * np.ones((T.shape[1],1)) 
        index = np.concatenate((view1, view2))
        XX = np.concatenate((X,T),axis=1)
        if nviews == 3:
            S = np.array(self.semantic_feature)
            assert(S.shape[0] == T.shape[0])
            assert(nr_train >= X.shape[1] + T.shape[1] + S.shape[1])
            view3 = 3 * np.ones((S.shape[1],1))
            index = np.concatenate((index, view3))
            XX = np.concatenate((XX,S),axis=1)
        decision = [True] * nr_train + [False] * (XX.shape[0] - nr_train)
        random.seed(52)
        random.shuffle(decision)
    	XX = XX[np.ix_([i for i in range(XX.shape[0]) if decision[i]],[i for i in range(XX.shape[1])])]
        print "done. XX dimension (%d,%d)"%(XX.shape[0],XX.shape[1])
        [V,D] = multiviewCCA(XX, index, 0.0001)
        Wx = V
        index_f1 = np.nonzero(index == 1)[0].tolist() 
        index_f2 = np.nonzero(index == 2)[0].tolist()
        d = 81 
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
        print "writing CCA model into binary files...",
        with open('cca.model','wb') as f:
            pickle.dump(self.idx_mapping, f)
            pickle.dump(self.nviews, f)
            pickle.dump([self.W1, self.W2, self.W3], f)
            pickle.dump([self.D1, self.D2, self.D3], f)
    
        


if __name__ == "__main__":
    cca = ccaLearner('flickr81train','dsift',featureFromRaw=False,loadFromSample=True)
    #utilCCA.buildSampleFeatures(cca.idx_mapping, cca.visual_feature, cca.textual_feature, cca.semantic_feature)
    
    #print 'get 2668663226 tag feature...\nexpected:cat kitty tabby lynx jasmine ragdoll creamcheeselover '
    #print cca.get_img_tags('2668663226')
    cca.cca_training(3, trainRatio = 0.08)
    cca.save_cca_model()
    #tagDict = utilCCA.buildTagsDictionary(ROOT_PATH, 'flickr81train', 1000)
    #cca.test_feature_correctness("1149309055")
    
