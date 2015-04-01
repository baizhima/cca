# Shan Lu's CCA implementation


import sys, os, time, pickle, random

#from basic.common import ROOT_PATH, checkToSkip, makedirsforfile, printStatus
#from basic.data import FEATURE_TO_DIM, COLLECTION_TO_CONCEPTSET
from bigfile import BigFile
from common import ROOT_PATH
from multiviewCCA import *
import utilCCA


INFO = 'cca.trainCCA'

# part 1, read in data

class ccaLearner:
    def __init__(self, collection, feature, rootpath=ROOT_PATH, sampleRatio = 0.25, featureFromRaw = False):
        self.W1, self.W2, self.W3 = None, None, None
        self.D1, self.D2, self.D3 = None, None, None
        if featureFromRaw:
            utilCCA.buildAnnotations(collection, feature, rootpath)
            utilCCA.buildFeatures(collection,feature,rootpath)
        #self.idx_mapping, self.visual_feature, self.textual_feature = utilCCA.readFeatures('feature.bin',sampleRatio)
        self.idx_mapping, self.visual_feature, self.textual_feature = utilCCA.readSampleFeatures('sample_feature.bin')
        print "training sample of size %d extracted from binary file, ready for cca training"%(len(self.idx_mapping))

        #test_feature_correctness("1149309055")

        

    

    def get_original_bigfile_feature(self, img_id):
        return self.train_feature_file.read_one(img_id)

    def get_visual_feature(self, requested, isImgId=True):
        if isImgId:
            return self.visual_feature[self.idx_mapping[requested]]
        else: # requested is matrix index(range from 0 to self.nr_of_images-1)
            return self.visual_feature[requested]

    def get_textual_feature(self, requested, isImgId=True):
        if isImgId:
            return self.textual_feature[self.idx_mapping[requested]]
        else: # requested is matrix index(range from 0 to self.nr_of_images-1)
            return self.textual_feature[requested]

    def get_id_mapping(self, img_id):
        return self.idx_mapping[img_id]

    def test_feature_correctness(self, img_id):
        print "This function test the correctness of feature metrix.."
        visual_row = self.visual_feature[self.idx_mapping[img_id]]
        textual_row = self.textual_feature[self.idx_mapping[img_id]]
        
        if visual_row != self.train_feature_file.read_one(img_id):
            print "[Error] %s visual feature mismatches!" %(img_id)
            return False
        
        if textual_row != self.annotations[img_id]:
            print "[Error] %s textual feature mismatches!" %(img_id)
            return False

        print "%s feature correctness guaranteed"%(img_id)
        return True




    def cca_training(self, nviews=2):
    	if nviews == 2:
            print "training a 2-view CCA model."
            print " Loading textual and visual feature into np arrays(est. 3min)..."
            #idx_mapping, visual_feature, textual_feature = utilCCA.readFeatures()
            X = np.array(self.visual_feature)
            T = np.array(self.textual_feature)
            self.visual_feature, self.textual_feature = None, None
            assert(T.shape[0] == X.shape[0])
            XX = np.concatenate((X,T),axis=1)
            # rearrange XX to a square matrix
            decision = [True] * XX.shape[1] + [False] * (XX.shape[0] - XX.shape[1])
            random.shuffle(decision)
            XX = XX[np.ix_([i for i in range(XX.shape[0]) if decision[i]],[i for i in range(XX.shape[1])])]
            print "done. XX dimension (%d,%d)"%(XX.shape[0],XX.shape[1])
            view1 = np.ones((X.shape[1],1))
            view2 = 2 * np.ones((T.shape[1],1))
            index = np.concatenate((view1, view2))
            [V,D] = multiviewCCA(XX, index, 0.0001)
            Wx = V
            index_f1 = np.nonzero(index == 1)[0].tolist() 
            index_f2 = np.nonzero(index == 2)[0].tolist() 
            W1,W2,W3 = Wx[np.ix_(index_f1,index_f1)],Wx[np.ix_(index_f2,index_f2)], None
            D1,D2,D3 = D[np.ix_(index_f1,index_f1)],D[np.ix_(index_f2,index_f2)], None
            
            return [W1,W2,W3,D1,D2,D3]
        else:
            print "has not implemented 3-view cca"



    def compute_similarity(self, x, y):
        x_visual, x_textual = self.get_visual_feature(x), self.get_textual_feature(x)
        y_visual, y_textual = self.get_visual_feature(y), self.get_textual_feature(y)


    def save_cca_model(self):
        print "writing CCA model into binary files...",
        with open('cca.model','wb') as f:
            pickle.dump(self.idx_mapping, f)
            pickle.dump(self.sample_size, f)
            pickle.dump([self.W1, self.W2, self.W3], f)
            pickle.dump([self.D1, self.D2, self.D3], f)
    
    def load_cca_model(self, filename='cca.model'):
        print "reading CCA model from binary files...",
        with open(filename,'rb') as f:
            self.idx_mapping = pickle.load(f)
            self.sample_size = pickle.load(f)
            self.W1, self.W2, self.W3 = pickle.load(f)
            self.D1, self.D2, self.D3 = pickle.load(f)

        


if __name__ == "__main__":
    cca = ccaLearner('flickr81train','dsift')
    cca.cca_training()
    cca.save_cca_model()
    #cca.test_feature_correctness("1149309055")
    
