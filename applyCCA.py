# test CCA model
# Shan Lu, Renmin University of China
import pickle
import numpy as np
import math

import utilCCA
from common import ROOT_PATH

class ccaModel:
	def __init__(self, collection, feature, path=ROOT_PATH, modelfile='cca.model'):
		
		self.idx_mapping, self.nviews, self.W, self.D = load_cca_model(modelfile)
		self.concepts = utilCCA.buildConcepts(collection, feature, path)
	
	def load_cca_model(self, modelfile):
		print "reading CCA model from binary files..."
		with open(modelfile,'rb') as f:
		    idx_mapping = pickle.load(f)
		    nviews = pickle.load(f)
		    W1, W2, W3 = pickle.load(f) 
		    D1, D2, D3 = pickle.load(f) 
		print "done"
		return idx_mapping, nviews, [W1,W2,W3],[D1,D2,D3]



	def get_top_n_concepts(self, img_id, top_n = 5):
		ranking = [(self.concepts[i], compute_similarity(img_id,1,self.concepts[i],2)) for i in range(len(self.concepts))]
		ranking.sort(key=lambda x:x[1], reverse=True)
		topConcepts = [ranking[i][0] for i in range(top_n)]
		return topConcepts

	# sim(x,y) = ((phi_x * W1[i] * D1[i]) * (phi_y * W2[i] * D2[i]).T) / (sqrt((phi_x * W1[i] * D1[i]).^2) * sqrt((phi_y * W2[i] * D2[i]).T).^2))
	def compute_similarity(self, x, view1, y, view2, pwr=4):
		assert(view1 >= 1 and view1 <= 3 and view2 >= 1 and view2 <= 3)
		W1, W2 = self.W[view1-1], self.W[view2-1]
		D1, D2 = self.D[view1-1], self.D[view2-1]
		x_feature, y_feature = get_feature(x,view1), get_feature(y,view2)
		phi_x = self.kernel_mapping(x_feature, view1)
		phi_y = self.kernel_mapping(y_feature, view2)
		return (phi_x.dot(W1).dot(D1)).dot((phi_y.dot(W2).dot(D2)).T) / (np.linalg.norm(phi_x.dot(W1).dot(D1)) * np.linalg.norm(phi_y.dot(W2).dot(D2)))

	def get_feature(x, viewNo):
		if viewNo == 1: # x is img_id, from img_id to dsift feature vector
			return 1
		elif viewNo == 2: # x is list of top_n tags
			return 



	def kernel_mapping(x, viewNo):
		if viewNo == 1: # Bhattacharyya kernel term-wise sqrt 
			return np.array([math.sqrt(elem) for elem in x])
		return x

	


        


if __name__ == '__main__':
	mymodel = ccaModel('flickr81train','dsift', ROOT_PATH,'cca.model')
	mymodel.compute_similarity('')
