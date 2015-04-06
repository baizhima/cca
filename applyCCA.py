# test CCA model
# Shan Lu, Renmin University of China
import pickle
import numpy as np
import math

import utilCCA
from common import ROOT_PATH

class ccaModel:
	def __init__(self, collection, feature, path=ROOT_PATH, modelfile='cca.model'):
		print 'Initiating CCA model...'
		_, self.nviews, self.W, self.D = self.load_cca_model(modelfile)
		self.idx_mapping, self.visual_feature, self.textual_feature, self.semantic_feature = self.load_image_dataset()
		self.concepts = utilCCA.buildConcepts(collection, feature, path)
		self.tagIdxMapping = {v:k for (k,v) in utilCCA.getTagIdxMapping('freqtags.txt').items()}
		assert(len(self.tagIdxMapping) == 1000)
		print 'Done'
	
	def load_cca_model(self, modelfile):
		print "[load_cca_model] reading CCA model from binary files..."
		with open(modelfile,'rb') as f:
		    idx_mapping = pickle.load(f)
		    nviews = pickle.load(f)
		    W1, W2, W3 = pickle.load(f) 
		    D1, D2, D3 = pickle.load(f) 
		print "Done"
		return idx_mapping, nviews, [W1,W2,W3],[D1,D2,D3]

	def load_image_dataset(self, featureFile='sample_feature.bin'): # sample_feature for easy purposes
		print '[load_image_dataset] Loading candidate images...'
		return utilCCA.readSampleFeatures(featureFile) 

	def maintain_board(self,board, top_n):
		board.sort(key=lambda x:x[1], reverse=True)
		if len(board) > top_n:
			board = board[0:top_n]
		return board

	def I2I_get_top_n_images(self, img_id, top_n = 5):
		assert(img_id in self.idx_mapping.keys())
		leadboard = [(None, -1) for i in range(top_n)] # list of tuples (img_id, similarity)
		home_visual = self.visual_feature[self.idx_mapping[img_id]]
		count = 0
		for (k,v) in self.idx_mapping.items():
			if k == img_id:continue
			curr_visual = self.visual_feature[self.idx_mapping[k]]
			similarity = self.compute_similarity(home_visual, 1, curr_visual, 1)
			leadboard.append((k,similarity))
			count += 1
			if count % 2000 == 0:
				print '%d images have been compared with %s'%(count, img_id)
			if len(leadboard) % 100 == 0:
				leadboard = self.maintain_board(leadboard, top_n)
		leadboard = self.maintain_board(leadboard, top_n)
		return leadboard

	def T2I_get_top_n_images(self, tags, top_n = 5):
		leadboard = [(None, -1) for i in range(top_n)] # list of tuples (img_id, similarity)
		tagList = [0 for i in range(len(self.tagIdxMapping))]
		for tag in tags:
			if tag in self.tagIdxMapping.keys():
				tagList[self.tagIdxMapping[tag]] = 1
		home_textual = np.array(tagList)
		count = 0
		for (k,v) in self.idx_mapping.items():
			#if k == img_id:continue
			curr_visual = self.visual_feature[self.idx_mapping[k]]
			similarity = self.compute_similarity(curr_visual, 1, home_textual, 2)
			leadboard.append((k,similarity))
			count += 1
			if count % 2000 == 0:
				print '%d images have been compared'%(count)
			if len(leadboard) % 100 == 0:
				leadboard = self.maintain_board(leadboard, top_n)
		leadboard = self.maintain_board(leadboard, top_n)
		return leadboard




	def I2I_check_correctness(self, requested, topList):
		truth_concepts = self.get_ground_truth(requested)
		print "%s truth concepts: %s"%(requested, truth_concepts[0]),truth_concepts[1]
		for (img_id, _) in topList:
			concepts = self.get_ground_truth(img_id)
			print "%s: "%img_id, concepts[1]

	def T2I_check_correctness(self, requested, topList):
		print "requested concepts: ", requested
		for (img_id, _) in topList:
			concepts = self.get_ground_truth(img_id)
			print "%s: "%img_id, concepts[1]


	def get_ground_truth(self, img_id):
		concepts = []
		#print '[get_ground_truth] img_id: %s'%(img_id)
		curr_semantic = self.semantic_feature[self.idx_mapping[img_id]]
		for i in range(len(curr_semantic)):
			if curr_semantic[i] == 1:
				concepts.append(self.concepts[i])
		return (img_id, concepts)


	# sim(x,y) = ((phi_x * W1[i] * D1[i]) * (phi_y * W2[i] * D2[i]).T) / (sqrt((phi_x * W1[i] * D1[i]).^2) * sqrt((phi_y * W2[i] * D2[i]).T).^2))
	def compute_similarity(self, x, view1, y, view2, pwr=4):
		assert(view1 >= 1 and view1 <= 3 and view2 >= 1 and view2 <= 3)
		W1, W2 = self.W[view1-1], self.W[view2-1]
		D1, D2 = self.D[view1-1], self.D[view2-1]
		phi_x = self.kernel_mapping(x, view1)
		phi_y = self.kernel_mapping(y, view2)
		return (phi_x.dot(W1).dot(D1**pwr)).dot((phi_y.dot(W2).dot(D2**pwr)).T) / \
		(np.linalg.norm(phi_x.dot(W1).dot(D1**pwr)) * np.linalg.norm(phi_y.dot(W2).dot(D2**pwr)))


	def kernel_mapping(self, x, viewNo):
		if viewNo == 1: # Bhattacharyya kernel term-wise sqrt 
			return np.array([math.sqrt(elem) for elem in x])
		if viewNo == 2:
			norm = np.linalg.norm(x) # normalized vector(not mentioned in paper explicitly)
			#return x/norm
			return x



if __name__ == '__main__':
	mymodel = ccaModel('flickr81train','dsift', ROOT_PATH,'cca.model')
	
	print 'I2I test search'
	test_img = '201430039'
	topList = mymodel.I2I_get_top_n_images(test_img, 8)
	mymodel.I2I_check_correctness(test_img, topList)
	
	print 'T2I test search'
	test_tags = ['girl','black']
	topList = mymodel.T2I_get_top_n_images(test_tags, 8)
	mymodel.T2I_check_correctness(test_tags, topList)
	

	
 
	