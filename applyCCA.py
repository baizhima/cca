# test CCA model
# Shan Lu, Renmin University of China
import pickle
import numpy as np
import math
import os
import random

import utilCCA
import metric
from common import ROOT_PATH

class ccaModel:
	def __init__(self, collection, feature, path=ROOT_PATH, modelfile='cca.model'):
		print 'Initiating CCA model...'
		_, self.nviews, self.W, self.D = self.load_cca_model(modelfile)
		self.concepts = utilCCA.buildConcepts(collection, feature, path, 'concepts81test.txt')
		self.concept_mapping = {self.concepts[i]:i for i in range(len(self.concepts))}
		self.tagIdxMapping = {v:k for (k,v) in utilCCA.getTagIdxMapping('freqtags.txt').items()}
		self.tagWeights = utilCCA.getTagWeights('freqtags.txt')
		assert(len(self.tagIdxMapping) == 1000)
	
	def load_cca_model(self, modelfile):
		print "[load_cca_model] reading CCA model from binary files..."
		with open(modelfile,'rb') as f:
		    idx_mapping = pickle.load(f)
		    nviews = pickle.load(f)
		    W1, W2, W3 = pickle.load(f) 
		    D1, D2, D3 = pickle.load(f) 
		print "Done"
		return idx_mapping, nviews, [W1,W2,W3],[D1,D2,D3]

	def load_test_data(self, dataDir = 'bin/test'): # sample_feature for easy purposes
		print '[load_test_data] Loading test set features(est. 1min)...'
		with open(os.path.join(dataDir, 'idx_info.bin'),'rb') as f:
			self.idx_mapping = pickle.load(f)
		self.visual_feature = np.load(os.path.join(dataDir,'visual_feature.npy'))
		self.textual_feature = np.load(os.path.join(dataDir,'textual_feature.npy'))
		if self.nviews == 3:
			self.semantic_feature = np.load(os.path.join(dataDir, 'semantic_feature.npy'))
		print 'Done'

	def maintain_board(self,board, top_n, descending=True):
		board.sort(key=lambda x:x[1], reverse=descending)
		if len(board) > top_n:
			board = board[0:top_n]
		return board

	def I2I(self, img_id, top_n = 100):
		assert(img_id in self.idx_mapping.keys())
		leadboard = list() # list of tuples (img_id, similarity)
		home_visual = self.visual_feature[self.idx_mapping[img_id]]
		count = 0
		for (k,v) in self.idx_mapping.items():
			if k == img_id:continue
			if sum(self.semantic_feature[self.idx_mapping[k]]) == 0: continue # ignore nonsense img
			curr_visual = self.visual_feature[self.idx_mapping[k]]

			similarity = self.compute_similarity(home_visual, 1, curr_visual, 1)
			#similarity = np.linalg.norm(home_visual-curr_visual) # dsift original
			leadboard.append((k,similarity))
			count += 1
			if count % 10000 == 0:
				print '[I2I] %d images have been compared with %s'%(count, img_id)
			if len(leadboard) % 200 == 0:
				leadboard = self.maintain_board(leadboard, top_n)
		leadboard = self.maintain_board(leadboard, top_n)
		return leadboard

	def knn(self, img_id, top_n=100):
		assert(img_id in self.idx_mapping.keys())
		leadboard = [(None, 1e20) for i in range(top_n)] # list of tuples (img_id, similarity)
		home_visual = self.visual_feature[self.idx_mapping[img_id]]
		count = 0
		for (k,v) in self.idx_mapping.items():
			if k == img_id:continue
			if sum(self.semantic_feature[self.idx_mapping[k]]) == 0: continue # ignore nonsense img
			curr_visual = self.visual_feature[self.idx_mapping[k]]

			similarity = np.linalg.norm(home_visual-curr_visual)
			leadboard.append((k,similarity))
			count += 1
			if count % 10000 == 0:
				print '[knn] %d images have been compared with %s'%(count, img_id)
			if len(leadboard) % 200 == 0:
				leadboard = self.maintain_board(leadboard, top_n, descending=False)
		leadboard = self.maintain_board(leadboard, top_n, descending=False)
		return leadboard


	def T2I_old(self, tags, top_n = 100):
		leadboard = list() # list of tuples (img_id, similarity)
		tagList = [-1 for i in range(len(self.tagIdxMapping))]
		for tag in tags:
			if tag in self.tagIdxMapping.keys():
				tagList[self.tagIdxMapping[tag]] = 1
			
		home_textual = np.array(tagList)
		#print home_textual
		count = 0
		for (k,v) in self.idx_mapping.items():
			#if k == img_id:continue
			if sum(self.semantic_feature[self.idx_mapping[k]]) == 0: continue # ignore nonsense img
			curr_visual = self.visual_feature[self.idx_mapping[k]]
			similarity = self.compute_similarity(curr_visual, 1, home_textual, 2)
			leadboard.append((k,similarity))
			count += 1
			if count % 10000 == 0:
				print '[T2I]%d images have been compared'%(count)
			if len(leadboard) % 500 == 0:
				leadboard = self.maintain_board(leadboard, top_n)
		leadboard = self.maintain_board(leadboard, top_n)
		return leadboard

	def T2I(self, concept, top_n = 100):
		home_textual = np.array([0 for i in range(81)])
		for i in range(len(self.concepts)):
			if self.concepts[i] == concept:
				home_textual[i] = 1
		leadboard = list()
		for (k,v) in self.idx_mapping.items():
			curr_visual = self.visual_feature[self.idx_mapping[k]]
			if sum(self.semantic_feature[self.idx_mapping[k]]) == 0: continue
			similarity = self.compute_similarity(home_textual, 3, curr_visual, 1)
			leadboard.append((k,similarity))
			if len(leadboard) % 200 == 0:
				leadboard = self.maintain_board(leadboard, top_n)
		leadboard = self.maintain_board(leadboard, top_n)
		return leadboard



	def I2T(self, img_id, threshold=0.1):
		'''
		return an ordered litst of tuple (concept, similarity)
		'''

		assert(img_id in self.idx_mapping.keys())
		retrieved = []
		home_visual = self.visual_feature[self.idx_mapping[img_id]]

		for i in range(len(self.concepts)):
			curr_semantic = np.array([0 for j in range(len(self.concepts))])
			curr_semantic[i] = 1
			similarity = self.compute_similarity(curr_semantic, 3, home_visual, 1)
			retrieved.append((self.concepts[i], similarity.real))
			
		retrieved.sort(key=lambda x:x[1], reverse=True)
		return retrieved






	def I2I_check_correctness(self, requested, topList):
		truth_concepts = self.get_ground_truth(requested)
		print "%s truth concepts: %s"%(requested, truth_concepts[0]),truth_concepts
		for (img_id, _) in topList:
			concepts = self.get_ground_truth(img_id)
			print "%s: "%img_id, concepts

	def T2I_check_correctness(self, requested, leadboard):
		print "requested concepts: ", requested
		for (img_id, _) in leadboard:
			concepts = self.get_ground_truth(img_id)
			print "%s: "%img_id, concepts


	def I2T_check_correctness(self, requested, retrieved):
		print 'retrived concepts with similarity:'
		for tup in retrieved:
			print tup[0],tup[1]
		print 'truth:',self.get_ground_truth(requested)
		
	


	def get_ground_truth(self, img_id):
		concepts = []
		curr_semantic = self.semantic_feature[self.idx_mapping[img_id]]
		for i in range(len(curr_semantic)):
			if curr_semantic[i] == 1:
				concepts.append(self.concepts[i])
		return concepts


	# return 1 if img_id has given concept else return 0
	def has_concept(self, img_id, concept):
		curr_semantic = self.semantic_feature[self.idx_mapping[img_id]]
		return curr_semantic[self.concept_mapping[concept]]

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
		elif viewNo == 2: # linear kernel function
			vec = x * self.tagWeights
			return vec / np.linalg.norm(vec)
		else:
			return x * 5

	def compute_I2I_AP(self, requested, leadboard, typeNo, top_k=0,):
		if typeNo == 1: # I2I
			truth_concepts = set(self.get_ground_truth(requested))
		elif typeNo == 2: # T2I
			truth_concepts = requested
		sorted_labels = []
		for (img_id, _) in leadboard:
			labels = self.get_ground_truth(img_id)
			hasLabel = False
			for label in labels:
				if label in truth_concepts:
					hasLabel = True
					break
			if hasLabel:
				sorted_labels.append(1)
			else:
				sorted_labels.append(-1)
		scorer =  metric.APScorer(top_k)
		ap_value = scorer.score(sorted_labels)
		print truth_concepts, scorer.name(), ap_value
		return ap_value

	def compute_T2I_AP(self, concept, leadboard):
		'''
		leadboard is a list of img_ids
		'''
		sorted_labels = []
		for (img_id, _) in leadboard:
			curr_concepts = set(self.get_ground_truth(img_id))
			if concept in curr_concepts:
				sorted_labels.append(1)
			else:
				sorted_labels.append(-1)
		scorer =  metric.APScorer(0)
		ap_value = scorer.score(sorted_labels)
		print concept, scorer.name(), ap_value
		return ap_value

	def compute_I2T_AP(self, requested, leadboard):
		'''
		leadboard is a list of (concept, similarity) descended by similarty
		'''
		truth_concepts = set(self.get_ground_truth(requested))
		sorted_labels = []
		for (concept, _) in leadboard:
			if concept in truth_concepts:
				sorted_labels.append(1)
			else:
				sorted_labels.append(-1)
		scorer =  metric.APScorer(0)
		ap_value = scorer.score(sorted_labels)
		print truth_concepts, scorer.name(), ap_value
		return ap_value




	def compute_AP(self, concept, top_k = 100, threshold=0.2):
		home_semantic = np.array([-1 for i in range(len(self.concepts))])
		home_semantic[self.concept_mapping[concept]] = 1

		test_img_ids = [(v,k) for (k,v) in self.idx_mapping.items()]
		test_img_ids.sort(key=lambda x:x[0])
		test_img_ids_ordered = [x[1] for x in test_img_ids]
		truths = [self.has_concept(test_img_ids_ordered[i],concept) for i in range(len(test_img_ids))]
		predictions = [0 for i in range(len(test_img_ids))]
		for i in range(len(test_img_ids)):
			curr_visual = self.visual_feature[i]
			predictions[i] = self.compute_similarity(home_semantic, 3, curr_visual, 1)
			if i % 20000 == 0:
				print '[compute_AP] %d images has been checked with concept %s'%(i,concept)
		labels = [(test_img_ids_ordered[i],predictions[i],truths[i]) for i in range(len(test_img_ids))]
		labels.sort(key=lambda x:x[1],reverse=True)
		print labels[0:10]
		diffs = [1 for i in range(len(labels))]
		for i in range(len(labels)):
			# labels[i][1]->prediction,labels[i][2]->truth
			prediction = labels[i][1]
			truth = labels[i][2]
			if prediction.real >= threshold and truth == 0:
				diffs[i] = -1
			if prediction.real < threshold and truth == 1:
				diffs[i] = -1
		print diffs[1:100]
		scorer =  metric.APScorer(top_k)
		ap_value = scorer.score(diffs)
		print concept, scorer.name(), ap_value
		return ap_value

	def get_test_sample(self, sampleSize=1000,seedNo=26):
		sample_ids = []
		n = len(self.visual_feature)
		random.seed(seedNo)
		decision = [True] * (sampleSize) + [False] * (n-sampleSize)
		random.shuffle(decision)
		for (img_id, row_idx) in self.idx_mapping.items():
		    if decision[row_idx] and len(self.get_ground_truth(img_id)) >= 2:
		        sample_ids.append(img_id)
		random.shuffle(sample_ids)
		return sample_ids[0:min(100, len(sample_ids))]				




if __name__ == '__main__':
	mymodel = ccaModel('flickr81test','dsift', ROOT_PATH,'cca.model')
	mymodel.load_test_data(dataDir='bin/test')

	
	
	'''
	print 'I2I search'
	#test_imgs = mymodel.get_test_sample(sampleSize=1000)
	img_aps = []
	#test_imgs = ['76593245','545176797','141451694','141784798'] #
	for test_img in test_imgs:
		print 'I2I test search on image No. %s'%test_img
		
		leadboard = mymodel.knn(test_img)
		mymodel.I2I_check_correctness(test_img,leadboard)
		curr_ap = mymodel.compute_I2I_AP(test_img, leadboard, 1)
		img_aps.append(curr_ap)
		#print 'obtaining available links...'
		#utilCCA.printImgLink(test_img)
	print "I2I mAP: ", float(sum(img_aps))/len(img_aps)
	'''

	
	
	print 'T2I test search'
	#test_tags = mymodel.get_test_sample(sampleSize=200,seedNo=15)
	T2I_aps = []
	for (i, concept) in enumerate(mymodel.concepts):
		leadboard = mymodel.T2I(concept)
		mymodel.T2I_check_correctness(concept, leadboard)
		curr_ap = mymodel.compute_T2I_AP(concept, leadboard)
		print concept, curr_ap
		T2I_aps.append(curr_ap)

	print "T2I mAP: ", float(sum(T2I_aps))/len(T2I_aps)
	
	

	'''
	I2T_aps = []
	test_imgs = mymodel.get_test_sample(sampleSize=200,seedNo=10)
	test_imgs = ['492970362','204616617','122033461','391002501','2108534954']
	for test_img in test_imgs:
		print 'I2T test search on image No. %s'%test_img
		leadboard = mymodel.I2T(test_img)
		curr_ap = mymodel.compute_I2T_AP(test_img, leadboard)
		I2T_aps.append(curr_ap)
		mymodel.I2T_check_correctness(test_img,leadboard)
		#utilCCA.printImgLink(test_img)
	print "I2T mAP: ", float(sum(I2T_aps))/len(I2T_aps)
	'''
	
