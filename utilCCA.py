# CCA preprocessing utilities
# Shan Lu (GitHub: baizhima), Renmin University of China
import os, sys, time, random
import pickle
import numpy as np

from bigfile import BigFile


'''
Function: buildConcepts
Usage: read concepts from raw txt files
Return value: list of str(concepts)
'''

def buildConcepts(collection, feature, rootpath):
	concept_file = os.path.join(rootpath, collection, "Annotations", "concepts81train.txt")
	return [line.strip() for line in open(concept_file)]


'''
Function: buildAnnotations
read annotation from raw txt files
save img_annotations_dict and nr_images into a binary file for later purposes
'''

def buildAnnotations(collection, feature, rootpath, outputFile="image_tags.bin"):
	id_file = os.path.join(rootpath, collection, "FeatureData", feature, "id.txt")
	concepts = buildConcepts(collection, feature, rootpath)
	'''
	  explanation of img_annotations_dict
	  dict of list of booleans in order of [airplane, airport, animal, beach, ...]
	  img_annotations_dict['img_idx'] = [0,0,1,1,..] means img_idx contains animal and beach
	'''
	img_annotations_dict = dict() # img_annotations_dict['img_no'] = [0, 1, ..]

	img_ids = [line for line in open(id_file).readline().strip().split(' ')]
	nr_images = len(img_ids)
	for img_id in img_ids:
	    img_annotations_dict[img_id] = [False for i in range(len(concepts))]
	for i in range(len(concepts)):
	    concept = concepts[i]
	    print "concept:",concept
	    annotation_file = os.path.join(rootpath, collection, "Annotations","Image",\
	        "concepts81train.txt",'%s.txt'%concept)
	    for line in open(annotation_file):
	        [img_id, hasConcept] = line.strip().split(' ')
	        hasConcept = int(hasConcept)
	        if hasConcept == -1: hasConcept = 0
	        img_annotations_dict[img_id][i] = hasConcept
	print "writing varriables 1.annotations and 2.nr_images into binary file %s...."%outputFile,
	with open(outputFile,'wb') as f:
	    pickle.dump(img_annotations_dict, f)
	    pickle.dump(nr_images, f)
	with open("shape.txt","w") as f:
		f.write(str(nr_images))
	print "done"
	


def readAnnotations(filename='image_tags.bin'):
	print "reading annotaion data from binary file %s...."%(filename)
	with open(filename,'rb') as f:
	    annotations = pickle.load(f)
	    nr_images = pickle.load(f)
	return annotations, nr_images




'''
Function: buildFeatures
build visual and textual matrix
save idx_mapping and both matrices into a binary file for later purposes
 
'''

def buildFeatures(collection,feature,rootpath,outputFile='feature.bin'):
	annotations, nr_images = readAnnotations('image_tags.bin')
	feature_file_dir = os.path.join(rootpath, collection, "FeatureData", feature)
	visual_feature, idx_mapping = buildVisualFeatures(feature_file_dir, nr_images)
	tagDict = buildTagsDictionary(rootpath, collection, top_n = 1000)
	textual_feature = buildTextualFeatures(rootpath, collection, tagDict, nr_images, idx_mapping)
	semantic_feature = buildSemanticFeatures(annotations, idx_mapping)
	print "[buildFeatures] writing feature matrix into %s..."%(outputFile)
	with open(outputFile,'wb') as f:
		pickle.dump(idx_mapping, f)
		print '[buildFeatures] writing visual feature matrix...'
		pickle.dump(visual_feature, f)
		print '[buildFeatures] writing textual feature matrix...'
		pickle.dump(textual_feature, f)
		print '[buildFeatures] writing semantic feature matrix...'
		pickle.dump(semantic_feature, f)
	print 'Done'







def buildFilteredFeatures(idx_mapping, visual_feature, textual_feature, semantic_feature, outputFile='bin/filtered_feature.bin'):
	print "[buildFilteredFeatures] writing feature matrix into %s..."%(outputFile)
	with open(outputFile,'wb') as f:
		pickle.dump(idx_mapping, f)
		print '[buildFilteredFeatures] writing visual feature matrix...'
		pickle.dump(visual_feature, f)
		print '[buildFilteredFeatures] writing textual feature matrix...'
		pickle.dump(textual_feature, f)
		print '[buildFilteredFeatures] writing semantic feature matrix...'
		pickle.dump(semantic_feature, f)
	print 'Done'





def readFilteredFeatures(inputFile='bin/filtered_feature.bin'):
	print "[readFilteredFeatures] reading filtered features from %s(est. 20min)"%inputFile
	with open(inputFile,'rb') as f:
		idx_mapping = pickle.load(f)
		print '[buildreadFilteredFeatures] reading visual feature matrix...'
		visual_feature = pickle.load(f)
		print '[buildreadFilteredFeatures] reading textual feature matrix...'
		textual_feature = pickle.load(f)
		print '[buildreadFilteredFeatures] reading semantic feature matrix...'
		semantic_feature = pickle.load(f)
	print 'saving visual feature seperately into npy file...'
	np.save('bin/filtered_visual_feature.npy', np.array(visual_feature))
	print 'saving textualfeature seperately into npy file...'
	np.save('bin/filtered_textual_feature.npy', np.array(textual_feature))
	print 'saving semanticfeature seperately into npy file...'
	np.save('bin/filtered_semantic_feature.npy', np.array(semantic_feature))
	print 'Done'
	with open('bin/idx_info.bin','wb') as f:
		pickle.dump(idx_mapping, f)
	print 'Done'


	print 'filtered training set size: %d'%len(idx_mapping)
	return idx_mapping, visual_feature, textual_feature, semantic_feature
	
def readFeatures(feature_file='feature.bin'):
	print "reading original features from %s(est. ~20mins)...."%feature_file
	with open(feature_file,'rb') as f:
		idx_mapping = pickle.load(f)
		visual_feature = pickle.load(f)
		textual_feature = pickle.load(f)
		semantic_feature = pickle.load(f)

	assert(len(visual_feature) == len(textual_feature))
	print "downsampling training visual and textual feature, filtered by occurence of freq tags"
	decision = [True] * len(visual_feature) 
	for i in range(len(visual_feature)):
		if sum(textual_feature[i]) == 0: # not suitable to be a training example
			decision[i] = False
	sample_visual_feature = [visual_feature[i] for i in range(len(visual_feature)) if decision[i]]
	sample_textual_feature = [textual_feature[i] for i in range(len(textual_feature)) if decision[i]]
	sample_semantic_feature = [semantic_feature[i] for i in range(len(semantic_feature)) if decision[i]]
	print "rearranging sample idx_mapping..."
	idx_mapping_kv_lst = [(k,v) for (k,v) in idx_mapping.items()]
	idx_mapping_kv_lst.sort(key=lambda x:x[1]) # sort by idx in feature matrix
	idx_mapping_lst = [v1 for (v1,v2) in idx_mapping_kv_lst]
	sample_idx_mapping_lst = [idx_mapping_lst[i] for i in range(len(idx_mapping_lst)) if decision[i]]
	sample_idx_mapping = {sample_idx_mapping_lst[i]:i for i in range(len(sample_idx_mapping_lst))}
	assert(len(sample_visual_feature) == len(sample_textual_feature))
	assert(len(sample_idx_mapping) == len(sample_visual_feature))
	assert(len(sample_visual_feature) == len(sample_semantic_feature))
	buildFilteredFeatures(sample_idx_mapping, sample_visual_feature, sample_textual_feature, sample_semantic_feature)
	return sample_idx_mapping, sample_visual_feature, sample_textual_feature, sample_semantic_feature



def buildVisualFeatures(feature_file_dir, nr_images):
	print "[buildVisualFeatures] creating visual feature matrix by reading BigFile %s..."%feature_file_dir
	train_feature_file = BigFile(feature_file_dir)
	idx_mapping = train_feature_file.name2index
	feature_matrix = [None for i in range(nr_images)]
	assert (nr_images == len(train_feature_file.names))
	count = 0
	for (k,v) in idx_mapping.items():
	    if count % 1000 == 0:
	        print "%d images' visual feature added..."%(count)
	    feature_matrix[v] = train_feature_file.read_one(k) # k is raw img_id
	    count += 1
	return feature_matrix, idx_mapping


def buildTextualFeatures(rootpath, collection, tagDict, nr_images, idx_mapping):
	print "[buildTextualFeatures] creating tag features"
	tagFile = os.path.join(rootpath,collection, "TextData", "id.userid.lemmtags.txt")
	feature_matrix = [[0 for j in range(len(tagDict))] for i in range(nr_images)]
	with open(tagFile) as f:
		img_count = sum(1 for _ in f)
	assert(img_count == nr_images)
	count = 0
	for line in open(tagFile):
		if count % 1000 == 0:
			print "%d images tag features added"%count
		columns = line.strip('\n').split('\t')
		img_id = columns[0]
		tags = columns[2].split(' ')
		for tag in tags:
			tag = tag.strip('\r')
			if tag in tagDict.keys():
				feature_matrix[idx_mapping[img_id]][tagDict[tag]] = 1
		count += 1
	return feature_matrix



def buildSemanticFeatures(annotations, idx_mapping):
	nr_images = len(idx_mapping)
	feature_matrix = [None for i in range(nr_images)]
	for (k,v) in annotations.items():
		feature_matrix[idx_mapping[k]] = v
	return feature_matrix

# build a dictionary of top N frequent words from lemmtag file	

def buildTagsDictionary(rootpath, collection, top_n = 1000):
	tagFile = os.path.join(rootpath,collection, "TextData", "id.userid.lemmtags.txt")
	tagDict = dict()
	for line in open(tagFile):
		columns = line.strip('\n').split('\t')
		tags = columns[2].split(' ')
		for tag in tags:
			tag = tag.strip('\r')
			if tag not in tagDict:
				tagDict[tag] = 1
			else:
				tagDict[tag] += 1
	print "%d tags in total have been added to full dictionary"%len(tagDict)
	tagList = [(k,v) for (k,v) in tagDict.items()]
	tagList.sort(key=lambda x:x[1], reverse=True)
	return {tagList[i][0]:i for i in range(min(top_n, len(tagList)))}
	

def buildFreqTags(rootpath, collection, top_n = 1000, outputFile = "freqtags.txt"):
	print "collecting top %d frequent tags and saving into %s..."%(top_n, outputFile)
	tagFile = os.path.join(rootpath,collection, "TextData", "id.userid.lemmtags.txt")
	tagDict = dict()
	for line in open(tagFile):
		columns = line.strip('\n').split('\t')
		tags = columns[2].split(' ')
		for tag in tags:
			tag = tag.strip('\r')
			if tag not in tagDict:
				tagDict[tag] = 1
			else:
				tagDict[tag] += 1
	
	tagList = [(k,v) for (k,v) in tagDict.items()]
	tagList.sort(key=lambda x:x[1], reverse=True)
	with open(outputFile,"w") as f:
		for i in range(min(top_n, len(tagList))):
			f.write("%d %s %d\n"%(i,tagList[i][0],tagList[i][1]))
	print "Done"

def readFreqTags(inputFile, top_n = 1000):
	tagDict = dict()
	for line in open(inputFile,'r'):
		curr = line.strip('\n').split(' ')
		tagDict[curr[1]] = curr[0]
	assert(top_n <= len(tagDict))
	return tagDict


def getTagIdxMapping(inputFile='freqtags.txt'):
	tagIdxMapping = dict()
	for line in open(inputFile,'r'):
		curr = line.strip('\n').split(' ')
		#print curr[0],curr[1]
		tagIdxMapping[int(curr[0])] = curr[1]

	return tagIdxMapping



				
if __name__ == '__main__':
	readFilteredFeatures()



	




