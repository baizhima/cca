# CCA preprocessing utilities
# Shan Lu (GitHub: baizhima), Renmin University of China
import os, sys, time, random
import pickle

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
	annotations, nr_images = readAnnotations()
	feature_file_dir = os.path.join(rootpath, collection, "FeatureData", feature)
	visual_feature, idx_mapping = buildVisualFeatures(feature_file_dir, nr_images)
	textual_feature = buildTextualFeatures(annotations, nr_images, idx_mapping)
	print "[buildFeatures] writing feature matrix into %s..."%(outputFile)
	with open(outputFile,'wb') as f:
		pickle.dump(idx_mapping, f)
		pickle.dump(visual_feature, f)
		pickle.dump(textual_feature, f)
	print 'Done'


def buildSampleFeatures(idx_mapping, visual_feature, textual_feature, outputFile='sample_feature.bin'):
	with open(outputFile,'wb') as f:
		pickle.dump(idx_mapping, f)
		pickle.dump(visual_feature, f)
		pickle.dump(textual_feature, f)

def readSampleFeatures(inputFile='sample_feature.bin'):
	with open(inputFile,'rb') as f:
		idx_mapping = pickle.load(f)
		visual_feature = pickle.load(f)
		textual_feature = pickle.load(f)
	return idx_mapping, visual_feature, textual_feature
	
def readFeatures(feature_file='feature.bin', sampleRatio=0.25):
	print "reading original features...."
	with open(feature_file,'rb') as f:
		idx_mapping = pickle.load(f)
		visual_feature = pickle.load(f)
		textual_feature = pickle.load(f)
	assert(len(visual_feature) == len(textual_feature))
	print "downsampling training visual and textual feature, sampleRatio=%f...."%(sampleRatio)
	random.seed(24)
	sample_size = int(len(visual_feature) * sampleRatio)
	decision = [True] * sample_size + [False] * (len(visual_feature) - sample_size)
	random.shuffle(decision)
	sample_visual_feature = [visual_feature[i] for i in range(len(visual_feature)) if decision[i]]
	sample_textual_feature = [textual_feature[i] for i in range(len(textual_feature)) if decision[i]]
	print "rearranging sample idx_mapping..."
	idx_mapping_kv_lst = [(k,v) for (k,v) in idx_mapping.items()]
	idx_mapping_kv_lst.sort(key=lambda x:x[1]) # sort by idx in feature matrix
	idx_mapping_lst = [v1 for (v1,v2) in idx_mapping_kv_lst]
	sample_idx_mapping_lst = [idx_mapping_lst[i] for i in range(len(idx_mapping_lst)) if decision[i]]
	sample_idx_mapping = {sample_idx_mapping_lst[i]:i for i in range(len(sample_idx_mapping_lst))}
	assert(len(sample_visual_feature) == len(sample_textual_feature))
	assert(len(sample_idx_mapping) == len(sample_visual_feature))
	return sample_idx_mapping, sample_visual_feature, sample_textual_feature



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



def buildTextualFeatures(annotations, nr_images, idx_mapping):
	feature_matrix = [None for i in range(nr_images)]
	assert(nr_images == len(annotations))
	count = 0
	for (k,v) in annotations.items():
	    if count % 1000 == 0:
	        print "%d images' textual feature added..."%(count)
	    idx = idx_mapping[k]
	    feature_matrix[idx] = v
	    count += 1
	return feature_matrix
	



