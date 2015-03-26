# Shan Lu's CCA implementation


import sys, os, time, pickle, random

#from basic.common import ROOT_PATH, checkToSkip, makedirsforfile, printStatus
#from basic.data import FEATURE_TO_DIM, COLLECTION_TO_CONCEPTSET
from bigfile import BigFile
from common import ROOT_PATH


INFO = 'cca.trainCCA'

# part 1, read in data

class ccaLearner:
    def __init__(self, collection, feature, rootpath=ROOT_PATH, readFromRaw=False, featureFromRaw = True):
        id_file = os.path.join(rootpath, collection, "FeatureData", feature, "id.txt")
        feature_file = os.path.join(rootpath, collection, "FeatureData", feature, "feature.bin")
        self.nr_of_images = len(open(id_file).readline().strip().split(' '))
        print 'nr_of_images', self.nr_of_images
        concept_file = os.path.join(rootpath, collection, "Annotations", "concepts81train.txt")
        self.concepts = [line.strip() for line in open(concept_file)]
        '''
          explanation of self.annotation
          dict of list of booleans in order of [airplane, airport, animal, beach, ...]
          annotations['img_no'] = [0,0,1,1,..] means img_no contains animal and beach
        '''
        if readFromRaw:
            self.annotations = self.read_annotations(collection, feature, rootpath, id_file, concept_file,len(self.concepts))  
            self.write_annotation_compact()    
        else:
            self.annotations = self.read_annotation_compact() # read tag data from image_tags.bin
        feature_file_dir = os.path.join(rootpath, collection, "FeatureData", "dsift")
        self.train_feature_file = BigFile(feature_file_dir)
        self.idx_mapping = self.train_feature_file.name2index
        
        if featureFromRaw:
            self.visual_feature = self.create_visual_feature_matrix()
            self.textual_feature = self.create_textual_feature_maxtrix()
            self.write_feature_compact()
        else:
            self.visual_feature, self.textual_feature = self.read_feature_compact()
        print  "finish creating feature matrices"    
        print "text correctness.."

        #print self.annotations[1149309055]

    def read_annotations(self, collection, feature, rootpath, id_file, concept_file, concepts_count):
        # make a table of size nr_of_images * concepts_count
        img_annotations_dict = dict() # img_annotations_dict['img_no'] = [0, 1, ..]

        img_ids = [line for line in open(id_file).readline().strip().split(' ')]
        for img_id in img_ids:
            img_annotations_dict[img_id] = [False for i in range(concepts_count)]
        for i in range(concepts_count):
            concept = self.concepts[i]
            print "concept:",concept
            annotation_file = os.path.join(rootpath, collection, "Annotations","Image",\
                "concepts81train.txt",'%s.txt'%concept)
            for line in open(annotation_file):
                [img_id, hasConcept] = line.strip().split(' ')
                #img_id = int(img_id)
                hasConcept = int(hasConcept)
                if hasConcept == -1: hasConcept = 0
                #print "photoid:", img_id, " hasConcept:", hasConcept
                img_annotations_dict[img_id][i] = hasConcept

        return img_annotations_dict

    def write_annotation_compact(self):
        print "writing annotaion data into binary file...."
        with open('image_tags.bin','wb') as f:
            pickle.dump(self.annotations, f)

    def read_annotation_compact(self):
        print "reading annotaion data from binary file...."
        with open('image_tags.bin','rb') as f:
            annotations = pickle.load(f)
        return annotations
    
    def create_visual_feature_matrix(self):
        print "creating visual feature matrix..."
        feature_matrix = [None for i in range(self.nr_of_images)]
        print self.nr_of_images, len(self.train_feature_file.names)
        assert (self.nr_of_images == len(self.train_feature_file.names))
        count = 0
        for (k,v) in self.idx_mapping.items():
            if count % 1000 == 0:
                print "%d images' visual feature added..."%(count)
            feature_matrix[v] = self.train_feature_file.read_one(k) # k is raw img_id
            count += 1
        return feature_matrix
    
    def create_textual_feature_maxtrix(self):
        print "creating textual feature matrix..."
        feature_matrix = [None for i in range(len(self.train_feature_file.names))]
        assert(self.nr_of_images == len(self.annotations))
        count = 0
        for (k,v) in self.annotations.items():
            if count % 1000 == 0:
                print "%d images' textual feature added..."%(count)
            idx = self.idx_mapping[k]
            feature_matrix[idx] = v
            count += 1
        return feature_matrix

    def write_feature_compact(self):
        print "writing feature data into binary files...",
        with open('visual_feature.bin','wb') as f:
            pickle.dump(self.visual_feature, f)
        with open('textual_feature.bin','wb') as f:
            pickle.dump(self.textual_feature, f)
        print "done"

    def read_feature_compact(self):
        print "reading feature data from binary files...",
        s = time.time()
        with open('visual_feature.bin','rb') as f:
            visual_feature = pickle.load(f)
        with open('textual_feature.bin','rb') as f:
            textual_feature = pickle.load(f)
        print "done. Time eplased: %f seconds"%(time.time()-s)
        return visual_feature, textual_feature

    def get_visual_feature(self):
        #return self.X
        pass

    def get_textual_feature(self):
        pass

    def get_id_mapping(self, img_id):
        return self.idx_mapping[img_id]

    def test_feature_correctness(self, img_id):
        print "This function test the correctness of feature metrix.."
        idx = self.idx_mapping[img_id]
        visual_row = self.visual_feature[idx]
        textual_row = self.textual_feature[idx]
        if visual_row != self.train_feature_file.read([img_id],isname=True):
            print "[Error] %s visual feature mismatches!" %(img_id)
            return False
        if textual_row != self.annotations[img_id]:
            print "[Error] %s textual feature mismatches!" %(img_id)
            return False
        print "%s feature correctness guaranteed"%(img_id)
        return True




    def cca_training(self):
    	return 0


if __name__ == "__main__":
    cca = ccaLearner('flickr81train','dsift',readFromRaw=False,featureFromRaw=False)
    cca.test_feature_correctness("1149309055")
    

# part 2, build model

# part 3, validate model