import cPickle as pickle
import numpy as np
#from collections import defaultdict
import os,random


def save_feat(ark):
    print "preprocessing data for "+ark+"ing set"
    feat = pickle.load(open(ark,'r'))
    # context 414
    tmp = [feat[sent] for sent in feat.keys()]
    x = np.concatenate(tmp)
    np.random.shuffle(x)
    print "total frames : "+str(len(x))
    if ark=='train_sent':
        pickle.dump(x[:12800],open('dev','w'))
        pickle.dump(x[12801:],open('train','w'))
    else :
        pickle.dump(x,open('test','w'))

#save_feat('train_sent')
save_feat('test_sent')
#x = pickle.load(open('dev','r'))
#print len(x)
#for i in xrange(len(feat)):
    #print i
#    print feat[i][0]

#for sent in feat.keys():
#    print feat[sent].shape
    
#print feat.keys()


# pickle2
