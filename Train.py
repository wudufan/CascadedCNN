
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import h5py
# import caffe
import random


# In[3]:

from PublicFunctions import ReadFromCavaderData
from PublicFunctions import CadaverTrain


# In[ ]:

indices = [0,1] # regarding dataset L067 and L096
baseTrainDir = '/home/data0/dufan/MedicalCNNDenoising/Example/SampleTrain'
if not os.path.exists(baseTrainDir):
    os.makedirs(baseTrainDir)


# In[ ]:

# generate training patches
# read the dicom images
for i in indices:
    print 'Patching %d'%i
    imgh = ReadFromCavaderData.ReadFromLowdoseChallengeData(i, True, None, 
                                                            basePath='/home/data0/dufan/CT_images/')
    imgl = ReadFromCavaderData.ReadFromLowdoseChallengeData(i, False)
    
    #  the ouput are lists, make imgh the residue for residue learning
    for j in range(0, len(imgh)):
        imgh[j] = imgl[j] - imgh[j]  # do no make the wrong order
        
    # generate image patches
    imgPatches = CadaverTrain.PatchingImgs(imgl,imgh, nPatchesPerLayer=50)
    
    # random shuffle
    inds = range(0,imgPatches.shape[0])
    random.shuffle(inds)
    imgPatches = imgPatches[inds, ...]
    
    # write hd5f files
    with h5py.File(os.path.join(baseTrainDir, 'trainingData_%d.h5'%i), 'w') as f:
        f['data'] = imgPatches.astype(np.float32)
        f['label'] = np.zeros(imgPatches.shape[0], dtype=np.float32)
        f.close()

# write the trainingList.txt, not neccessary
with open(os.path.join(baseTrainDir, 'trainingList.txt'), 'w') as f:
    for i in indices:
        f.write(os.path.join(baseTrainDir, 'trainingData_%d.h5\n'%i)) # better use absolute path here
    f.close()


# In[ ]:

caffe.set_mode_gpu()
caffe.set_device(3)

depth = 5

# first round training
print 'Iteration 0'
trainPath = os.path.join(baseTrainDir, 'trainedNets')
curPath = os.path.join(trainPath, '0')
CadaverTrain.TrainLowDoseChallenge(indices, 5, curPath, max_iter=1000, 
                                  patchPath=baseTrainDir)


# In[ ]:

# the cascaded training
for i in range(1,4):
    print 'Iteration %d'%i
    curPath = os.path.join(trainPath, str(i))
    
    # make a list of all the past test prototxt files and caffemodel files
    prototxts = list()
    caffemodels = list()
    for j in range(0,i):
        prototxts.append(os.path.join(trainPath, str(j), 'DnCNN_Test.prototxt'))
        caffemodels.append(os.path.join(trainPath, str(j), 'DnCNN.caffemodel'))
    
    # denoise the training dataset 
    # 2 for dual layer input, remove 2 for single layer input
    CadaverTrain.GenSeqTrainingDataLowDoseChallenge2(indices, prototxts, caffemodels, curPath, 
                                                    nPatchesPerLayer=100, rTopLayers=0.05)
    
    print 'Training...',
    # again, channel =2 for dual layer input
    CadaverTrain.TrainLowDoseChallenge(indices, depth, curPath, 1000, nChannel=2,
                                       patchPath=curPath)
    print 'Done'

print 'All Done'


# In[ ]:



