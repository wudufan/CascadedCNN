
# coding: utf-8

# In[1]:

import caffe
import numpy as np
import ReadFromCavaderData
import NNDenoise
import GenerateNetwork
import os
import h5py
import sys
import random
import copy


# In[1]:

# make patched images
# src - the noisy input image list 
# refs - the residue image list (noisy - noiseless)
# indices - indicate wich layers to be included in the patching, None for all layers
# nPatchesPerLayer - number of patches per layer
# patchSize - size of patch
def PatchingImgs(srcs, refs, indices=None, nPatchesPerLayer=100, patchSize=[40,40]):
    if indices is None:
        indices = range(0,len(srcs))
    
    print 'Patching:'
    imgPatches = np.zeros([len(indices), nPatchesPerLayer, 2, patchSize[0], patchSize[1]])
    for iLayer in range(0, len(indices)):
        if iLayer%10 == 0:
            print '%d...'%iLayer,
            sys.stdout.flush()
        startPosX = np.random.randint(0, srcs[0].shape[0]-patchSize[0], [nPatchesPerLayer])
        startPosY = np.random.randint(0, srcs[0].shape[1]-patchSize[1], [nPatchesPerLayer])
        endPosX = startPosX + patchSize[0]
        endPosY = startPosY + patchSize[1]
        for iPatch in range(0, nPatchesPerLayer):
            imgPatches[iLayer, iPatch, 0, ...] = srcs[indices[iLayer]][startPosX[iPatch]:endPosX[iPatch], startPosY[iPatch]:endPosY[iPatch]]
            imgPatches[iLayer, iPatch, 1, ...] = refs[indices[iLayer]][startPosX[iPatch]:endPosX[iPatch], startPosY[iPatch]:endPosY[iPatch]]
    print 'Done'
    
    # reshape
    imgPatches = imgPatches.reshape([len(indices) * nPatchesPerLayer, 2, patchSize[0], patchSize[1]])
    
    # random transform
    print 'Transforming...', 
    for iPatch in range(0,imgPatches.shape[0]):
        if np.random.rand(1) < 0.5:
            imgPatches[iPatch,...] = imgPatches[iPatch,:,::-1,:]
        if np.random.rand(1) < 0.5:
            imgPatches[iPatch,...] = imgPatches[iPatch,:,:,::-1]
    print 'Done'
    
    # random the patches 
    inds = range(0, imgPatches.shape[0])
    random.shuffle(inds)
    imgPatches = imgPatches[inds, ...]
    
    sys.stdout.flush()
    
    return imgPatches


# In[3]:

# do almost the same thing as PatchingImgs, the input is the two layers including the original noisy image
# srcs - the input
# srcs2 - the original noisy image
# refs - the residue (noisy - noiseless)
def PatchingImgs2(srcs, srcs2, refs, indices=None, nPatchesPerLayer=100, patchSize=[40,40]):
    if indices is None:
        indices = range(0,len(srcs))
    
    print 'Patching:'
    imgPatches = np.zeros([len(indices), nPatchesPerLayer, 3, patchSize[0], patchSize[1]])
    for iLayer in range(0, len(indices)):
        if iLayer%10 == 0:
            print '%d...'%iLayer,
            sys.stdout.flush()
        startPosX = np.random.randint(0, srcs[0].shape[0]-patchSize[0], [nPatchesPerLayer])
        startPosY = np.random.randint(0, srcs[0].shape[1]-patchSize[1], [nPatchesPerLayer])
        endPosX = startPosX + patchSize[0]
        endPosY = startPosY + patchSize[1]
        for iPatch in range(0, nPatchesPerLayer):
            imgPatches[iLayer, iPatch, 0, ...] = srcs[indices[iLayer]][startPosX[iPatch]:endPosX[iPatch], startPosY[iPatch]:endPosY[iPatch]]
            imgPatches[iLayer, iPatch, 1, ...] = srcs2[indices[iLayer]][startPosX[iPatch]:endPosX[iPatch], startPosY[iPatch]:endPosY[iPatch]]
            imgPatches[iLayer, iPatch, 2, ...] = refs[indices[iLayer]][startPosX[iPatch]:endPosX[iPatch], startPosY[iPatch]:endPosY[iPatch]]
    print 'Done'
    
    # reshape
    imgPatches = imgPatches.reshape([len(indices) * nPatchesPerLayer, 3, patchSize[0], patchSize[1]])
    
    # random transform
    print 'Transforming...', 
    for iPatch in range(0,imgPatches.shape[0]):
        if np.random.rand(1) < 0.5:
            imgPatches[iPatch,...] = imgPatches[iPatch,:,::-1,:]
        if np.random.rand(1) < 0.5:
            imgPatches[iPatch,...] = imgPatches[iPatch,:,:,::-1]
    print 'Done'
    
    sys.stdout.flush()
    
    return imgPatches


# In[ ]:

# train denoising net for low dose challenge data
# indices - the indices of the data to be used
# depth - depth of the network
# outputPath - outputPath of the trained networks
# max_iter - max iteration numbers of the training
# nChannel - 1 for single layer input, 2 for input including original noisy image
# patchSize - size of patch, assuming square
# patchPath - the path of the training datasets, looking for the hd5f index file 'trainingList.txt'
def TrainLowDoseChallenge(indices, depth, outputPath, max_iter=55000, nChannel=1, patchSize=40,
                          patchPath='/home/data0/dufan/MedicalCNNDenoising/data/40x40_Res_Full/'):   
    # make working directory

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    
    # Path names
    tlPath = os.path.join(outputPath, 'trainingList.txt')
    trainPath = os.path.join(outputPath, 'DnCNN_Train.prototxt')
    testPath = os.path.join(outputPath, 'DnCNN_Test.prototxt')
    solverPath = os.path.join(outputPath, 'DnCNN_Solver.prototxt')
    
    # Generate trainingList
    with open(tlPath, 'w') as f:
        for i in indices:
            f.write(os.path.join(patchPath, 'trainingData_'+str(i)+'.h5') + '\n')
        f.close()
    
    # Generate training prototxt
    GenerateNetwork.GenerateNetPrototxt(trainPath, tlPath, depth, caffe.TRAIN, 100, dim2=patchSize, nChannel=nChannel)
    GenerateNetwork.GenerateNetPrototxt(testPath, tlPath, depth, caffe.TEST, 20, dim2=patchSize, nChannel=nChannel)
    
    # Generate solver prototxt
    GenerateNetwork.GenerateADAMSolverPrototxt(solverPath, trainPath, os.path.join(outputPath, 'DnCNN'),
                                               max_iter, snapshot=10000)
    
    # Train
    solver = caffe.AdamSolver(solverPath)
    solver.solve()
    solver.net.save(os.path.join(outputPath, 'DnCNN.caffemodel'))


# In[2]:

# generate the training data for cascaded networks in the low dose challenge
# indices - a list containing all the indices of the datasets
# prototxts - a list containing all the trained test prototxts
# caffemodels - a list containing all the trained model weights
# outputPath - output dir for the newly trained net
# nPatchesPerLayer - number of patches per layer
# rTopLayers - select the rTopLayers*nLayers with top denoising error 
# basePath - the path of the low dose challenge dataset
def GenSeqTrainingDataLowDoseChallenge(indices, prototxts, caffemodels, outputPath, 
                                       nPatchesPerLayer=1000, rTopLayers = 0.05,
                                       basePath='/home/data0/dufan/CT_images/'):
    # make outputPath
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    
    # load networks
    nets = list()
    for i in range(0,len(prototxts)):
        nets.append(caffe.Net(prototxts[i], caffemodels[i], caffe.TEST))
    
    # Read all the high dose images and low dose images
    for index in indices:
        print 'Reading dataset %d...' %index,
        hdImgs = ReadFromCavaderData.ReadFromLowdoseChallengeData(index, True)
        ldImgs = ReadFromCavaderData.ReadFromLowdoseChallengeData(index, False)
        print 'Done'
        
        print 'Denoising:'
        for i in range(0,len(ldImgs)):
            if i%50 == 0:
                print '%d...' %i,
            for j in range(0, len(nets)):
                ldImgs[i] = ldImgs[i] - NNDenoise.PatchDenoiseParallel(nets[j], ldImgs[i])
            hdImgs[i] = ldImgs[i] - hdImgs[i]
        print 'Done'
        
        # get error list and select the indices
        errList = np.zeros([len(hdImgs)])
        for i in range(0,len(hdImgs)):
            errList[i] = np.linalg.norm(hdImgs[i], ord='fro')
        errIndices = np.argsort(errList)
        errIndices = errIndices[::-1]
        errIndices = errIndices[0:int(len(errIndices)*rTopLayers)]
        print 'Selected %d layers with maximum error out of %d layers'%(len(errIndices), len(hdImgs))
        print errIndices
        
        # patching
        imgPatches = PatchingImgs(ldImgs, hdImgs, errIndices, nPatchesPerLayer)
        
        # output hdf5 file
        print 'Writing hdf5...',
        with h5py.File(os.path.join(outputPath, 'trainingData_'+str(index)+'.h5'), 'w') as f:
            f['data'] = imgPatches.astype(np.float32)
            f['label'] = np.zeros([imgPatches.shape[0]], dtype=np.float32)
            f.close()
        print 'Done'
    
    # generate trainingList
    with open(os.path.join(outputPath, 'trainingList.txt'), 'w') as f:
        for index in indices:
            f.write(os.path.join(outputPath, 'trainingData_'+str(index)+'.h5\n'))
        f.close()


# In[2]:

# basically the same with the genSeqTrain....
# this one is for dual layer input
def GenSeqTrainingDataLowDoseChallenge2(indices, prototxts, caffemodels, outputPath, 
                                       nPatchesPerLayer=1000, rTopLayers = 0.05,
                                       basePath='/home/data0/dufan/CT_images/'):
    # make outputPath
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    
    # load networks
    nets = list()
    for i in range(0,len(prototxts)):
        nets.append(caffe.Net(prototxts[i], caffemodels[i], caffe.TEST))
    
    # Read all the high dose images and low dose images
    for index in indices:
        print 'Reading dataset %d...' %index,
        hdImgs = ReadFromCavaderData.ReadFromLowdoseChallengeData(index, True)
        ldImgs = ReadFromCavaderData.ReadFromLowdoseChallengeData(index, False)
        oriLdImgs = ReadFromCavaderData.ReadFromLowdoseChallengeData(index, False)
        print 'Done'
        
        print 'Denoising:'
        for i in range(0,len(ldImgs)):
            if i%50 == 0:
                print '%d...' %i,
            for j in range(0, len(nets)):
                if nets[j].blobs['dataSrc'].data.shape[1] == 1:
                    ldImgs[i] = ldImgs[i] - NNDenoise.PatchDenoiseParallel(nets[j], ldImgs[i])
                else:
                    ldImgs[i] = ldImgs[i] - NNDenoise.PatchDenoiseParallel2(nets[j], ldImgs[i], oriLdImgs[i])
            hdImgs[i] = ldImgs[i] - hdImgs[i]
        print 'Done'
        
        # get error list and select the indices
        errList = np.zeros([len(hdImgs)])
        for i in range(0,len(hdImgs)):
            errList[i] = np.linalg.norm(hdImgs[i], ord='fro')
        errIndices = np.argsort(errList)
        errIndices = errIndices[::-1]
        errIndices = errIndices[0:int(len(errIndices)*rTopLayers)]
        print 'Selected %d layers with maximum error out of %d layers'%(len(errIndices), len(hdImgs))
        print errIndices
        
        # patching
        imgPatches = PatchingImgs2(ldImgs, oriLdImgs, hdImgs, errIndices, nPatchesPerLayer)
        
        # output hdf5 file
        print 'Writing hdf5...',
        with h5py.File(os.path.join(outputPath, 'trainingData_'+str(index)+'.h5'), 'w') as f:
            f['data'] = imgPatches.astype(np.float32)
            f['label'] = np.zeros([imgPatches.shape[0]], dtype=np.float32)
            f.close()
        print 'Done'
    
    # generate trainingList
    with open(os.path.join(outputPath, 'trainingList.txt'), 'w') as f:
        for index in indices:
            f.write(os.path.join(outputPath, 'trainingData_'+str(index)+'.h5\n'))
        f.close()


# In[ ]:



