
# coding: utf-8

# In[1]:

import caffe
import scipy
import numpy as np
import math
import numpy.matlib


# In[2]:

def PatchDenoise(net, src, patchSize = 40, stride = 32):
    src = src[np.newaxis, np.newaxis, ...]
    outputImg = np.zeros(src.shape)
    outputWeight = np.zeros(src.shape)

    sz = patchSize
    fullSz = src.shape[-1]
    
    mask1D = scipy.signal.gaussian(sz, sz / 3.0)
    mask = np.matlib.repmat(mask1D, sz, 1)
    mask = np.multiply(mask, mask.transpose())
    mask = mask.astype(np.float32)    
    
    nPatches = int(scipy.ceil((fullSz-sz/2) / float(stride)))

    for ix in range(0,nPatches):
        for iy in range(0,nPatches):
            basex = ix*stride
            basey = iy*stride
            if basex+sz > fullSz:
                basex = fullSz - sz
            if basey+sz > fullSz:
                basey = fullSz - sz
            subImg = src[...,basey:basey+sz, basex:basex+sz]
            net.blobs['dataSrc'].data[...]=subImg
            net.forward();
            outputImg[...,basey:basey+sz, basex:basex+sz] += np.multiply(net.blobs[net.outputs[0]].data[...], mask)
            outputWeight[...,basey:basey+sz, basex:basex+sz] += mask

    outputImg = np.divide(outputImg, outputWeight)

    return outputImg.squeeze()


# In[3]:

# parallel denoising, the patches are extracted first, stacked later, then put into the network
# the network should accept multiple patches as input
def PatchDenoiseParallel(net, src, patchSize = 40, stride = 32):
    outputImg = np.zeros(src.shape)
    outputWeight = np.zeros(src.shape)

    sz = patchSize
    fullSz = src.shape[-1]
    
    mask1D = scipy.signal.gaussian(sz, sz / 3.0)
    mask = np.matlib.repmat(mask1D, sz, 1)
    mask = np.multiply(mask, mask.transpose())
    mask = mask.astype(np.float32)
    
    
    nPatches = int(scipy.ceil((fullSz-sz/2) / float(stride)))
    
    # extract patches
    patches = np.zeros([nPatches * nPatches, 1, patchSize, patchSize], dtype=np.float32)
    baseCoords = np.zeros([nPatches * nPatches, 2], dtype=np.int)
    ind = 0
    for ix in range(0,nPatches):
        for iy in range(0,nPatches):
            basex = ix*stride
            basey = iy*stride
            if basex+sz > fullSz:
                basex = fullSz - sz
            if basey+sz > fullSz:
                basey = fullSz - sz
            patches[ind, 0, ...] = src[basey:basey+sz, basex:basex+sz]
            baseCoords[ind, 0] = basex
            baseCoords[ind, 1] = basey
            ind += 1
    
    # put the patches through the network batch by batch
    batchSize = net.blobs['dataSrc'].data.shape[0]
    nBatches = int(math.ceil(patches.shape[0] / float(batchSize)))
    for i in range(0,nBatches-1):
        indStart = i * batchSize
        net.blobs['dataSrc'].data[...] = patches[indStart:indStart+batchSize, ...]
        net.forward()
        patches[indStart:indStart+batchSize,...] = net.blobs[net.outputs[0]].data
    indStart = (nBatches-1) * batchSize
    nLeftPatches = patches.shape[0] - indStart
    net.blobs['dataSrc'].data[0:nLeftPatches,...] = patches[indStart:, ...]
    net.forward()
    patches[indStart:,...] = net.blobs[net.outputs[0]].data[0:nLeftPatches,...]
    
    # put the patches back together
    for ind in range(0,baseCoords.shape[0]):
        basex = baseCoords[ind,0]
        basey = baseCoords[ind,1]
        outputImg[basey:basey+sz, basex:basex+sz] += np.multiply(patches[ind, 0, ...], mask)
        outputWeight[basey:basey+sz, basex:basex+sz] += mask
    
    outputImg = np.divide(outputImg, outputWeight)
    
    return outputImg


# In[4]:

# parallel denoising, the patches are extracted first, stacked later, then put into the network
# the network should accept multiple patches as input
# this version works for dual layer input
def PatchDenoiseParallel2(net, src, originalLowdoseImg, patchSize = 40, stride = 32):
    outputImg = np.zeros(src.shape)
    outputWeight = np.zeros(src.shape)

    sz = patchSize
    fullSz = src.shape[-1]
    
    mask1D = scipy.signal.gaussian(sz, sz / 3.0)
    mask = np.matlib.repmat(mask1D, sz, 1)
    mask = np.multiply(mask, mask.transpose())
    mask = mask.astype(np.float32)
    
    
    nPatches = int(scipy.ceil((fullSz-sz/2) / float(stride)))
    
    # extract patches
    patches = np.zeros([nPatches * nPatches, 2, patchSize, patchSize], dtype=np.float32)
    baseCoords = np.zeros([nPatches * nPatches, 2], dtype=np.int)
    ind = 0
    for ix in range(0,nPatches):
        for iy in range(0,nPatches):
            basex = ix*stride
            basey = iy*stride
            if basex+sz > fullSz:
                basex = fullSz - sz
            if basey+sz > fullSz:
                basey = fullSz - sz
            patches[ind, 0, ...] = src[basey:basey+sz, basex:basex+sz]
            patches[ind, 1, ...] = originalLowdoseImg[basey:basey+sz, basex:basex+sz]
            baseCoords[ind, 0] = basex
            baseCoords[ind, 1] = basey
            ind += 1
    
    # put the patches through the network batch by batch
    batchSize = net.blobs['dataSrc'].data.shape[0]
    nBatches = int(math.ceil(patches.shape[0] / float(batchSize)))
    for i in range(0,nBatches-1):
        indStart = i * batchSize
        net.blobs['dataSrc'].data[...] = patches[indStart:indStart+batchSize, ...]
        net.forward()
        patches[indStart:indStart+batchSize,0,...] = net.blobs[net.outputs[0]].data.squeeze()
    indStart = (nBatches-1) * batchSize
    nLeftPatches = patches.shape[0] - indStart
    net.blobs['dataSrc'].data[0:nLeftPatches,...] = patches[indStart:, ...]
    net.forward()
    patches[indStart:,0,...] = net.blobs[net.outputs[0]].data[0:nLeftPatches,...].squeeze()
    
    # put the patches back together
    for ind in range(0,baseCoords.shape[0]):
        basex = baseCoords[ind,0]
        basey = baseCoords[ind,1]
        outputImg[basey:basey+sz, basex:basex+sz] += np.multiply(patches[ind, 0, ...], mask)
        outputWeight[basey:basey+sz, basex:basex+sz] += mask
    
    outputImg = np.divide(outputImg, outputWeight)
    
    return outputImg


# In[6]:

# sequential patch based denoising
# for each net, the program will use the 2 versions according to the nets' input size
def SeqPatchDenoiseParallel(nets, src, originalLowdoseImg=None, patchSize = 40, stride = 32):
    outputImgs = list()
    
    for i in range(0, len(nets)):
        if nets[i].blobs['dataSrc'].data.shape[1] == 1:
            src = src - PatchDenoiseParallel(nets[i], src, patchSize, stride)
        else:
            src = src - PatchDenoiseParallel2(nets[i], src, originalLowdoseImg, patchSize, stride)
        outputImgs.append(src)
    
    return outputImgs
            
    


# In[ ]:



