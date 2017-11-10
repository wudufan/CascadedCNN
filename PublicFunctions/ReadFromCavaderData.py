
# coding: utf-8

# In[9]:

import os
import dicom
import numpy as np
import re
import glob


# In[11]:

# Read from low dose chanllege data
# index - the index for the datasets to read, from 0 to 9
# readFullDose - True to read full dose, False to read quater dose
# slices - a list of slices to be read, None to read all
# basePath - the path of the low dose challenge dataset
def ReadFromLowdoseChallengeData(index, readFullDose, slices=None, 
                                basePath='/home/data0/dufan/CT_images/'):
    if readFullDose is True:
        basePath = os.path.join(basePath, 'full_dose_image')
    else:
        basePath = os.path.join(basePath, 'quater_dose_image')
    
    # locate folder
    imgDirs = os.listdir(basePath)
    basePath = os.path.join(basePath, imgDirs[index])
    
    # read images
    imgs = list()
    fileNames = glob.glob(os.path.join(basePath, '*.IMA'))
    if slices is None:
        slices = range(0,len(fileNames))
    for i in slices:
        curImg = dicom.read_file(fileNames[i]).pixel_array.astype(np.float32)
        imgs.append(curImg)
    
    return imgs
    


# In[ ]:




# In[ ]:



