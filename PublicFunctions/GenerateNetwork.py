
# coding: utf-8

# In[1]:

# generate the net prototxt file
# easy to deduct what the functions are doing from the name, no need for documentation
import caffe
from caffe import layers as L, params as P
import re
import StringIO


# In[2]:

def AddConvModule(net, name, bottom=None, kernel_size=3, num_output=64,  
                  pad = None, stride = 1, bnorm = True, ReLU = True, phase = caffe.TRAIN):
    if pad is None:
        pad = int(kernel_size / 2)
    if bottom is None:
        bottom = net.tops.keys()[-1]  #Use the last layer
    
    # Add convolution layer
    net.tops['%s_conv' % name] = L.Convolution(
        bottom=bottom, kernel_size=kernel_size, num_output=num_output, pad=pad, stride=stride,
        weight_filler={'type':'xavier'},
        bias_filler={'type':'constant', 'value':0}
        )
    
    # Add bnorm layer
    if bnorm is True:
        net.tops['%s_conv_bnorm' % name] = L.BatchNorm(
            bottom=net.tops.keys()[-1],
            batch_norm_param={'use_global_stats':(phase==caffe.TEST)},
            param={'lr_mult':0}, # remember to use regex to repeat this for two more times
        )
        net.tops['%s_conv_scale' % name] = L.Scale(
            bottom=net.tops.keys()[-1],
            scale_param={'bias_term':True}
        )
    
    # Add ReLU layer
    if ReLU is True:
        net.tops['%s_conv_relu' % name] = L.ReLU(
            bottom=net.tops.keys()[-1], 
        )
    
    return net


# In[3]:

def GenerateNetPrototxt(outputPath, dataPath, nConvModule, phase, dim0, dim2, nChannel=1, kernel_size=3):
    ns = caffe.NetSpec()
    
    # data layers
    if phase == caffe.TRAIN:
        ns.data = L.Data(
            type='HDF5Data',
            input_param={'shape':{'dim':[1,nChannel+1,dim2,dim2]}},  
            include={'phase':caffe.TRAIN}, 
            hdf5_data_param={'source':dataPath, 'batch_size': dim0})
        ns.slice_data = L.Slice(
            bottom='data', top=['dataSrc', 'dataRef'], # remember to remove the extra top
            slice_param={'axis':1, 'slice_point':nChannel}, 
            include={'phase':caffe.TRAIN}) 
    else:
        ns.data = L.Data(
            type='Input',
            input_param={'shape':{'dim':[dim0,nChannel,dim2,dim2]}},
            top='dataSrc'
            )

    # conv layers
    ns = AddConvModule(ns, '0', 'dataSrc', kernel_size, 64, bnorm = False, ReLU=True, phase = phase)
    for i in range(1,nConvModule+1):
        ns = AddConvModule(ns, '%d' %i, kernel_size=kernel_size, phase = phase)
    ns = AddConvModule(ns, '%d' % (nConvModule+1), None, kernel_size, 1, bnorm = False, ReLU=False, phase = phase)

    # loss layers
    if phase == caffe.TRAIN:
        ns.loss = L.EuclideanLoss(
            bottom=[ns.tops.keys()[-1], 'dataRef'], 
            include={'phase':caffe.TRAIN})
    
    # post processing
    sio = StringIO.StringIO()
    print >> sio, ns.to_proto()
    s = sio.getvalue()
    if phase == caffe.TRAIN:
        s2 = re.sub('( *)top:( *)"slice_data"\n', '', s, re.DOTALL | re.M);
    else:
        s2 = re.sub('( *)top:( *)"data"\n', '', s, re.DOTALL | re.M);

    # find the param {lr_mult:0} and repeat it for another 2 times
    searchPattern = '( *)param( *){\n( *)lr_mult:( *)0\n( *)}\n'
    m = re.search(searchPattern, s2)
    if m is not None:
        subString = m.group(0)
        s3 = re.sub(searchPattern, subString+subString+subString, s2, re.DOTALL | re.M)
    else:
        s3 = s2
    
    # save file
    with open(outputPath, 'w') as f:
        f.write(s3)
        f.close()
    


# In[4]:

def GenerateADAMSolverPrototxt(outputPath, net, snapshot_prefix, 
                               max_iter=50000, snapshot=10000, display=10):
    with open(outputPath, 'w') as f:
        f.write('net: "%s"\n'%net)
        f.write('type: "ADAM"\nbase_lr: 0.001\nlr_policy: "fixed"\ngamma: 1\ndelta: 1e-8\n')
        f.write('momentum: 0.9\nmomentum2: 0.999\nweight_decay: 0.0001\n')
        f.write('max_iter: %d\n'%max_iter)
        f.write('display: %d\n'%display)
        f.write('snapshot: %d\n'%snapshot)
        f.write('snapshot_prefix: "%s"\n'%snapshot_prefix)
        f.write('solver_mode: GPU\n')
        f.close()
        

