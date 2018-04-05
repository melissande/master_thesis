import h5py
import numpy as np
from numpy import newaxis
import cv2
import sys
import os
from image_utils import read_data_h5,write_data_h5
import matplotlib.pyplot as plt


'''
This script is used to build the dataset for GHANA image of Accra splitting everything into TRAINING,VALIDATION and TEST set.
The default setting is for Pancrhomatic + 8 pansharpened bands but for other settings see parts commented in the code. The other settings of dataset tested were:
- Panchromatic + 8 MS bands
- Panchromatic + 4 pansharpened bands (2,3,5,7)
- Panchromatic + 8 MS bands + 4 pansharpened bands (2,3,5,7)
python ../DATA_GHANA/RAW_PATCHES/120_x_120/ ../DATA_GHANA/DATASET/120_x_120_8_bands/ training_ratio=0.8 validation_ratio=0.2
where the two first arguments are compulsory. The first one is the path where are stored the patches for each band, the second one is where to store the dataset created using this script. training_ratio and validation_ratio indicate how much data we want in traning, validation and test set. 

'''
if __name__ == '__main__':

    if len(sys.argv)<3:
        print('Specify the path of the folders for Raw patches and dataset ')
        exit()
        
    ##BUILD 500X500 8 bands Dataset
    path_patches=sys.argv[1]
    #path_patches='../DATA_GHANA/RAW_PATCHES/120_x_120/'
    path_dataset=sys.argv[2]
    #path_dataset='../DATA_GHANA/DATASET/120_x_120_4_bands_PANSH_8_bands_MS/'
    if not os.path.exists(path_dataset):
            os.makedirs(path_dataset)
    
        
    training_ratio=0.8 #so    test_ratio=0.2
    validation_ratio=0.2
    
    
    for i in range(3, len(sys.argv)):
        arg = sys.argv[i]
        if arg.startswith('--training_ratio'):
            training_ratio=float(arg[len('--training_ratio='):])
        if arg.startswith('--validation_ratio'):
            validation_ratio=float(arg[len('--validation_ratio='):])

    list_input_panchro=[]
    list_input_ms=[]
    list_input_pansharp=[]
    for filename in sorted(os.listdir(path_patches)):
        if filename.startswith('panchro'):
            print('Reading %s'%filename)
            list_input_panchro.append(read_data_h5(path_patches+filename))
        if filename.startswith('ms'):
            print('Reading %s'%filename)
            list_input_ms.append(read_data_h5(path_patches+filename))
        if filename.startswith('pansharp'):
            print('Reading %s'%filename)
            list_input_pansharp.append(read_data_h5(path_patches+filename))
        if filename.startswith('groundtruth'):
            print('Reading %s'%filename)
            list_output=read_data_h5(path_patches+filename)
            
#     ##BUILD Panchro + Pansharp --> 9 bands
    
    list_input_panchro=np.squeeze(np.asarray(list_input_panchro))[newaxis,:,:,:]
    list_input_pansharp=np.squeeze(np.asarray(list_input_pansharp))
    list_input=np.concatenate((list_input_panchro,list_input_pansharp),axis=0)
    
#     ## Build Pancrho + MS --> 9 bands
    
#     list_input_panchro=np.squeeze(np.asarray(list_input_panchro))[newaxis,:,:,:]
#     list_input_ms=np.squeeze(np.asarray(list_input_ms))
#     list_input=np.concatenate((list_input_panchro,list_input_ms),axis=0)
    
#     ## Build Panchro + Pansharp 2,3,5,7

#     list_input_panchro=np.squeeze(np.asarray(list_input_panchro))[newaxis,:,:,:]
#     list_input_pansharp=np.stack((np.squeeze(np.asarray(list_input_pansharp))[1,:,:,:],
#                                         np.squeeze(np.asarray(list_input_pansharp))[2,:,:,:],
#                                         np.squeeze(np.asarray(list_input_pansharp))[4,:,:,:],
#                                         np.squeeze(np.asarray(list_input_pansharp))[6,:,:,:]),axis=0)
#     list_input=np.concatenate((list_input_panchro,list_input_pansharp),axis=0)

    ## Build Panchro + Pansharp 2,3,5,7 + MS -->14 bands
    
#     list_input_panchro=np.squeeze(np.asarray(list_input_panchro))[newaxis,:,:,:]
#     list_input_pansharp=np.stack((np.squeeze(np.asarray(list_input_pansharp))[1,:,:,:],
#                                         np.squeeze(np.asarray(list_input_pansharp))[2,:,:,:],
#                                         np.squeeze(np.asarray(list_input_pansharp))[4,:,:,:],
#                                         np.squeeze(np.asarray(list_input_pansharp))[6,:,:,:]),axis=0)
#     list_input_ms=np.squeeze(np.asarray(list_input_ms))
#     list_input=np.concatenate((list_input_panchro,list_input_pansharp,list_input_ms),axis=0)
    
    
    
    ## Followup
    
    list_input=np.transpose(list_input,(1,2,3,0))
    
    list_output=np.squeeze(list_output)
    
    
    print('Dataset read')
    idx_shuffle = np.arange(len(list_input))
    np.random.shuffle(idx_shuffle)
   
    print('Dataset shuffled')    
    list_input=list_input[idx_shuffle]
    list_output=list_output[idx_shuffle]
    
    #Do the split
    training_size=int(round(training_ratio*list_input.shape[0]))
    test_size=list_input.shape[0]-training_size
    validation_size=int(round(validation_ratio*training_size))
    training_size=training_size-validation_size
    
    
    print('Split (TRAINING - VALIDATION:%f) - TEST:%f  done'%(1-validation_ratio,training_ratio))
    print('Training size:%d, Validation size:%d, Test size: %d'%(training_size,validation_size,test_size))
    
    #Save the dataset
    
    if not os.path.exists(path_dataset+'TRAINING'):
            os.makedirs(path_dataset+'TRAINING')
            if not os.path.exists(path_dataset+'TRAINING/INPUT'):
                os.makedirs(path_dataset+'TRAINING/INPUT')
            if not os.path.exists(path_dataset+'TRAINING/OUTPUT'):
                os.makedirs(path_dataset+'TRAINING/OUTPUT')
    if not os.path.exists(path_dataset+'VALIDATION'):
            os.makedirs(path_dataset+'VALIDATION')
            if not os.path.exists(path_dataset+'VALIDATION/INPUT'):
                os.makedirs(path_dataset+'VALIDATION/INPUT')
            if not os.path.exists(path_dataset+'VALIDATION/OUTPUT'):
                os.makedirs(path_dataset+'VALIDATION/OUTPUT')
    if not os.path.exists(path_dataset+'TEST'):
            os.makedirs(path_dataset+'TEST')
            if not os.path.exists(path_dataset+'TEST/INPUT'):
                os.makedirs(path_dataset+'TEST/INPUT')
            if not os.path.exists(path_dataset+'TEST/OUTPUT'):
                os.makedirs(path_dataset+'TEST/OUTPUT')
            
#     print('BUILD TRAINING SET')
#     for i in range(training_size):
#         print('Patch %d'%i)
#         write_data_h5(path_dataset+'TRAINING/INPUT/input_'+str(i)+'.h5',list_input[i,:,:,:])
#         write_data_h5(path_dataset+'TRAINING/OUTPUT/output_'+str(i)+'.h5',list_output[i,:,:])
        
#     print('BUILD VALIDATION SET')
#     for i in range(training_size,training_size+validation_size):
#         print('Patch %d'%i)
#         write_data_h5(path_dataset+'VALIDATION/INPUT/input_'+str(i)+'.h5',list_input[i,:,:,:])
#         write_data_h5(path_dataset+'VALIDATION/OUTPUT/output_'+str(i)+'.h5',list_output[i,:,:])
        
#     print('BUILD TEST SET')
#     for i in range(training_size+validation_size,list_input.shape[0]):
#         print('Patch %d'%i)
#         write_data_h5(path_dataset+'TEST/INPUT/input_'+str(i)+'.h5',list_input[i,:,:,:])
#         write_data_h5(path_dataset+'TEST/OUTPUT/output_'+str(i)+'.h5',list_output[i,:,:])
            
    
            
        
        