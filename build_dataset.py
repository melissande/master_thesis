import h5py
import numpy as np
from numpy import newaxis
import cv2
import sys
import os
from image_utils import read_data_h5,write_data_h5
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    ##BUILD 500X500 8 bands Dataset
    path_patches='../DATA_GHANA/RAW_PATCHES/500_x_500/'
    path_dataset='../DATA_GHANA/DATASET/500_x_500_8_bands/'
    if not os.path.exists(path_dataset):
            os.makedirs(path_dataset)
    
   
    training_ratio=0.8 #so    test_ratio=0.2
    validation_ratio=0.2
    
    
    list_input=[]
    for filename in sorted(os.listdir(path_patches)):
        if filename.startswith('panchro'):
            print('Reading %s'%filename)
            list_input.append(read_data_h5(path_patches+filename))
        if filename.startswith('pansharpened'):
            print('Reading %s'%filename)
            list_input.append(read_data_h5(path_patches+filename))
        if filename.startswith('groundtruth'):
            print('Reading %s'%filename)
            list_output=read_data_h5(path_patches+filename)
            
          
    list_input=np.squeeze(np.asarray(list_input))
    print('list input shape [%d,%d,%d,%d]'%list_input.shape)  
    list_input=np.transpose(list_input,(1,2,3,0))
    print('list input shape [%d,%d,%d,%d]'%list_input.shape)
    
    print('list output shape [%d,%d,%d,%d]'%list_output.shape) 
    list_output=np.squeeze(list_output)
    print('list output shape [%d,%d,%d]'%list_output.shape) 
    
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
            
    print('BUILD TRAINING SET')
    for i in range(training_size):
        print('Patch %d'%i)
        write_data_h5(path_dataset+'TRAINING/INPUT/input_'+str(i)+'.h5',list_input[i,:,:,:])
        write_data_h5(path_dataset+'TRAINING/OUTPUT/output_'+str(i)+'.h5',list_output[i,:,:])
        
    print('BUILD VALIDATION SET')
    for i in range(training_size,training_size+validation_size):
        print('Patch %d'%i)
        write_data_h5(path_dataset+'VALIDATION/INPUT/input_'+str(i)+'.h5',list_input[i,:,:,:])
        write_data_h5(path_dataset+'VALIDATION/OUTPUT/output_'+str(i)+'.h5',list_output[i,:,:])
        
    print('BUILD TEST SET')
    for i in range(training_size+validation_size,list_input.shape[0]):
        print('Patch %d'%i)
        write_data_h5(path_dataset+'TEST/INPUT/input_'+str(i)+'.h5',list_input[i,:,:,:])
        write_data_h5(path_dataset+'TEST/OUTPUT/output_'+str(i)+'.h5',list_output[i,:,:])
            
    
            
        