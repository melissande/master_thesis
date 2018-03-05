import tensorflow as tf
import h5py
import numpy as np
from numpy import newaxis
import cv2
import sys
import os
from osgeo import gdal
from image_utils import read_images,write_data_h5,read_labels
import matplotlib.pyplot as plt



NAME_PANCHRO='panchro.tif'
NAME_PANSHARP='pansharp.tif'
NAME_MS='ms.tif'
NAME_LABELS='groundtruth.png'

WIDTH=500
STRIDE=250


def prepare_ms_hr(ms_lr,size_hr):
    '''
    Prepares the upsampled MS image 
    :ms_lr input image to upsample
    :size_hr to upsample the Low Resolution MS image to the dimension of the High Resolution panchromatic image
    '''
    ms_ph=tf.placeholder(tf.float64, [ms_lr.shape[0],ms_lr.shape[1],ms_lr.shape[2],ms_lr.shape[3]], name='ms_placeholder')
    ms_hr=tf.image.resize_images(ms_ph, [size_hr[0], size_hr[1]])
    ms_hr=tf.cast(ms_hr,tf.float64,name='cast_ms_hr')

    with tf.Session() as sess:
        ms_hr= sess.run(ms_hr,feed_dict={ms_ph: ms_lr})
        return ms_hr
        
        
def extract_patches(data,width,stride,path_out):
    '''
    Extract patches from images and writes the output to .h5 file format
    :data input image 
    :width dimensiton of the patch
    :stride stride of patch selection on the image
    :path_out where to save the patches
    '''
    print('Patch extraction with stride=%d and width=%d begins for: %s'%(stride,width,path_out) )
    data_pl=tf.placeholder(tf.float64, [data.shape[0],data.shape[1],data.shape[2],data.shape[3]], name='data_placeholder')
    data_o=tf.extract_image_patches(images=data_pl,ksizes=[1,width,width,1],strides=[1,stride,stride,1],rates=[1,1,1,1],padding='VALID')
    print('Patch extraction done')
    size_tot=data_o.get_shape().as_list()
    data_o=tf.reshape(data_o,[size_tot[1]*size_tot[2],width,width,data.shape[3]])
    with tf.Session() as sess:
        Data_o= sess.run(data_o,feed_dict={data_pl: data})
        write_data_h5(path_out,Data_o)
        print('%d patches of size %d x %d saved as list for %s'%(Data_o.shape[0],Data_o.shape[1],Data_o.shape[2],path_out))
        return Data_o
    
if __name__ == '__main__':
    path='../DATA_GHANA/RAW_DATA/'

    path_patches='../DATA_GHANA/RAW_PATCHES/500_x_500/'
    if not os.path.exists(path_patches):
            os.makedirs(path_patches)
  
    patch_test_number=300
#     ## Panchromatic
    panchromatic_file=path+NAME_PANCHRO
    panchromatic=read_images(panchromatic_file)
    hr_size=panchromatic.shape
    panchromatic=panchromatic[newaxis,:,:,newaxis]
    
    print('\n PANCHROMATIC \n\n')
    panchromatic=extract_patches(panchromatic,WIDTH,STRIDE,path_patches+'panchro.h5')
    plt.imshow(panchromatic[patch_test_number,:,:,0])
    plt.show()

    
# #     ##MS bands
    
    ms_file=path+NAME_MS 
    ms=read_images(ms_file)
    ms=np.transpose(ms,(1,2,0))
    print('\n MS BANDS\n\n')
    for i in range(ms.shape[2]):
        print('\n Band %d \n'%i)
        ms_i=ms[:,:,i]
        print('Upscale')
        ms_hr=prepare_ms_hr(ms_i[newaxis,:,:,newaxis],hr_size)
        ms_hr=extract_patches(ms_hr,WIDTH,STRIDE,path_patches+'ms_hr_'+str(i)+'.h5')
        plt.imshow(ms_hr[patch_test_number,:,:,0])
        plt.show()

    
    ##Pansharpened bands
    
    pansharpened_file=path+NAME_PANSHARP 
    pansharpened=read_images(pansharpened_file)
    pansharpened=np.transpose(pansharpened,(1,2,0))
    print('\n BANDS\n\n')
    for i in range(pansharpened.shape[2]):
        print('\n Band %d \n'%i)
        pansharpened_i=pansharpened[:,:,i]
        print('Upscale')
        pansharpened_i=extract_patches(pansharpened_i[newaxis,:,:,newaxis],WIDTH,STRIDE,path_patches+'pansharpened_'+str(i)+'.h5')
        plt.imshow(pansharpened_i[patch_test_number,:,:,0])
        plt.show()
        
#     ## Label patches
    
    labels_file=path+NAME_LABELS
    labels=read_labels(labels_file)
    labels=labels[newaxis,:,:,newaxis]
    
    print('\n LABELS \n\n')
    labels=extract_patches(labels,WIDTH,STRIDE,path_patches+'groundtruth.h5')
    plt.imshow(labels[patch_test_number,:,:,0])
    plt.show()
    
    

    
    
    

    