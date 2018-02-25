#Virtual environments on cluster

##Thales 

###Basic libraries
Run
```sh
$ conda create -n env_thales python=3.6 numpy pip
$ source activate env_thales
$ pip install scipy
$ pip install matplotlib
$ pip install h5py
$ pip install tensorflow-gpu 
$ conda install -c menpo opencv
```
pay attention to the cuda version installed, you need to know what version of tensorflow-gpu and cuda/cdnn is corresponding to 								add it to the bashrc 

###Add to bashrc for TensorFlow 1.5 is expecting Cuda 9.0 ( NOT 9.1 ), as well as cuDNN 7
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-9.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-9.0
export LD_LIBRARY_PATH=/usr/local/cuDNNv7.0-8/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH

###For data augmentation
Run
```sh
$ pip install git+https://github.com/aleju/imgaug


##DHI 
Run
```sh
$ conda create --name env_dhi --clone env_thales
$ source activate env_dhi
$ pip install pandas
$ conda install gdal
$ conda install -c ioos rtree 
$ pip install centerline
$ pip install osmnx
```


##create jupyter notebook
### on the cluster
```sh
$CUDA_VISIBLE_DEVICES=0 jupyter notebook --no-browser --port=8888
```
###from local machine
```sh
$ssh -N -f -L localhost:8881:localhost:8888 s161362@mnemosyne.compute.dtu.dk
```

###For jupyter notebook
```sh
$pip install ipykernel
$python -m ipykernel install --user --name=env_dhi
```