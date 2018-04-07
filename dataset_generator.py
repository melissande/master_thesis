import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

PATH_INPUT='INPUT/'
PATH_OUTPUT='OUTPUT/'
SIZE_PATCH=120

INPUT_SIZE=9
OUTPUT_SIZE=2

def _parse_images(paths_input,paths_output):
    '''
    Reads and saves as as an array image input and output
    :paths_input array of paths of the input images that have to be read  
    :paths_output array of paths of the output images that have to be read  
    returns input and output images as array
    '''
    input_ = []
    output_ = []

    for path_i in paths_input:
        with h5py.File(path_i, 'r') as hf:
            X =np.array(hf.get('data'))
            input_.append(X)

    for path_o in paths_output:
        with h5py.File(path_o, 'r') as hf:
            Y_build=np.array(hf.get('data'))
            Y_build=(Y_build>0).astype(int)
            Y_other= (1-Y_build).astype(int)
            Y=np.stack((Y_build,Y_other),axis=2)
            output_.append(Y)

        
            
    return np.asarray(input_),np.asarray(output_)
        

class DatasetGenerator():
    '''
    DatasetGenerator class
    '''

    # This decides whether "unique" keys should be included in the generator for each datapoint (typically useful for feature caching)
    include_keys = False

    #img_size = DATA_PATCH_INPUT_SIZE



    def __init__(self, paths_input: np.ndarray,paths_output: np.ndarray, batch_size: int = None):
        self.paths_input = paths_input
        self.paths_output = paths_output
        self.batch_size = batch_size

    @classmethod
    def from_root_folder(cls, root_folder: str, *, batch_size: int = None,max_data_size:  int = None):
        paths_input = []
        paths_output=[]
        
        
        for filename in sorted(os.listdir(root_folder+PATH_INPUT))[:max_data_size]:
            paths_input.append(os.path.join(root_folder+PATH_INPUT, filename))

        for filename in sorted(os.listdir(root_folder+PATH_OUTPUT))[:max_data_size]:

            paths_output.append(os.path.join(root_folder+PATH_OUTPUT, filename))
        
        
        return DatasetGenerator(np.asarray(paths_input), np.asarray(paths_output), batch_size=batch_size)

#     def shuffled(self, seed=None):
    def shuffled(self):
#         if seed is not None:
#             np.random.seed(seed)

        idx = np.arange(len(self.paths_input))
        np.random.shuffle(idx)
        generator = DatasetGenerator(self.paths_input[idx], self.paths_output[idx],batch_size=self.batch_size)
        generator.include_keys = self.include_keys


        return generator

    def __iter__(self):
        if self.batch_size is None:
            raise ValueError('Must set a batch size before iterating!')

        self.index = 0

        return self

    def __next__(self):

        while(self.index * self.batch_size) < len(self.paths_input):
            start = self.index * self.batch_size
            stop = min(start + self.batch_size, len(self.paths_input))

            X,Y = _parse_images(self.paths_input[start:stop],self.paths_output[start:stop])


            self.index += 1
            if self.include_keys:
                return self.paths_input[start:stop], X,self.paths_output[start:stop], Y
            else:
                return X, Y


        raise StopIteration
    def __data_aug__(self,X,Y):

        X,Y=data_augment(X,Y,self.batch_size)

        return X,Y


    def __len__(self):
        return len(self.paths_input)

    def __getitem__(self, val):
        if type(val) is not slice:
            raise ValueError('DatasetGenerators can only be sliced')

        sliced = DatasetGenerator(self.paths_input[val], self.paths_output[val],batch_size=self.batch_size)
        sliced.include_keys = self.include_keys


        return sliced



if __name__ == '__main__':

    root_folder = '../DATA_GHANA/DATASET/120_x_120_8_bands/TRAINING/'
    test_save= '../DATA_GHANA/DATASET/120_x_120_8_bands/TEST_SAVE/'
    if not os.path.exists(test_save):
            os.makedirs(test_save)

    batch_size = 5
    
    generator = DatasetGenerator.from_root_folder(root_folder, batch_size=batch_size)

    generator.shuffled()
    generator =generator.__iter__()
    
    
    for iteration in range(2):
        X,Y=generator.__next__()


        for i in range(len(X)):
            for j in range(INPUT_SIZE):
                plt.imsave(test_save+'X_iter'+str(iteration)+'batch_'+str(i)+'_band_'+str(j)+'.jpg',X[i,:,:,j])
            for j in range(OUTPUT_SIZE):
                plt.imsave(test_save+'Y_iter'+str(iteration)+'batch_'+str(i)+'_band_'+str(j)+'.jpg',Y[i,:,:,j])
        exit()
    
    