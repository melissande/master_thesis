import sys
import numpy as np
import os
import cv2
import h5py
import logging
import tensorflow as tf
from math import log10
from collections import OrderedDict
from image_utils import standardize
from IOU_computations import *


import matplotlib.pyplot as plt
from typing import Iterator, Tuple
from dataset_generator import DatasetGenerator
from layers import weight_variable,bias_variable,conv2d,weight_variable_devonc,deconv2d,max_pool,pixel_wise_softmax,cross_entropy,features_concat,pixel_wise_softmax_2


##########
GLOBAL_PATH='MODEL_BASIC_TEST_120/'
##########

if not os.path.exists(GLOBAL_PATH):
            os.makedirs(GLOBAL_PATH)
        
#############
PATH_TRAINING='TRAINING/'
PATH_VALIDATION='VALIDATION/'
PATH_TEST='TEST/'

PATH_INPUT='INPUT/'
PATH_OUTPUT='OUTPUT/'
##############

        
INPUT_CHANNELS=9
OUTPUT_CHANNELS=2
NB_CLASSES=2

SIZE_PATCH=120
##############
MODEL_PATH_SAVE=GLOBAL_PATH+'RESUNET_BASIC_TEST.ckpt'
MODEL_PATH_RESTORE=''
TEST_SAVE=GLOBAL_PATH+'TEST_SAVE/'
if not os.path.exists(TEST_SAVE):
            os.makedirs(TEST_SAVE)
        
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

##############

REC_SAVE=2000#2000
DROPOUT=0.9#0.9
DEFAULT_BATCH_SIZE = 10#10
DEFAULT_EPOCHS = 15#15
DEFAULT_ITERATIONS =495#495
DEFAULT_VALID=100#70
DISPLAY_STEP=50#50


##############

DEFAULT_LAYERS=4
DEFAULT_FEATURES_ROOT=32
DEFAULT_FILTERS_SIZE=3
DEFAULT_POOL_SIZE=2
OPTIMIZER='adam'

####### TMP folder for IOU

TMP_IOU=TEST_SAVE+'TMP_IOU/'
if not os.path.exists(TMP_IOU):
            os.makedirs(TMP_IOU)

def create_conv_net(x, keep_prob, channels, n_class, layers=DEFAULT_LAYERS, features_root=DEFAULT_FEATURES_ROOT, filter_size=DEFAULT_FILTERS_SIZE, pool_size=DEFAULT_POOL_SIZE,phase_train=True):
    """
    Creates a new convolutional unet for the given parametrization.
    
    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    """

    logging.info("Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(layers=layers,
                                                                                                           features=features_root,
                                                                                                           filter_size=filter_size,
                                                                                                           pool_size=pool_size))
    # Placeholder for the input image
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    x_image = tf.reshape(x, tf.stack([-1,nx,ny,channels]))
    in_node = x_image
    batch_size = tf.shape(x_image)[0]
 
    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()
    
    in_size = 1000
    size = in_size
    # down layers
    for layer in range(0, layers):
        features = 2**layer*features_root
        stddev = np.sqrt(2 / (filter_size**2 * features))
        if layer == 0:
            w1 = weight_variable([filter_size, filter_size, channels, features], stddev)
        else:
            w1 = weight_variable([filter_size, filter_size, features//2, features], stddev)
            
        w2 = weight_variable([filter_size, filter_size, features, features], stddev)
        b1 = bias_variable([features])
        b2 = bias_variable([features])
        

        conv1 = conv2d(in_node, w1, keep_prob)
        conv1=tf.contrib.layers.batch_norm(conv1, center=True, scale=True, is_training=phase_train)
        tmp_h_conv = tf.nn.relu(conv1 + b1)
        conv2 = conv2d(tmp_h_conv, w2, keep_prob)
        conv2=tf.contrib.layers.batch_norm(conv2, center=True, scale=True, is_training=phase_train)
        conv2=conv2+conv1
        dw_h_convs[layer] = tf.nn.relu(conv2 + b2)
        
        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))
        
        size -= 4
        if layer < layers-1:
            pools[layer] = max_pool(dw_h_convs[layer], pool_size)
            in_node = pools[layer]
            size /= 2
        
    in_node = dw_h_convs[layers-1]

    # up layers
    for layer in range(layers-2, -1, -1):
        features = 2**(layer+1)*features_root
        stddev = np.sqrt(2 / (filter_size**2 * features))
        
        wd = weight_variable_devonc([pool_size, pool_size, features//2, features], stddev)
        bd = bias_variable([features//2])
        h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
        h_deconv_concat = features_concat(dw_h_convs[layer], h_deconv)
        deconv[layer] = h_deconv_concat
        
        w1 = weight_variable([filter_size, filter_size, features, features//2], stddev)
        w2 = weight_variable([filter_size, filter_size, features//2, features//2], stddev)
        b1 = bias_variable([features//2])
        b2 = bias_variable([features//2])
        
        conv1 = conv2d(h_deconv_concat, w1, keep_prob)
        conv1=tf.contrib.layers.batch_norm(conv1, center=True, scale=True, is_training=phase_train)
        h_conv = tf.nn.relu(conv1 + b1)
        conv2 = conv2d(h_conv, w2, keep_prob)
        conv2=tf.contrib.layers.batch_norm(conv2, center=True, scale=True, is_training=phase_train)
        conv2=conv2+conv1
        in_node = tf.nn.relu(conv2 + b2)
        up_h_convs[layer] = in_node

        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))
        
        size *= 2
        size -= 4

    # Output Map
    weight = weight_variable([1, 1, features_root, n_class], stddev)
    bias = bias_variable([n_class])
    conv = conv2d(in_node, weight, tf.constant(1.0))
    output_map = tf.nn.relu(conv + bias)
    up_h_convs["out"] = output_map
    

    variables = []
    for w1,w2 in weights:
        variables.append(w1)
        variables.append(w2)
        
    for b1,b2 in biases:
        variables.append(b1)
        variables.append(b2)
    
    
    return output_map, variables, int(in_size - size)
class CustomCNN():
    
    
    def __init__(self, channels=INPUT_CHANNELS, n_class=NB_CLASSES,layers=DEFAULT_LAYERS, features_root=DEFAULT_FEATURES_ROOT, filter_size=DEFAULT_FILTERS_SIZE, pool_size=DEFAULT_POOL_SIZE ):
        """
        Initializes the custom CNN 
        """
        self.n_class = n_class
        tf.reset_default_graph()
        
        self.X_placeholder = tf.placeholder(tf.float32, [None, None, None, INPUT_CHANNELS], name='X_placeholder')
        self.y_placeholder = tf.placeholder(tf.float32, [None, None, None,NB_CLASSES], name='y_placeholder')
        self.keep_prob = tf.placeholder(tf.float32)
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')

        logits, self.variables, self.offset = create_conv_net(self.X_placeholder, self.keep_prob, channels, n_class,layers, features_root, filter_size,pool_size)
        
        self.cross_entropy = tf.reduce_mean(cross_entropy(tf.reshape(self.y_placeholder, [-1, n_class]),
                                                          tf.reshape(pixel_wise_softmax_2(logits), [-1, n_class])))
        self.cost=self.cross_entropy
#         self.gradients_node = tf.gradients(self.cross_entropy, self.variables)
        
        self.predicter = pixel_wise_softmax_2(logits)
        self.correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y_placeholder, 3))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        
        
    def save(self, sess,saver, model_path,step):
        """
        Saves the current session to a checkpoint
        
        :param sess: current session
        :param saver: saver of the model
        :param model_path: path to file system location
        :step: step of the iterations during training when the model is stored
        """
       
        save_path=saver.save(sess, model_path,global_step=step)
        
        return save_path

    
    def restore(self, sess,saver, model_path):
        """
        Restores a session from a checkpoint
        
        :param sess: current session instance
        :param saver: saver of the model
        :param model_path: path to file system checkpoint location
        """

        print("Reading checkpoints...")
        saver.restore(sess, model_path)
        summary="Model restored from file: %s" % (model_path)
        print(summary)
 
        
    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data
        
        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2) 
        """
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
        
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            
            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
            prediction = sess.run(self.predicter, feed_dict={self.X_placeholder: x_test, self.y_placeholder: y_dummy, self.keep_prob: 1.})
            
        return prediction
        
        
class Trainer(object):
    """
    Trains a unet instance
    
    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer ->learning_rate, decay_rate,momentum (momentum) or learning_rate (adam)   
    """
    def __init__(self, net, batch_size=10, optimizer="adam", opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        
    def _get_optimizer(self, training_iters, global_step):
        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)
            
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate, 
                                                        global_step=global_step, 
                                                        decay_steps=training_iters,  
                                                        decay_rate=decay_rate, 
                                                        staircase=True)
            
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs).minimize(self.net.cost,global_step=global_step)
        elif self.optimizer == "adam":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.0001)
            self.learning_rate_node = tf.Variable(learning_rate)
            
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node, 
                                               **self.opt_kwargs).minimize(self.net.cost,global_step=global_step)
        return optimizer
    
    def _initialize(self, training_iters, prediction_path):
        global_step = tf.Variable(0)
        
#         self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]))

        self.optimizer = self._get_optimizer(training_iters, global_step)        
        init = tf.global_variables_initializer()
        
        self.prediction_path = prediction_path
        
        return init
    
    def train(self, data_provider_path, save_path='', restore_path='', training_iters=4, epochs=3, dropout=0.9, display_step=1, validation_batch_size=30,rec_save=1, prediction_path = ''):
        """
        Lauches the training process
        
        :param data_provider_path: where the DATASET folder is
        :param save_path: path where to store checkpoints
        :param restore_path: path where is the model to restore is stored
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored 
        :param prediction_path: path where to save predictions on each epoch
        """
        
        PATH_TRAINING=data_provider_path+'TRAINING/'
        PATH_VALIDATION=data_provider_path+'VALIDATION/'
        PATH_TEST=data_provider_path+'TEST/'
        ####### TMP folder for IOU

        TMP_IOU=prediction_path+'TMP_IOU/'
        if not os.path.exists(TMP_IOU):
            os.makedirs(TMP_IOU)
            
        #STORE PSNR for ANALYSIS
        loss_train=np.zeros(training_iters*epochs)
        file_train = open(prediction_path+'loss_train.txt','w') 
        loss_verif=np.zeros(epochs)
        file_verif = open(prediction_path+'loss_verif.txt','w')
        #STORE IOU for ANALYSIS
        IOU_verif=np.zeros(epochs)
        IOU_file_verif = open(TEST_SAVE+'iou_verif.txt','w')
        #STORE f1_IOU for ANALYSIS
        f1_IOU_verif=np.zeros(epochs)
        f1_IOU_file_verif = open(TEST_SAVE+'f1_iou_verif.txt','w')
        
        
        if epochs == 0:
            return save_path
        if save_path=='':
            return 'Specify a path where to store the Model'
        init = self._initialize(training_iters, prediction_path)
        saver = tf.train.Saver()
        with tf.Session() as sess:        
            
            
            if restore_path=='':
                sess.run(init)
            else:
                self.net.restore(sess,saver,restore_path )

            

            val_generator = DatasetGenerator.from_root_folder(PATH_VALIDATION, batch_size=validation_batch_size)
            val_generator=val_generator.shuffled()
            val_generator=val_generator.__iter__()
            X_val,Y_val=val_generator.__next__()
            X_val=standardize(X_val)
            
            self.store_prediction(sess, X_val, Y_val, "_init",validation_batch_size,True)
            
            train_len = self.batch_size*training_iters
            training_generator = DatasetGenerator.from_root_folder(PATH_TRAINING, batch_size=self.batch_size)
            
            logging.info("Start optimization")
            
#             avg_gradients = None
            counter=0
            for epoch in range(epochs):
                total_loss = 0
                training_generator_ite=training_generator.shuffled()
                training_generator_ite=training_generator_ite.__iter__()
                
                for step in range((epoch*training_iters), ((epoch+1)*training_iters)):
    
                    batch_x,batch_y =training_generator_ite.__next__()
                    batch_x=standardize(batch_x)
                    
                    # Run optimization op (backprop)
#                     _, loss, lr, gradients = sess.run((self.optimizer, self.net.cost, self.learning_rate_node, self.net.gradients_node), 
#                                                       feed_dict={self.net.X_placeholder: batch_x,
#                                                                  self.net.y_placeholder: batch_y,
#                                                                  self.net.keep_prob: dropout,self.net.phase_train:True})
                    _, loss, lr = sess.run((self.optimizer, self.net.cost, self.learning_rate_node), 
                                                      feed_dict={self.net.X_placeholder: batch_x,
                                                                 self.net.y_placeholder: batch_y,
                                                                 self.net.keep_prob: dropout,self.net.phase_train:True})
                  
#                     avg_gradients = _update_avg_gradients(avg_gradients, gradients, step)
#                     norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
#                     self.norm_gradients_node.assign(norm_gradients).eval()
                    
                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, step, batch_x,batch_y)
                        
                    total_loss += loss
                    loss_train[counter]=loss
                    file_train.write(str(loss_train[counter])+'\n')
                    counter += 1
                    if counter % rec_save == 0:
                        self.net.save(sess,saver,save_path, counter)
                        
                    

                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                
                loss_v,prediction_v=self.store_prediction(sess, X_val, Y_val, "epoch_%s"%epoch,validation_batch_size,False)
                iou_v,f1_v=predict_score_batch(TMP_IOU,Y_val[:,:,:,0],1-np.argmax(prediction_v,3))
                
                loss_verif[epoch]=loss_v
                IOU_verif[epoch]=iou_v
                f1_IOU_verif[epoch]=f1_v
                
                IOU_file_verif.write(str(IOU_verif[epoch])+'\n')
                f1_IOU_file_verif.write(str(f1_IOU_verif[epoch])+'\n')
                file_verif.write(str(loss_verif[epoch])+'\n')
                print("Validation IoU {:.4f}%,Validation F1 IoU {:.4f}%".format(iou_v,f1_v))
                
            self.store_prediction(sess, X_val, Y_val, "epoch_%s"%epoch,validation_batch_size,True)
            save_path=self.net.save(sess,saver,save_path, counter)
                
            logging.info("Optimization Finished!")
            
            return save_path, loss_train,loss_verif,IOU_verif,f1_IOU_verif
        
    def store_prediction(self, sess, batch_x, batch_y, name,validation_batch_size,save_patches):
        prediction = sess.run(self.net.predicter, feed_dict={self.net.X_placeholder: batch_x, 
                                                             self.net.y_placeholder: batch_y, 
                                                             self.net.keep_prob: 1.,self.net.phase_train:False})
        
        loss = sess.run(self.net.cost, feed_dict={self.net.X_placeholder: batch_x, 
                                                       self.net.y_placeholder: batch_y, 
                                                       self.net.keep_prob: 1.,self.net.phase_train:False})

        logging.info("Verification error= {:.1f}%, loss= {:.4f}".format(error_rate(prediction,batch_y),loss))
        if save_patches:
            plot_summary(prediction,batch_y,batch_x[:,:,:,0],validation_batch_size,name,self.prediction_path)
        return loss,prediction

    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info("Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))

    def output_minibatch_stats(self, sess, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        loss, acc, predictions = sess.run([self.net.cost,self.net.accuracy, 
                                                            self.net.predicter], 
                                                           feed_dict={self.net.X_placeholder: batch_x,
                                                                      self.net.y_placeholder: batch_y,
                                                                      self.net.keep_prob: 1.,self.net.phase_train:False})
        logging.info("Iter {:}, Minibatch Loss= {:.4f}, Training Accuracy= {:.4f}, Minibatch error= {:.1f}%".format(step,
                                                                                                                    loss,
                                                                                                                    acc,
                                                                                                                    error_rate(predictions, batch_y)))
            
#         fig,axs=plt.subplots(3, 3,figsize=(8*3,24))

#         axs[0,0].set_title(str(step)+' Panchromatic ', fontsize='large')
#         axs[1,0].set_title(str(step)+' Groundtruth ', fontsize='large')
#         axs[2,0].set_title(str(step)+' Predictions ', fontsize='large')


#         for i in range(3):

#             axs[0,i].imshow(batch_x[i,:,:,0])
#             axs[1,i].imshow(batch_y[i,:,:,0])
#             logits=np.argmax(predictions, 3)
#             axs[2,i].imshow(1-logits[i,:,:])

#         plt.subplots_adjust()
#     #     suptitle.set_y(0.95)
#     #     fig.subplots_adjust(top=0.96)
#         plt.show()


def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """

    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
        (predictions.shape[0]*predictions.shape[1]*predictions.shape[2]))
def plot_summary(predictions,labels,panchro,batch_size,epoch,prediction_path):
    
#     fig,axs=plt.subplots(3, batch_size,figsize=(8*batch_size,24))

#     axs[0,0].set_title(epoch+' Panchromatic ', fontsize='large')
#     axs[1,0].set_title(epoch+' Groundtruth ', fontsize='large')
#     axs[2,0].set_title(epoch+' Predictions ', fontsize='large')

        
    for i in range(batch_size):
        
#         axs[0,i].imshow(panchro[i,:,:])
        plt.imsave(prediction_path+epoch+'_Panchro_'+str(i)+'.jpg',panchro[i,:,:])
#         axs[1,i].imshow(labels[i,:,:,0])
        plt.imsave(prediction_path+epoch+'_Groundtruth_'+str(i)+'.jpg',labels[i,:,:,0])
        logits=np.argmax(predictions, 3)
#         axs[2,i].imshow(1-logits[i,:,:])
        plt.imsave(prediction_path+epoch+'_Predictions_'+str(i)+'.jpg',1-logits[i,:,:])

#     plt.subplots_adjust()
#     suptitle.set_y(0.95)
#     fig.subplots_adjust(top=0.96)
#     plt.show()

# def _update_avg_gradients(avg_gradients, gradients, step):
#     if avg_gradients is None:
#         avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
#     for i in range(len(gradients)):
#         avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step+1)))) + (gradients[i] / (step+1))
        
#     return avg_gradients


if __name__ == '__main__':
    
    #python unet_baseline.py ../DATA_GHANA/DATASET/120_x_120_8_bands/ MODEL_BASIC_TEST_120/ RESUNET_BASIC_TEST.ckpt '' --input_channels=9 --nb_classes=2 --nb_layers=4 --nb_features_root=32 --filters_size=3 --pool_size=2 --batch_size=15 --optimizer=adam --epochs=2 --iterations=3 --dropout=0.9 --display_step=50 --validation_size_batch=70 --rec_save_model=1

    root_folder=sys.argv[1]
    #root_folder = '../DATA_GHANA/DATASET/120_x_120_8_bands/'
    
    ##########
    GLOBAL_PATH=sys.argv[2]
    

    if not os.path.exists(GLOBAL_PATH):
            os.makedirs(GLOBAL_PATH)
    TEST_SAVE=GLOBAL_PATH+'TEST_SAVE/'
    if not os.path.exists(TEST_SAVE):
            os.makedirs(TEST_SAVE)
            
    
    ##########
    
    
    MODEL_PATH_SAVE=GLOBAL_PATH+sys.argv[3]
    MODEL_PATH_RESTORE=sys.argv[4]
    
    for i in range(5, len(sys.argv)):
        arg = sys.argv[i]
        if arg.startswith('--input_channels'):
            INPUT_CHANNELS=int(arg[len('--input_channels='):])
        elif arg.startswith('--nb_classes'):
            NB_CLASSES=int(arg[len('--nb_classes='):])
        elif arg.startswith('--nb_layers'):
            DEFAULT_LAYERS=int(arg[len('--nb_layers='):])
        elif arg.startswith('--nb_features_root'):
            DEFAULT_FEATURES_ROOT=int(arg[len('--nb_features_root='):])
        elif arg.startswith('--filters_size'):
            DEFAULT_FILTERS_SIZE=int(arg[len('--filters_size='):])
        elif arg.startswith('--pool_size'):
            DEFAULT_POOL_SIZE=int(arg[len('--pool_size='):])
        elif arg.startswith('--batch_size'):
            DEFAULT_BATCH_SIZE = int(arg[len('--batch_size='):])
        elif arg.startswith('--optimizer'):
            OPTIMIZER = arg[len('--optimizer='):]
        elif arg.startswith('--epochs'):
            DEFAULT_EPOCHS = int(arg[len('--epochs='):])
        elif arg.startswith('--iterations'):
            DEFAULT_ITERATIONS = int(arg[len('--iterations='):])
        elif arg.startswith('--dropout'):
            DROPOUT = float(arg[len('--dropout='):])
        elif arg.startswith('--display_step'):
            DISPLAY_STEP = int(arg[len('--display_step='):])
        elif arg.startswith('--validation_size_batch'):
            DEFAULT_VALID = int(arg[len('--validation_size_batch='):])  
        elif arg.startswith('--rec_save_model'):
            REC_SAVE = int(arg[len('--rec_save_model='):]) 
        else:
            raise ValueError('Unknown argument %s' % str(arg))


    
    model = CustomCNN(INPUT_CHANNELS,NB_CLASSES,DEFAULT_LAYERS, DEFAULT_FEATURES_ROOT, DEFAULT_FILTERS_SIZE, DEFAULT_POOL_SIZE)


    trainer=Trainer(model,DEFAULT_BATCH_SIZE,OPTIMIZER)
    
    save_path,loss_train,loss_verif,iou_verif,f1_iou_verif=trainer.train( root_folder, MODEL_PATH_SAVE, MODEL_PATH_RESTORE, DEFAULT_ITERATIONS,DEFAULT_EPOCHS,DROPOUT, DISPLAY_STEP, DEFAULT_VALID,REC_SAVE, TEST_SAVE)
    
    
    
    print('Last model saved is %s: '%save_path)
    #SAVE loss
#     plt.title('Plot Loss', fontsize=20)
#     ite = np.arange(0,DEFAULT_EPOCHS*DEFAULT_ITERATIONS,1)
#     epo=np.arange((DEFAULT_ITERATIONS-1),(DEFAULT_EPOCHS*DEFAULT_ITERATIONS+(DEFAULT_ITERATIONS-1)),DEFAULT_ITERATIONS)
#     plt.plot(ite,loss_train,'b',epo,loss_verif,'g')
#     plt.ylabel('Loss')
#     plt.savefig(GLOBAL_PATH+'losses.jpg')
#     plt.show()

#     #SAVE IOU
#     plt.title('Plot IOU', fontsize=20)
#     epo=np.arange((DEFAULT_ITERATIONS-1),(DEFAULT_EPOCHS*DEFAULT_ITERATIONS+(DEFAULT_ITERATIONS-1)),DEFAULT_ITERATIONS)
#     plt.plot(epo,iou_verif,'g')
#     plt.ylabel('IOU in %')
#     plt.show()

    
#      #SAVE f1 IOU
#     plt.title('Plot f1 IOU', fontsize=20)
#     epo=np.arange((DEFAULT_ITERATIONS-1),(DEFAULT_EPOCHS*DEFAULT_ITERATIONS+(DEFAULT_ITERATIONS-1)),DEFAULT_ITERATIONS)
#     plt.plot(epo,f1_iou_verif,'g')
#     plt.ylabel('f1 IOU in %')
#     plt.show()



