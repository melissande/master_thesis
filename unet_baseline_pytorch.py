import sys
import numpy as np
import os
import cv2
import logging
from image_utils import standardize
import matplotlib.pyplot as plt
from dataset_generator import DatasetGenerator
import torch
import torch.nn.functional as Fu
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from unet_val import UNet
import torch.backends.cudnn as cudnn
from IOU_computations import *


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
MODEL_PATH_SAVE=GLOBAL_PATH+'RESUNET_pytorch_BASIC_test.ckpt'
MODEL_PATH_RESTORE=''
TEST_SAVE=GLOBAL_PATH+'TEST_SAVE/'
if not os.path.exists(TEST_SAVE):
            os.makedirs(TEST_SAVE)
        
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

##############

REC_SAVE=2000#2000
DROPOUT=0.9#0.9
DEFAULT_BATCH_SIZE = 10#10
DEFAULT_EPOCHS = 5#50
DEFAULT_ITERATIONS =495#495
DEFAULT_VALID=100#100
DISPLAY_STEP=50#50

###############
DEFAULT_LAYERS=3
DEFAULT_FEATURES_ROOT=32
DEFAULT_FILTERS_SIZE=3
DEFAULT_LR=0.0001


        
class Trainer(object):
    """
    Trains a unet instance
    
    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param lr: learning rate
    """
    def __init__(self, net, batch_size=10, lr=0.0001):
        self.net = net
        self.batch_size = batch_size
        self.lr = lr
 
    
    def _initialize(self, prediction_path):
        
        self.optimizer = optim.Adam(self.net.parameters(),lr=self.lr)
        
        self.prediction_path = prediction_path
        
    
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

        
        if epochs == 0:
            return save_path
        if save_path=='':
            return 'Specify a path where to store the Model'
        self._initialize(prediction_path)
            
        if restore_path=='':
            print('Model trained from scratch')
            loss_train,file_train,loss_verif,file_verif,IOU_verif,IOU_file_verif,IOU_acc_verif,IOU_acc_file_verif,f1_IOU_verif,f1_IOU_file_verif=save_metrics(epochs,training_iters,TEST_SAVE,'w')
        else:
            self.net.load_state_dict(torch.load(restore_path))
            print('Model loaded from {}'.format(restore_path))
            loss_train,file_train,loss_verif,file_verif,IOU_verif,IOU_file_verif,IOU_acc_verif,IOU_acc_file_verif,f1_IOU_verif,f1_IOU_file_verif=save_metrics(epochs,training_iters,TEST_SAVE,'a')
            

        val_generator = DatasetGenerator.from_root_folder(PATH_VALIDATION, batch_size=validation_batch_size)
        val_generator=val_generator.shuffled()
        val_generator=val_generator.__iter__()
        X_val,Y_val=val_generator.__next__()
        X_val=standardize(X_val)
        
        
        self.store_validation(X_val, Y_val, "_init",validation_batch_size,save_patches=True)

        train_len = self.batch_size*training_iters
        training_generator = DatasetGenerator.from_root_folder(PATH_TRAINING, batch_size=self.batch_size)

        logging.info("Start optimization")

        counter=0
        for epoch in range(epochs):
            total_loss = 0
            training_generator_ite=training_generator.shuffled()
            training_generator_ite=training_generator_ite.__iter__()

            for step in range((epoch*training_iters), ((epoch+1)*training_iters)):

                batch_x,batch_y =training_generator_ite.__next__()
                batch_x=standardize(batch_x)
                prediction,loss=self.predict(batch_x,batch_y)
                # Run optimization op (backprop)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                if step % display_step == 0:
                    self.output_minibatch_stats(step, batch_x,batch_y)

                total_loss += loss.data[0]
                loss_train[counter]=loss
                file_train.write(str(loss_train[counter])+'\n')
                counter += 1
                if counter % rec_save == 0:
                    torch.save(self.net.state_dict(),save_path + 'CP{}.pth'.format(counter))
                    print('Checkpoint {} saved !'.format(counter))



            self.output_epoch_stats(epoch, total_loss, training_iters, self.lr)
            loss_v,prediction_v=self.store_validation( X_val, Y_val, "epoch_%s"%epoch,validation_batch_size,save_patches=False)
            iou_acc_v,f1_v,iou_v=predict_score_batch(TMP_IOU,Y_val[:,:,:,0],1-np.argmax(prediction_v,3))
            
            
            IOU_verif[epoch]=iou_v
            IOU_acc_verif[epoch]=iou_acc_v
            f1_IOU_verif[epoch]=f1_v
            loss_verif[epoch]=loss_v
            
            IOU_file_verif.write(str(IOU_verif[epoch])+'\n')
            IOU_acc_file_verif.write(str(IOU_acc_verif[epoch])+'\n')
            f1_IOU_file_verif.write(str(f1_IOU_verif[epoch])+'\n')
            file_verif.write(str(loss_verif[epoch])+'\n')
            
            print("Validation IoU {:.4f}%, Validation IoU_acc {:.4f}%,Validation F1 IoU {:.4f}%".format(iou_v,iou_acc_v,f1_v))
            
            
        torch.save(self.net.state_dict(),save_path + 'CP{}.pth'.format(counter))
        loss_v=self.store_validation( X_val, Y_val, "epoch_%s"%epoch,validation_batch_size,save_patches=True)
        print('Checkpoint {} saved !'.format(counter))

        logging.info("Optimization Finished!")

        return save_path, loss_train,loss_verif,IOU_verif,IOU_acc_verif,f1_IOU_verif
    
    
    def predict(self,batch_x,batch_y):
        X=np.transpose(batch_x, axes=[0,3,1,2])
        X = torch.FloatTensor(X)
        X = Variable(X).cuda()
        Y=np.transpose(batch_y, axes=[0,3,1,2])
        Y = torch.FloatTensor(Y)
        Y = Variable(Y).cuda()
        
        y_pred=self.net(X)
        probs = Fu.softmax(y_pred,dim=1)
        loss=Fu.binary_cross_entropy_with_logits(probs,Y)
        probs=probs.data.cpu().numpy()
        probs=np.transpose(probs, axes=[0,2,3,1])
        return probs,loss
            

    def store_validation(self,batch_x, batch_y, name,validation_batch_size,*,save_patches=True):
        
        prediction,loss=self.predict(batch_x,batch_y)
        loss=loss.data[0]
        logging.info("Verification error= {:.1f}%, loss= {:.4f}".format(error_rate(prediction,batch_y),loss))
        pansharp=np.stack((batch_x[:,:,:,5],batch_x[:,:,:,3],batch_x[:,:,:,2]),axis=3)
        plot_summary(prediction,batch_y,pansharp,validation_batch_size,name,self.prediction_path,save_patches)

        return loss,prediction
    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info("Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))

    def output_minibatch_stats(self, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        predictions,loss=self.predict(batch_x,batch_y)
        loss=loss.data[0]
        logging.info("Iter {:}, Minibatch Loss= {:.4f}, Training Accuracy= {:.4f}, Minibatch error= {:.1f}%".format(step,
                                                                                                                    loss,
                                                                                                                    accuracy_(predictions,batch_y),
                                                                                                                    error_rate(predictions, batch_y)))

    # #         fig,axs=plt.subplots(3, 3,figsize=(8*3,24))

    # #         axs[0,0].set_title(str(step)+' Panchromatic ', fontsize='large')
    # #         axs[1,0].set_title(str(step)+' Groundtruth ', fontsize='large')
    # #         axs[2,0].set_title(str(step)+' Predictions ', fontsize='large')


    # #         for i in range(3):

    # #             axs[0,i].imshow(batch_x[i,:,:,0])
    # #             axs[1,i].imshow(batch_y[i,:,:,0])
    # #             logits=np.argmax(predictions, 3)
    # #             axs[2,i].imshow(1-logits[i,:,:])

    # #         plt.subplots_adjust()
    # #     #     suptitle.set_y(0.95)
    # #     #     fig.subplots_adjust(top=0.96)
    # #         plt.show()
    
    
def predict_pytorch(net,batch_x,batch_y):
    X=np.transpose(batch_x, axes=[0,3,1,2])
    X = torch.FloatTensor(X)
    X = Variable(X).cuda()
    Y=np.transpose(batch_y, axes=[0,3,1,2])
    Y = torch.FloatTensor(Y)
    Y = Variable(Y).cuda()

    y_pred=net(X)
    probs = Fu.softmax(y_pred,dim=1)
    loss=Fu.binary_cross_entropy_with_logits(probs,Y)
    probs=probs.data.cpu().numpy()
    probs=np.transpose(probs, axes=[0,2,3,1])
    return probs,loss.data[0]


def save_metrics(epochs,training_iters,prediction_path,mode):
    #STORE loss for ANALYSIS
    loss_train=np.zeros(training_iters*epochs)
    file_train = open(prediction_path+'loss_train.txt',mode) 
    loss_verif=np.zeros(epochs)
    file_verif = open(prediction_path+'loss_verif.txt',mode) 
    #STORE IOU for ANALYSIS
    IOU_verif=np.zeros(epochs)
    IOU_file_verif = open(prediction_path+'iou_verif.txt',mode)
    #STORE IOU_ACC for ANALYSIS
    IOU_acc_verif=np.zeros(epochs)
    IOU_acc_file_verif = open(prediction_path+'iou_acc_verif.txt',mode)
    #STORE f1_IOU for ANALYSIS
    f1_IOU_verif=np.zeros(epochs)
    f1_IOU_file_verif = open(prediction_path+'f1_iou_verif.txt',mode) 
    
    return loss_train,file_train,loss_verif,file_verif,IOU_verif,IOU_file_verif,IOU_acc_verif,IOU_acc_file_verif,f1_IOU_verif,f1_IOU_file_verif
 
def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """

    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
        (predictions.shape[0]*predictions.shape[1]*predictions.shape[2]))

def accuracy_(predictions, labels):
    return 100.0 *np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /(predictions.shape[0]*predictions.shape[1]*predictions.shape[2])
    
    
def plot_summary(predictions,labels,pansharp,batch_size,epoch,prediction_path,save_patches):
    
#     fig,axs=plt.subplots(3, batch_size,figsize=(8*batch_size,24))

#     axs[0,0].set_title(epoch+' Pansharpened ', fontsize='large')
#     axs[1,0].set_title(epoch+' Groundtruth ', fontsize='large')
#     axs[2,0].set_title(epoch+' Predictions ', fontsize='large')

        
    for i in range(batch_size):
        
#         axs[0,i].imshow(pansharp[i]) 
#         axs[1,i].imshow(labels[i,:,:,0])
        logits=np.argmax(predictions, 3)
#         axs[2,i].imshow(1-logits[i,:,:])
        
        if save_patches:
            plt.imsave(prediction_path+epoch+'_Pansharp_'+str(i)+'.jpg',pansharp[i])
            plt.imsave(prediction_path+epoch+'_Groundtruth_'+str(i)+'.jpg',labels[i,:,:,0])
            plt.imsave(prediction_path+epoch+'_Predictions_'+str(i)+'.jpg',1-logits[i,:,:])

#     plt.subplots_adjust()
#     suptitle.set_y(0.95)
#     fig.subplots_adjust(top=0.96)
#     plt.show()





if __name__ == '__main__':
    #python unet_baseline_pytorch.py ../DATA_GHANA/DATASET/120_x_120_8_bands/ MODEL_BASIC_TEST_120/ RESUNET_BASIC_TEST.ckpt '' --input_channels=9 --nb_classes=2 --nb_layers=3 --nb_features_root=32  --learning_rate=0.0001 --batch_size=10  --epochs=2 --iterations=495 --dropout=0.9 --display_step=50 --validation_size_batch=100 --rec_save_model=1
    
    root_folder=sys.argv[1]
#     root_folder = '../DATA_GHANA/DATASET/120_x_120_8_bands/'
    
    
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
        elif arg.startswith('--learning_rate'):
            DEFAULT_LR=float(arg[len('--learning_rate='):])
        elif arg.startswith('--batch_size'):
            DEFAULT_BATCH_SIZE = int(arg[len('--batch_size='):])
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
    
    
    model=UNet(INPUT_CHANNELS,NB_CLASSES,DEFAULT_LAYERS,DEFAULT_FEATURES_ROOT,DROPOUT)
    model.cuda()
    cudnn.benchmark = True
    

    trainer=Trainer(model,DEFAULT_BATCH_SIZE,DEFAULT_LR)
    
    
    save_path,loss_train,loss_verif,iou_verif,iou_acc_verif,f1_iou_verif=trainer.train( root_folder, MODEL_PATH_SAVE, MODEL_PATH_RESTORE, DEFAULT_ITERATIONS,DEFAULT_EPOCHS,DROPOUT, DISPLAY_STEP, DEFAULT_VALID,REC_SAVE, TEST_SAVE)
    print('Last model saved is %s: '%save_path)
#     #SAVE PSNR
#     plt.title('Plot Loss', fontsize=20)
#     ite = np.arange(0,DEFAULT_EPOCHS*DEFAULT_ITERATIONS,1)
#     epo=np.arange((DEFAULT_ITERATIONS-1),(DEFAULT_EPOCHS*DEFAULT_ITERATIONS+(DEFAULT_ITERATIONS-1)),DEFAULT_ITERATIONS)
#     plt.plot(ite,loss_train,'b',epo,loss_verif,'g')
#     plt.ylabel('Loss')
#     plt.show()
    
#      #SAVE IOU
#     plt.title('Plot IOU', fontsize=20)
#     epo=np.arange((DEFAULT_ITERATIONS-1),(DEFAULT_EPOCHS*DEFAULT_ITERATIONS+(DEFAULT_ITERATIONS-1)),DEFAULT_ITERATIONS)
#     plt.plot(epo,iou_verif,'g')
#     plt.ylabel('IOU in %')
#     plt.show()

#     #SAVE IOU  acc
#     plt.title('Plot IOU Accuracy', fontsize=20)
#     epo=np.arange((DEFAULT_ITERATIONS-1),(DEFAULT_EPOCHS*DEFAULT_ITERATIONS+(DEFAULT_ITERATIONS-1)),DEFAULT_ITERATIONS)
#     plt.plot(epo,iou_acc_verif,'g')
#     plt.ylabel('IOU Accuracy in %')
#     plt.show()

    
#      #SAVE f1 IOU
#     plt.title('Plot f1 IOU', fontsize=20)
#     epo=np.arange((DEFAULT_ITERATIONS-1),(DEFAULT_EPOCHS*DEFAULT_ITERATIONS+(DEFAULT_ITERATIONS-1)),DEFAULT_ITERATIONS)
#     plt.plot(epo,f1_iou_verif,'g')
#     plt.ylabel('f1 IOU in %')
#     plt.show()



