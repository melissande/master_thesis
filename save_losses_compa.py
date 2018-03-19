import sys
import numpy as np
import os
import matplotlib.pyplot as plt



FILE_NAME='LOSSES_COMPARISON/number_bands'
        

f1='MODEL_BASIC_120/TEST_SAVE/'
f2='MODEL_BASIC_panchro_MS_120/TEST_SAVE/'
f3='MODEL_BASIC_panchro_pansharp_4_120/TEST_SAVE/'
f4='MODEL_BASIC_panchro_pansharp_4_MS_8_120/TEST_SAVE/'

leg=['Panchro+Pansharp 8','Panchro+MS 8','Panchro+Pansharp 4','Panchro+Pansharp 4+MS 8']
#TRAIN
plt.subplot(111)


with open(f1+'loss_train.txt', "r") as ins:
    t1 = []
    for line in ins:
        t1.append(line)
del t1[0]    
t1=np.asarray(t1)
ite1 = np.arange(0,len(t1),1)



with open(f2+'loss_train.txt', "r") as ins:
    t2 = []
    for line in ins:
        t2.append(line)
del t2[0] 
t2=np.asarray(t2)
ite2 = np.arange(0,len(t2),1)

with open(f3+'loss_train.txt', "r") as ins:
    t3 = []
    for line in ins:
        t3.append(line)
del t3[0] 
t3=np.asarray(t3)
ite3 = np.arange(0,len(t3),1)

with open(f4+'loss_train.txt', "r") as ins:
    t4 = []
    for line in ins:
        t4.append(line)
del t4[0] 
t4=np.asarray(t4)
ite4 = np.arange(0,len(t4),1)


plt.plot(ite1,t1,'b', label=leg[0])
plt.plot(ite2,t2,'g', label=leg[1])
plt.plot(ite3,t3,'r', label=leg[2])
plt.plot(ite4,t4,'y', label=leg[3])
plt.title('TRAINING')
plt.ylabel('LOSS')
plt.xlabel('iterations')
plt.legend(loc='lower right',fontsize='small') 
plt.show()
# plt.savefig(FILE_NAME+'_train_loss.png',dpi=120)


#TEST
plt.subplot(111)

with open(f1+'loss_verif.txt', "r") as ins:
    t1 = []
    for line in ins:
        t1.append(line)
del t1[0]    
t1=np.asarray(t1)
ite1 = np.arange(0,len(t1),1)



with open(f2+'loss_verif.txt', "r") as ins:
    t2 = []
    for line in ins:
        t2.append(line)
del t2[0] 
t2=np.asarray(t2)
ite2 = np.arange(0,len(t2),1)

with open(f3+'loss_verif.txt', "r") as ins:
    t3 = []
    for line in ins:
        t3.append(line)
del t3[0] 
t3=np.asarray(t3)
ite3 = np.arange(0,len(t3),1)


with open(f4+'loss_verif.txt', "r") as ins:
    t4 = []
    for line in ins:
        t4.append(line)
del t4[0] 
t4=np.asarray(t4)
ite4 = np.arange(0,len(t4),1)


plt.plot(ite1,t1,'b', label=leg[0])
plt.plot(ite2,t2,'g', label=leg[1])
plt.plot(ite3,t3,'r', label=leg[2])
plt.plot(ite4,t4,'y', label=leg[3])
plt.title('VERIFICATION')
plt.ylabel('LOSS')
plt.xlabel('epochs')
plt.legend(loc='lower right',fontsize='small') 
plt.show()
# plt.savefig(FILE_NAME+'_verif_loss.png',dpi=120)


