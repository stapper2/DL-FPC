# Training of networks using simulated data - OFF and ON data in same model
# Implemented by Sofie Tapper, 2020

from keras.models import Sequential
from keras.layers import Dense
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt 
import numpy as np
import os
import h5py
import random as rn
import cmath as cm

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load Simulated FIDs, which were simulated using FID-A
arrays = {}
f = h5py.File('TrainingData.mat')
for k, v in f.items():
    arrays[k] = np.array(v)    
    
FidsOFFtemp = np.transpose(arrays['OFF_fids']) 
FidsONtemp = np.transpose(arrays['ON_fids'])

# Construct the complex FIDs
FidsOFF = np.empty((len(FidsOFFtemp[:,0]), len(FidsOFFtemp[0,:])), dtype = np.complex_)
FidsON = np.empty((len(FidsONtemp[:,0]), len(FidsONtemp[0,:])), dtype = np.complex_)
for pp in range(len(FidsOFF[:,0])):
    for kk in range(len(FidsOFF[0,:])):
        FidsOFF[pp,kk] = FidsOFFtemp[pp,kk][0] + 1j * FidsOFFtemp[pp,kk][1] 
        FidsON[pp,kk] = FidsONtemp[pp,kk][0] + 1j * FidsONtemp[pp,kk][1]
     
#Concatenate OFF and ON data - Training is performed using both OFF and ON
Fids = np.concatenate((FidsOFF, FidsON), axis=1)     

# Define Offsets - Randomize 
Phs = np.arange(-90, 90.001, (180/(len(Fids[0])-1)))
F0 = np.arange(-20, 20.0000001, (40/(len(Fids[0])-1))) 
rn.shuffle(Phs) #Randomize
rn.shuffle(F0) #Randomize

#Apply Phase and Frequency Offsets to the simulated FIDs
time = np.arange(2048)/2000

FidsUnAligned = np.empty((len(Fids[:,0]), len(Fids[0,:])), dtype = np.complex_)
FidsUnAlignedPhase = np.empty((len(Fids[:,0]), len(Fids[0,:])), dtype = np.complex_)
spectUnAligned = np.empty((len(Fids[:,0]), len(Fids[0,:])), dtype = np.complex_)
spectUnAlignedPhase = np.empty((len(Fids[:,0]), len(Fids[0,:])), dtype = np.complex_)
for kk in range(len(Fids[0,:])):
   
    phasePart = np.exp(-1j * Phs[kk] * cm.pi / 180)
    freqPart = np.exp(-1j * time * F0[kk] * 2 * cm.pi)
    correction = np.multiply(freqPart,phasePart)
    
    FidsUnAligned[:,kk] = np.multiply(Fids[:,kk], correction) #[2048, 10000]
    FidsUnAlignedPhase[:,kk] = np.multiply(Fids[:,kk], phasePart) #[2048, 10000]
    spectUnAligned[:,kk] = fftshift(fft(FidsUnAligned[:,kk]))
    spectUnAlignedPhase[:,kk] = fftshift(fft(FidsUnAlignedPhase[:,kk]))
    
    scale = np.max(np.squeeze(np.abs(spectUnAligned[:,kk])))
    spectUnAligned[:,kk] = spectUnAligned[:,kk]/np.max(np.squeeze(np.abs(spectUnAligned[:,kk])))

    scalePh = np.max(np.squeeze(np.abs(spectUnAlignedPhase[:,kk])))
    spectUnAlignedPhase[:,kk] = spectUnAlignedPhase[:,kk]/scalePh


# Make Training Data suitable for the networks - 1024 samples
trainingData = np.absolute(spectUnAligned[512:-512,:])
trainingData = np.transpose(trainingData) #Transpose to [numSpect 1024]
trainingDataPhase = np.real(spectUnAlignedPhase[512:-512,:])
trainingDataPhase = np.transpose(trainingDataPhase) #Transpose to [numSpect 1024]

# Create Network Model for F0 Offset Estimation
model = Sequential()
model.add(Dense(1024,  kernel_initializer='normal', input_dim = len(trainingData[0,:]), activation = 'relu')) #Fully Connected Layer + ReLu
model.add(Dense(512,  kernel_initializer='normal', activation = 'relu')) #Fully Connected Layer + ReLU
model.add(Dense(1,  kernel_initializer='normal', activation = 'linear')) #Fully Connected Layer + Linear

eepochs = 500;

# Train Network Model
model.compile(loss='mean_absolute_error', optimizer = 'adam', metrics=['mse'])
hist_F0 = model.fit(trainingData, F0, epochs = eepochs, batch_size = 64)
model.save('Model_F0.model')

# Create Network Model for Phase Offset Estimation
model = Sequential()
model.add(Dense(1024,  kernel_initializer='normal', input_dim = len(trainingDataPhase[0,:]), activation = 'relu')) #Fully Connected Layer + ReLu
model.add(Dense(512,  kernel_initializer='normal', activation = 'relu')) #Fully Connected Layer + ReLU
model.add(Dense(1,  kernel_initializer='normal', activation = 'linear')) #Fully Connected Layer + Linear

# Train Network Model
model.compile(loss='mean_absolute_error', optimizer = 'adam', metrics=['mse'])
hist_Ph = model.fit(trainingDataPhase, Phs, epochs = eepochs, batch_size = 64)
model.save('Model_Ph.model')

# %% Plot Results

# Plot Loss During Training - F0
fig = plt.figure(figsize=(8/2.54, 5/2.54))
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)       
ax.spines["right"].set_visible(False)  

lw = 1
plt.plot(hist_F0.history['loss'],'b',linewidth=lw)

plt.xlabel('Epochs', fontname="Arial", fontsize=9, fontweight='bold')
ax.tick_params(width=1,color='k',length=2)
plt.xticks(np.arange(0,501,step=100), fontsize=9)
plt.xlim((-8, 501))
plt.ylabel('Loss', fontname="Arial", fontsize=9, fontweight='bold')
plt.yticks(np.arange(0,0.51,step=0.5), ['0', '0.5'], fontsize=9,fontname="Arial")
plt.ylim((0,0.5))

plt.subplots_adjust(left=0.2, right=0.95, top=0.95,bottom=0.25)

# Plot Loss During Training - Phase
fig = plt.figure(figsize=(8/2.54, 5/2.54))
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)       
ax.spines["right"].set_visible(False)  

lw = 1
plt.plot(hist_Ph.history['loss'],'b',linewidth=lw)

plt.xlabel('Epochs', fontname="Arial", fontsize=9, fontweight='bold')
ax.tick_params(width=1,color='k',length=2)
plt.xticks(np.arange(0,501,step=100), fontsize=9)
plt.xlim((-8, 501))
plt.ylabel('Loss', fontname="Arial", fontsize=9, fontweight='bold')
plt.yticks(np.arange(0,2.001,step=2), ['0', '2'], fontsize=9,fontname="Arial")
plt.ylim((0,2))

plt.subplots_adjust(left=0.2, right=0.95, top=0.95,bottom=0.2)


