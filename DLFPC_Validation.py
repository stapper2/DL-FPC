# Validation of networks using simulated data with random offsets.
# Implemented by Sofie Tapper, 2020

from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import h5py
import random as rn
import cmath as cm
from matplotlib import rcParams

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load validation data that were simulated using FID-A.
arrays = {}
f = h5py.File('ValidationData.mat')
for k, v in f.items():
    arrays[k] = np.array(v)    
    
FidsOFFtemp = np.transpose(arrays['OFF_fids'])
FidsONtemp = np.transpose(arrays['ON_fids'])

# Construct complex FIDs
FidsOFF = np.empty((len(FidsOFFtemp[:,0]), len(FidsOFFtemp[0,:])), dtype = np.complex_)
FidsON = np.empty((len(FidsONtemp[:,0]), len(FidsONtemp[0,:])), dtype = np.complex_)
for pp in range(len(FidsON[:,0])):
    for kk in range(len(FidsON[0,:])):
        FidsOFF[pp,kk] = FidsOFFtemp[pp,kk][0] + 1j * FidsOFFtemp[pp,kk][1]  
        FidsON[pp,kk] = FidsONtemp[pp,kk][0] + 1j * FidsONtemp[pp,kk][1]       
   
# Define Offsets - Randomize 
PhsOFF = np.arange(-90, 90.0001, (180/(len(FidsOFF[0])-1))) #Take 10000 phase offsets
F0OFF = np.arange(-20, 20.0000001, (40/(len(FidsOFF[0])-1))) #Take 10000 frequency offsets
rn.shuffle(PhsOFF) #Randomize
rn.shuffle(F0OFF) #Randomize

PhsON = np.arange(-90, 90.0001, (180/(len(FidsON[0])-1))) #Take 160 phase offsets
F0ON = np.arange(-20, 20.0000001, (40/(len(FidsON[0])-1))) #Take 160 frequency offsets
rn.shuffle(PhsON) #Randomize
rn.shuffle(F0ON) #Randomize

#Apply Phase and Frequency Offsets to the simulated FIDs
time = np.arange(2048)/2000
FidsUnAlignedOFF = np.empty((len(FidsOFFtemp[:,0]), len(FidsOFFtemp[0,:])), dtype = np.complex_)
spectUnAlignedOFF = np.empty((len(FidsOFFtemp[:,0]), len(FidsOFFtemp[0,:])), dtype = np.complex_)
spectAlignedOFF = np.empty((len(FidsOFFtemp[:,0]), len(FidsOFFtemp[0,:])), dtype = np.complex_)
FidsUnAlignedON = np.empty((len(FidsONtemp[:,0]), len(FidsONtemp[0,:])), dtype = np.complex_)
spectUnAlignedON = np.empty((len(FidsONtemp[:,0]), len(FidsONtemp[0,:])), dtype = np.complex_)
spectAlignedON = np.empty((len(FidsONtemp[:,0]), len(FidsONtemp[0,:])), dtype = np.complex_)
for kk in range(len(PhsOFF)):
    
    spectAlignedOFF[:,kk] = fftshift(fft(FidsOFF[:,kk]))
    
    phasePartOFF = np.exp(-1j * PhsOFF[kk] * cm.pi / 180)
    freqPartOFF = np.exp(-1j * time * F0OFF[kk] * 2 * cm.pi)
    correctionOFF = np.multiply(freqPartOFF,phasePartOFF)
    
    FidsUnAlignedOFF[:,kk] = np.multiply(FidsOFF[:,kk], correctionOFF) #[2048, 500]
    spectUnAlignedOFF[:,kk] = fftshift(fft(FidsUnAlignedOFF[:,kk]))
    
    spectAlignedON[:,kk] = fftshift(fft(FidsON[:,kk]))
    
    phasePartON = np.exp(-1j * PhsON[kk] * cm.pi / 180)
    freqPartON = np.exp(-1j * time * F0ON[kk] * 2 * cm.pi)
    correctionON = np.multiply(freqPartON,phasePartON)
    
    FidsUnAlignedON[:,kk] = np.multiply(FidsON[:,kk], correctionON) #[2048, 500]
    spectUnAlignedON[:,kk] = fftshift(fft(FidsUnAlignedON[:,kk]))
    
    scaleOFF = np.max(np.squeeze(np.abs(spectUnAlignedOFF[:,kk])))
    spectUnAlignedOFF[:,kk] = spectUnAlignedOFF[:,kk]/scaleOFF
    
    scaleON = np.max(np.squeeze(np.abs(spectUnAlignedON[:,kk])))
    spectUnAlignedON[:,kk] = spectUnAlignedON[:,kk]/scaleON

# Make data appropriate for the networks
validationDataOFF = np.absolute(spectUnAlignedOFF[512:-512,:])
validationDataOFF = np.transpose(validationDataOFF)

validationDataON = np.absolute(spectUnAlignedON[512:-512,:])
validationDataON = np.transpose(validationDataON)

# Load Trained Model - F0 OFF
model = tf.keras.models.load_model("Model_F0.model")
prediction_F0_OFF = model.predict(validationDataOFF)
error_F0_OFF = F0OFF-np.transpose(prediction_F0_OFF)

print("OFF F0: Mean Absolute Error: ", np.mean(np.absolute(error_F0_OFF)), " +- ", np.std(np.absolute(error_F0_OFF)))

# Perform Frequency Correction
FidsUnAlignedOFFPhase = np.empty((len(FidsOFFtemp[:,0]), len(FidsOFFtemp[0,:])), dtype = np.complex_)
spectUnAlignedOFFPhase = np.empty((len(FidsOFFtemp[:,0]), len(FidsOFFtemp[0,:])), dtype = np.complex_)
for kk in range(len(PhsOFF)):
    freqPart = np.exp(1j * time * prediction_F0_OFF[kk] * 2 * cm.pi)
    FidsUnAlignedOFFPhase[:,kk] = np.multiply(FidsUnAlignedOFF[:,kk], freqPart)
    spectUnAlignedOFFPhase[:,kk] = fftshift(fft(FidsUnAlignedOFFPhase[:,kk]))

    scaleOFF = np.max(np.squeeze(np.abs(spectUnAlignedOFFPhase[:,kk])))
    spectUnAlignedOFFPhase[:,kk] = spectUnAlignedOFFPhase[:,kk]/scaleOFF
    
validationDataOFFPhase = np.real(spectUnAlignedOFFPhase[512:-512,:])
validationDataOFFPhase = np.transpose(validationDataOFFPhase)

# The same for ON data
prediction_F0_ON = model.predict(validationDataON)
error_F0_ON = F0ON-np.transpose(prediction_F0_ON)

print("ON F0: Mean Absolute Error: ", np.mean(np.absolute(error_F0_ON)), " +- ", np.std(np.absolute(error_F0_ON)))

# Perform Frequency Correction
FidsUnAlignedONPhase = np.empty((len(FidsONtemp[:,0]), len(FidsONtemp[0,:])), dtype = np.complex_)
spectUnAlignedONPhase = np.empty((len(FidsONtemp[:,0]), len(FidsONtemp[0,:])), dtype = np.complex_)
for kk in range(len(PhsON)):
    freqPart = np.exp(1j * time * prediction_F0_ON[kk] * 2 * cm.pi)
    FidsUnAlignedONPhase[:,kk] = np.multiply(FidsUnAlignedON[:,kk], freqPart)
    spectUnAlignedONPhase[:,kk] = fftshift(fft(FidsUnAlignedONPhase[:,kk]))

    scaleON = np.max(np.squeeze(np.abs(spectUnAlignedONPhase[:,kk])))
    spectUnAlignedONPhase[:,kk] = spectUnAlignedONPhase[:,kk]/scaleON

validationDataONPhase = np.real(spectUnAlignedONPhase[512:-512,:])
validationDataONPhase = np.transpose(validationDataONPhase)

# Load Trained Model - Phase
model = tf.keras.models.load_model("Model_Ph.model")
prediction_Ph_OFF = model.predict(validationDataOFFPhase)
error_Ph_OFF = PhsOFF-np.transpose(prediction_Ph_OFF)

print("OFF Ph: Mean Absolute Error: ", np.mean(np.absolute(error_Ph_OFF)), " +- ", np.std(np.absolute(error_Ph_OFF)))

# Perform Phase Correction
FidsOFFCorrected = np.empty((len(FidsOFFtemp[:,0]), len(FidsOFFtemp[0,:])), dtype = np.complex_)
spectOFFCorrected = np.empty((len(FidsOFFtemp[:,0]), len(FidsOFFtemp[0,:])), dtype = np.complex_)
for kk in range(len(PhsOFF)):
    phasePart = np.exp(1j * prediction_Ph_OFF[kk] * cm.pi / 180)
    FidsOFFCorrected[:,kk] = np.multiply(FidsUnAlignedOFFPhase[:,kk], phasePart)
    spectOFFCorrected[:,kk] = fftshift(fft(FidsOFFCorrected[:,kk]))

# Load Trained Model - Ph ON
prediction_Ph_ON = model.predict(validationDataONPhase)
error_Ph_ON = PhsON-np.transpose(prediction_Ph_ON)

print("ON Ph: Mean Absolute Error: ", np.mean(np.absolute(error_Ph_ON)), " +- ", np.std(np.absolute(error_Ph_ON)))

# Perform Phase Correction
FidsONCorrected = np.empty((len(FidsONtemp[:,0]), len(FidsONtemp[0,:])), dtype = np.complex_)
spectONCorrected = np.empty((len(FidsONtemp[:,0]), len(FidsONtemp[0,:])), dtype = np.complex_)
for kk in range(len(PhsON)):
    phasePart = np.exp(1j * prediction_Ph_ON[kk] * cm.pi / 180)
    FidsONCorrected[:,kk] = np.multiply(FidsUnAlignedONPhase[:,kk], phasePart)
    spectONCorrected[:,kk] = fftshift(fft(FidsONCorrected[:,kk]))


# %% Creating the validation figure

fig = plt.figure(figsize=(8.67/2.54, 4.4/2.54))

ax = plt.subplot(121)   
plt.rc('axes', linewidth=1)

ax.spines["top"].set_visible(False)       
ax.spines["right"].set_visible(False)     
ax.tick_params(width=1,color='k',length=2)
plt.yticks(np.arange(-0.15,0.16, step=0.15),['-0.15','0','0.15'], fontsize=8)
plt.xticks(np.arange(-20,21,step=20),['-20','0','20'], fontsize=8)
plt.ylabel('DL - True / Hz', fontname="Arial", fontsize=9, fontweight='bold')
plt.xlabel('True / Hz', fontname="Arial", fontsize=9, fontweight='bold')
plt.xlim((-20,21))
plt.ylim((-0.15, 0.15))
plt.plot(F0OFF, np.squeeze(prediction_F0_OFF) - F0OFF, 'ko', markersize=1)
plt.plot(F0ON, np.squeeze(prediction_F0_ON) - F0ON, 'ko', markersize=1)
plt.title('Frequency Offsets', fontname="Arial", fontsize=9, fontweight='bold')

t1 = np.mean(np.squeeze(prediction_F0_OFF) - F0OFF)
t2 = np.mean(np.squeeze(prediction_F0_ON) - F0ON)
t3 = (t1 + t2)/2
e1 = np.std(np.squeeze(prediction_F0_OFF) - F0OFF)
e2 = np.std(np.squeeze(prediction_F0_ON) - F0ON)
e3 = (e1 + e2)/2

rcParams['mathtext.default'] = 'regular'
plt.text(-0.57*20,-0.13,r'$ {0:.2f} \pm {1:.2f} \/Hz$'.format(t3,e3),fontsize=9,fontname="Arial",color='black')
     
ax = plt.subplot(122)
ax.spines["top"].set_visible(False)       
ax.spines["right"].set_visible(False)     
ax.tick_params(width=1,color='k',length=2)
plt.yticks(np.arange(-1.5,1.6,step=1.5),['-1.5','0','1.5'], fontsize=8)
plt.xticks(np.arange(-90,91,step=90),['-90','0','90'], fontsize=8)
plt.ylabel('DL - True /' r' $\degree $', fontname="Arial", fontsize=9, fontweight='bold')
plt.xlabel('True / ' r'$\degree $', fontname="Arial", fontsize=9, fontweight='bold')
plt.xlim((-90,94))
plt.ylim((-1.5, 1.5))
plt.plot(PhsOFF, np.squeeze(prediction_Ph_OFF) - PhsOFF, 'ko', markersize=1)
plt.plot(PhsON, np.squeeze(prediction_Ph_ON) - PhsON, 'ko', markersize=1)
plt.title('Phase Offsets', fontname="Arial", fontsize=9, fontweight='bold')

t1 = np.mean(np.squeeze(prediction_Ph_OFF) - PhsOFF)
t2 = np.mean(np.squeeze(prediction_Ph_ON) - PhsON)
t3 = (t1 + t2)/2
e1 = np.std(np.squeeze(prediction_Ph_OFF) - PhsOFF)
e2 = np.std(np.squeeze(prediction_Ph_ON) - PhsON)
e3 = (e1 + e2)/2

plt.text(-0.72*90,-1.3,r'$ {0:.2f} \pm {1:.2f} \/\degree$'.format(t3,e3),fontsize=9,fontname="Arial",color='black')
plt.subplots_adjust(wspace=0.6,left=0.18, right=0.98, top=0.8, bottom=0.25)
    
freqRange = 2000/127.7
ppm = np.flip((np.arange(1,len(FidsOFFtemp[:,0]) + 1)) / len(FidsOFFtemp[:,0]) * freqRange + 4.68 - freqRange/2)
xlim = [1, 4.2]

fig = plt.figure(figsize=(8.67/2.54, 4.5/2.54))
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)       
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)   
ax.tick_params(width=1,color='k',length=2)
plt.xticks(np.arange(1,5,step=1), fontsize=9)
ax.set_yticklabels([])
ax.axes.get_yaxis().set_visible(False)  

lw = 1
plt.plot(ppm, np.mean(spectONCorrected, axis=1).real - np.mean(spectOFFCorrected, axis=1).real - (np.mean(spectAlignedON, axis=1).real - np.mean(spectAlignedOFF, axis=1).real) + 0.017,'k',linewidth=0.5)
plt.plot(ppm, np.mean(spectONCorrected, axis=1).real - np.mean(spectOFFCorrected, axis=1).real,'k',linewidth=lw)
plt.plot(ppm, np.mean(spectAlignedON, axis=1).real - np.mean(spectAlignedOFF, axis=1).real - 0.012,'k',linewidth=lw)

plt.xlabel('ppm', fontname="Arial", fontsize=9, fontweight='bold')
plt.xlim((xlim))
plt.gca().invert_xaxis()
plt.ylim((-0.02,0.02))
ax.text(4.2,0.018,'DL - True',fontsize=9,fontname="Arial",color='black', fontweight='bold')
ax.text(4.2,0.002,"DL",fontsize=9,fontname="Arial",color='black', fontweight='bold')
ax.text(4.2,-0.01,"True",fontsize=9,fontname="Arial",color='black', fontweight='bold')

plt.subplots_adjust(left=0.03, right=0.98, top=0.95,bottom=0.25)
