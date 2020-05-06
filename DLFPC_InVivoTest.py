# In vivo test of the networks and comparison to Spectral Registration implemented in Gannet
# Three different amounts of added offsets are investigated
# Implemented by Sofie Tapper, 2020

from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import h5py
import cmath as cm

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# %% Compute results for small added offsets

# Load in vivo data - 33 datasets with 160 ON and 160 OFF
arrays = {}
f = h5py.File('InVivoData.mat')
for k, v in f.items():
    arrays[k] = np.array(v)    
    
FidsOFFtemp = np.transpose(arrays['OFFdata']) #OFF data
FidsONtemp = np.transpose(arrays['ONdata']) #ON data

# Construct the OFF and ON Fids
FidsOFF = np.empty((FidsOFFtemp.shape), dtype = np.complex_)
FidsON = np.empty((FidsONtemp.shape), dtype = np.complex_)
for pp in range(len(FidsON[:,0,0])):
    for kk in range(len(FidsON[0,:,0])):
        for jj in range(len(FidsON[0,0,:])):
            FidsOFF[pp,kk,jj] = FidsOFFtemp[pp,kk,jj][0] + 1j * FidsOFFtemp[pp,kk,jj][1]
            FidsON[pp,kk,jj] = FidsONtemp[pp,kk,jj][0] + 1j * FidsONtemp[pp,kk,jj][1]       

# Load Added offsets (small) and corresponding results from Gannet computed in Matlab
arrays = {}
f = h5py.File('InVivo_smallAddedOffsets.mat') # -5:5 Hz and -20:20 Deg
for k, v in f.items():
    arrays[k] = np.array(v)   

F0_Added = np.transpose(arrays['F0Addoffsets']) #Added frequency offsets, size [33, 320]
Ph_Added = np.transpose(arrays['PhAddoffsets']) #Added Phase offsets 

F0_AddedOFF = F0_Added[:, ::2] #[33, 160]
F0_AddedON = F0_Added[:, 1::2]
Ph_AddedOFF = Ph_Added[:, ::2] #[33, 160]
Ph_AddedON = Ph_Added[:, 1::2]

F0_SR = np.transpose(arrays['F0_SR']) #Added frequency offsets, size [33, 320]
Ph_SR = np.transpose(arrays['Ph_SR']) #Added Phase offsets, size [33, 320]     
        
#Apply artifical small offsets to in vivo data and prepare data for network
time = np.arange(2048)/2000
spectOffsetOFF = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]),len(FidsOFF[0,0,:])), dtype = np.complex_)
spectOffsetON = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
FidsOffsetOFF = np.empty(spectOffsetOFF.shape, dtype = np.complex_)
FidsOffsetON = np.empty(spectOffsetON.shape, dtype = np.complex_)
spectOffsetOFFscale = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]),len(FidsOFF[0,0,:])), dtype = np.complex_)
spectOffsetONscale = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
for kk in range(len(FidsON[0,:,0])):
    for mm in range(len(FidsON[0,0,:])):
        phasePartOFF = np.exp(-1j * Ph_AddedOFF[mm,kk] * cm.pi / 180)
        freqPartOFF = np.exp(-1j * time * F0_AddedOFF[mm,kk] * 2 * cm.pi)
        correctionOFF = np.multiply(freqPartOFF, phasePartOFF)
        
        phasePartON = np.exp(-1j * Ph_AddedON[mm,kk] * cm.pi / 180)
        freqPartON = np.exp(-1j * time * F0_AddedON[mm,kk] * 2 * cm.pi)
        correctionON = np.multiply(freqPartON,phasePartON)
        
        FidsOffsetOFF[:,kk,mm] = np.multiply(FidsOFF[:,kk,mm], correctionOFF)
        FidsOffsetON[:,kk,mm] = np.multiply(FidsON[:,kk,mm], correctionON)
        
        spectOffsetOFF[:,kk,mm] = np.squeeze(fftshift(fft(FidsOffsetOFF[:,kk,mm])))
        spectOffsetON[:,kk,mm] = np.squeeze(fftshift(fft(FidsOffsetON[:,kk,mm])))
        
        scaleOFF = np.max(np.squeeze(np.abs(spectOffsetOFF[:,kk,mm])))
        spectOffsetOFFscale[:,kk,mm] = spectOffsetOFF[:,kk,mm]/scaleOFF
        scaleON = np.max(np.squeeze(np.abs(spectOffsetON[:,kk,mm])))
        spectOffsetONscale[:,kk,mm] = spectOffsetON[:,kk,mm]/scaleON
  
#Prepare data for network        
testDataOffsetOFF = np.absolute(spectOffsetOFFscale[512:-512,:,:])
testDataOffsetON = np.absolute(spectOffsetONscale[512:-512,:,:])

# Load Trained Model - F0
model_F0 = tf.keras.models.load_model("Model_F0.model")

prediction_Offset_F0_OFF = np.empty((len(FidsOFF[0,0,:]), len(FidsOFF[0,:,0])))
prediction_Offset_F0_ON = np.empty((len(FidsON[0,0,:]), len(FidsON[0,:,0])))
for mm in range(len(FidsOFF[0,0,:])):
    testTempOFF = np.transpose(np.squeeze(testDataOffsetOFF[:,:,mm])) 
    testTempON = np.transpose(np.squeeze(testDataOffsetON[:,:,mm]))
    prediction_Offset_F0_OFF[mm,:] = np.squeeze(model_F0.predict(testTempOFF))
    prediction_Offset_F0_ON[mm,:] = np.squeeze(model_F0.predict(testTempON))

# Perform Frequency Correction
FidsUnAlignedOffsetOFFPhase = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
spectUnAlignedOffsetOFFPhase = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
FidsUnAlignedOffsetONPhase = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
spectUnAlignedOffsetONPhase = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
spectUnAlignedOffsetOFFPhasescale = np.empty(spectUnAlignedOffsetOFFPhase.shape, dtype = np.complex_)
spectUnAlignedOffsetONPhasescale = np.empty(spectUnAlignedOffsetONPhase.shape, dtype = np.complex_)
for mm in range(len(FidsOFF[0,0,:])):
    for kk in range(len(FidsOFF[0,:,0])):       
        freqPartOFF = np.squeeze(np.exp(1j * time * prediction_Offset_F0_OFF[mm,kk] * 2 * cm.pi))
        freqPartON = np.squeeze(np.exp(1j * time * prediction_Offset_F0_ON[mm,kk] * 2 * cm.pi))
        
        FidsUnAlignedOffsetOFFPhase[:,kk,mm] = np.multiply(FidsOffsetOFF[:,kk,mm], freqPartOFF) 
        FidsUnAlignedOffsetONPhase[:,kk,mm] = np.multiply(FidsOffsetON[:,kk,mm], freqPartON) 
        
        spectUnAlignedOffsetOFFPhase[:,kk,mm] = np.squeeze(fftshift(fft(FidsUnAlignedOffsetOFFPhase[:,kk,mm])))
        spectUnAlignedOffsetONPhase[:,kk,mm] = np.squeeze(fftshift(fft(FidsUnAlignedOffsetONPhase[:,kk,mm])))
        
        scaleOFF = np.max(np.squeeze(np.abs(spectUnAlignedOffsetOFFPhase[:,kk,mm])))
        scaleON = np.max(np.squeeze(np.abs(spectUnAlignedOffsetONPhase[:,kk,mm])))
        
        spectUnAlignedOffsetOFFPhasescale[:,kk,mm] = spectUnAlignedOffsetOFFPhase[:,kk,mm]/scaleOFF
        spectUnAlignedOffsetONPhasescale[:,kk,mm] = spectUnAlignedOffsetONPhase[:,kk,mm]/scaleON
        
testDataOffsetOFFPhase = np.real(spectUnAlignedOffsetOFFPhasescale[512:-512,:,:])
testDataOffsetONPhase = np.real(spectUnAlignedOffsetONPhasescale[512:-512,:,:]) 

# Load Trained Model - Ph
model_Ph = tf.keras.models.load_model("Model_Ph.model")
prediction_Offset_Ph_OFF = np.empty((len(FidsOFF[0,0,:]), len(FidsOFF[0,:,0])))
prediction_Offset_Ph_ON = np.empty((len(FidsON[0,0,:]), len(FidsON[0,:,0])))
for mm in range(len(FidsOFF[0,0,:])):
    testTempOFF = np.transpose(np.squeeze(testDataOffsetOFFPhase[:,:,mm]))
    testTempON = np.transpose(np.squeeze(testDataOffsetONPhase[:,:,mm]))
    prediction_Offset_Ph_OFF[mm,:] = np.squeeze(model_Ph.predict(testTempOFF)) 
    prediction_Offset_Ph_ON[mm,:] = np.squeeze(model_Ph.predict(testTempON)) 

# Perform Phase Correction
FidsOffsetOFFCorrected = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
spectOffsetOFFCorrected = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
FidsOffsetONCorrected = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
spectOffsetONCorrected = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
for mm in range(len(FidsOFF[0,0,:])):
    for kk in range(len(FidsOFF[0,:,0])):     
        phasePartOFF = np.squeeze(np.exp(1j * prediction_Offset_Ph_OFF[mm,kk] * cm.pi / 180))
        phasePartON = np.squeeze(np.exp(1j * prediction_Offset_Ph_ON[mm,kk] * cm.pi / 180))
        
        FidsOffsetOFFCorrected[:,kk,mm] = np.multiply(FidsUnAlignedOffsetOFFPhase[:,kk,mm], phasePartOFF) 
        FidsOffsetONCorrected[:,kk,mm] = np.multiply(FidsUnAlignedOffsetONPhase[:,kk,mm], phasePartON) 
        
        spectOffsetOFFCorrected[:,kk,mm] = np.squeeze(fftshift(fft(FidsOffsetOFFCorrected[:,kk,mm])))
        spectOffsetONCorrected[:,kk,mm] = np.squeeze(fftshift(fft(FidsOffsetONCorrected[:,kk,mm])))

#Apply SR offsets and compute the SR FIDs for comparison
        
#Resulting offsets from Gannet
F0_SROFF = F0_SR[:, ::2]
F0_SRON = F0_SR[:, 1::2]
Ph_SROFF = Ph_SR[:, ::2]
Ph_SRON = Ph_SR[:, 1::2] 

time = np.arange(2048)/2000
spectSRcorrOFF = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]),len(FidsOFF[0,0,:])), dtype = np.complex_)
spectSRcorrON = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
FIDsSROFF = np.empty(spectSRcorrOFF.shape, dtype = np.complex_)
FIDsSRON = np.empty(spectSRcorrON.shape, dtype = np.complex_)
for kk in range(len(FidsON[0,:,0])):
    for mm in range(len(FidsON[0,0,:])):
        phasePartOFF = np.multiply(np.exp(1j * Ph_SROFF[mm,kk] * cm.pi / 180), np.exp(-1j * Ph_AddedOFF[mm,kk] * cm.pi / 180))
        freqPartOFF = np.multiply(np.exp(1j * time * F0_SROFF[mm,kk] * 2 * cm.pi), np.exp(-1j * time * F0_AddedOFF[mm,kk] * 2 * cm.pi))
        correctionOFF = np.multiply(freqPartOFF, phasePartOFF)
        
        phasePartON = np.multiply(np.exp(1j * Ph_SRON[mm,kk] * cm.pi / 180), np.exp(-1j * Ph_AddedON[mm,kk] * cm.pi / 180))
        freqPartON = np.multiply(np.exp(1j * time * F0_SRON[mm,kk] * 2 * cm.pi), np.exp(-1j * time * F0_AddedON[mm,kk] * 2 * cm.pi))
        correctionON = np.multiply(freqPartON,phasePartON)
        
        FIDsSROFF[:,kk,mm] = np.multiply(FidsOFF[:,kk,mm], correctionOFF)
        FIDsSRON[:,kk,mm] = np.multiply(FidsON[:,kk,mm], correctionON)
        
# Perform second Global Frequency and Phase Correction only for SR data
arrays = {}
f = h5py.File('GlobalSR_smallAddedOffsets.mat')
for k, v in f.items():
    arrays[k] = np.array(v)   

freqCorrSR = arrays['F0SR']
phaseCorrSR = arrays['P0SR']

timeR = np.arange(2048)/2000

for mm in range(len(FidsOFF[0,0,:])):
    for kk in range(len(FidsOFF[0,:,0])):
                
        #No Correction for DL 
        spectOffsetOFFCorrected[:,kk,mm] = np.squeeze(fftshift(fft(FidsOffsetOFFCorrected[:,kk,mm])))
        spectOffsetONCorrected[:,kk,mm] = np.squeeze(fftshift(fft(FidsOffsetONCorrected[:,kk,mm])))

        phasePartOFF = np.exp(1j * phaseCorrSR[mm] * cm.pi / 180)
        freqPartOFF = np.exp(1j * timeR * freqCorrSR[mm] * 2 * cm.pi)
        correctionOFF = np.multiply(freqPartOFF, phasePartOFF)
        
        phasePartON = np.exp(1j * phaseCorrSR[mm] * cm.pi / 180)
        freqPartON = np.exp(1j * timeR * freqCorrSR[mm] * 2 * cm.pi)
        correctionON = np.multiply(freqPartON,phasePartON)
        
        spectSRcorrOFF[:,kk,mm] = np.squeeze(fftshift(fft(np.multiply(FIDsSROFF[:,kk,mm], correctionOFF))))
        spectSRcorrON[:,kk,mm] = np.squeeze(fftshift(fft(np.multiply(FIDsSRON[:,kk,mm], correctionON))))
                
diffNO_offset1 = np.squeeze(np.mean(spectOffsetON[:,:,:], axis=1).real) - np.squeeze(np.mean(spectOffsetOFF[:,:,:], axis=1).real)
diffDL_offset1 = np.squeeze(np.mean(spectOffsetONCorrected[:,:,:], axis=1).real) - np.squeeze(np.mean(spectOffsetOFFCorrected[:,:,:], axis=1).real)
diffSR_offset1 = np.squeeze(np.mean(spectSRcorrON[:,:,:], axis=1).real) - np.squeeze(np.mean(spectSRcorrOFF[:,:,:], axis=1).real)

# %% Compute results for medium added offsets

# Load in vivo data - 33 datasets with 160 ON and 160 OFF
arrays = {}
f = h5py.File('InVivoData.mat')
for k, v in f.items():
    arrays[k] = np.array(v)    
    
FidsOFFtemp = np.transpose(arrays['OFFdata']) #OFF data
FidsONtemp = np.transpose(arrays['ONdata']) #ON data

# Construct the OFF and ON Fids
FidsOFF = np.empty((FidsOFFtemp.shape), dtype = np.complex_)
FidsON = np.empty((FidsONtemp.shape), dtype = np.complex_)
for pp in range(len(FidsON[:,0,0])):
    for kk in range(len(FidsON[0,:,0])):
        for jj in range(len(FidsON[0,0,:])):
            FidsOFF[pp,kk,jj] = FidsOFFtemp[pp,kk,jj][0] + 1j * FidsOFFtemp[pp,kk,jj][1]
            FidsON[pp,kk,jj] = FidsONtemp[pp,kk,jj][0] + 1j * FidsONtemp[pp,kk,jj][1]       

# Load Added offsets (medium) and corresponding results from Gannet computed in Matlab
arrays = {}
f = h5py.File('InVivo_mediumAddedOffsets.mat') # -10:-5 Hz, 5:10 Hz and -45:-20, 20:45 Deg
for k, v in f.items():
    arrays[k] = np.array(v)   

F0_Added = np.transpose(arrays['F0Addoffsets']) #Added frequency offsets, size [33, 320]
Ph_Added = np.transpose(arrays['PhAddoffsets']) #Added Phase offsets 

F0_AddedOFF = F0_Added[:, ::2] #[33, 160]
F0_AddedON = F0_Added[:, 1::2]
Ph_AddedOFF = Ph_Added[:, ::2] #[33, 160]
Ph_AddedON = Ph_Added[:, 1::2]

F0_SR = np.transpose(arrays['F0_SR']) #Added frequency offsets, size [33, 320]
Ph_SR = np.transpose(arrays['Ph_SR']) #Added Phase offsets, size [33, 320]     
        
#Apply artifical medium offsets to in vivo data and prepare data for network
time = np.arange(2048)/2000
spectOffsetOFF = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]),len(FidsOFF[0,0,:])), dtype = np.complex_)
spectOffsetON = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
FidsOffsetOFF = np.empty(spectOffsetOFF.shape, dtype = np.complex_)
FidsOffsetON = np.empty(spectOffsetON.shape, dtype = np.complex_)
spectOffsetOFFscale = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]),len(FidsOFF[0,0,:])), dtype = np.complex_)
spectOffsetONscale = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
for kk in range(len(FidsON[0,:,0])):
    for mm in range(len(FidsON[0,0,:])):
        phasePartOFF = np.exp(-1j * Ph_AddedOFF[mm,kk] * cm.pi / 180)
        freqPartOFF = np.exp(-1j * time * F0_AddedOFF[mm,kk] * 2 * cm.pi)
        correctionOFF = np.multiply(freqPartOFF, phasePartOFF)
        
        phasePartON = np.exp(-1j * Ph_AddedON[mm,kk] * cm.pi / 180)
        freqPartON = np.exp(-1j * time * F0_AddedON[mm,kk] * 2 * cm.pi)
        correctionON = np.multiply(freqPartON,phasePartON)
        
        FidsOffsetOFF[:,kk,mm] = np.multiply(FidsOFF[:,kk,mm], correctionOFF)
        FidsOffsetON[:,kk,mm] = np.multiply(FidsON[:,kk,mm], correctionON)
        
        spectOffsetOFF[:,kk,mm] = np.squeeze(fftshift(fft(FidsOffsetOFF[:,kk,mm])))
        spectOffsetON[:,kk,mm] = np.squeeze(fftshift(fft(FidsOffsetON[:,kk,mm])))
        
        scaleOFF = np.max(np.squeeze(np.abs(spectOffsetOFF[:,kk,mm])))
        spectOffsetOFFscale[:,kk,mm] = spectOffsetOFF[:,kk,mm]/scaleOFF
        scaleON = np.max(np.squeeze(np.abs(spectOffsetON[:,kk,mm])))
        spectOffsetONscale[:,kk,mm] = spectOffsetON[:,kk,mm]/scaleON
  
#Prepare data for network        
testDataOffsetOFF = np.absolute(spectOffsetOFFscale[512:-512,:,:])
testDataOffsetON = np.absolute(spectOffsetONscale[512:-512,:,:])

# Load Trained Model - F0
model_F0 = tf.keras.models.load_model("Model_F0.model")

prediction_Offset_F0_OFF = np.empty((len(FidsOFF[0,0,:]), len(FidsOFF[0,:,0])))
prediction_Offset_F0_ON = np.empty((len(FidsON[0,0,:]), len(FidsON[0,:,0])))
for mm in range(len(FidsOFF[0,0,:])):
    testTempOFF = np.transpose(np.squeeze(testDataOffsetOFF[:,:,mm])) 
    testTempON = np.transpose(np.squeeze(testDataOffsetON[:,:,mm]))
    prediction_Offset_F0_OFF[mm,:] = np.squeeze(model_F0.predict(testTempOFF))
    prediction_Offset_F0_ON[mm,:] = np.squeeze(model_F0.predict(testTempON))

# Perform Frequency Correction
FidsUnAlignedOffsetOFFPhase = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
spectUnAlignedOffsetOFFPhase = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
FidsUnAlignedOffsetONPhase = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
spectUnAlignedOffsetONPhase = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
spectUnAlignedOffsetOFFPhasescale = np.empty(spectUnAlignedOffsetOFFPhase.shape, dtype = np.complex_)
spectUnAlignedOffsetONPhasescale = np.empty(spectUnAlignedOffsetONPhase.shape, dtype = np.complex_)
for mm in range(len(FidsOFF[0,0,:])):
    for kk in range(len(FidsOFF[0,:,0])):       
        freqPartOFF = np.squeeze(np.exp(1j * time * prediction_Offset_F0_OFF[mm,kk] * 2 * cm.pi))
        freqPartON = np.squeeze(np.exp(1j * time * prediction_Offset_F0_ON[mm,kk] * 2 * cm.pi))
        
        FidsUnAlignedOffsetOFFPhase[:,kk,mm] = np.multiply(FidsOffsetOFF[:,kk,mm], freqPartOFF) 
        FidsUnAlignedOffsetONPhase[:,kk,mm] = np.multiply(FidsOffsetON[:,kk,mm], freqPartON) 
        
        spectUnAlignedOffsetOFFPhase[:,kk,mm] = np.squeeze(fftshift(fft(FidsUnAlignedOffsetOFFPhase[:,kk,mm])))
        spectUnAlignedOffsetONPhase[:,kk,mm] = np.squeeze(fftshift(fft(FidsUnAlignedOffsetONPhase[:,kk,mm])))
        
        scaleOFF = np.max(np.squeeze(np.abs(spectUnAlignedOffsetOFFPhase[:,kk,mm])))
        scaleON = np.max(np.squeeze(np.abs(spectUnAlignedOffsetONPhase[:,kk,mm])))
        
        spectUnAlignedOffsetOFFPhasescale[:,kk,mm] = spectUnAlignedOffsetOFFPhase[:,kk,mm]/scaleOFF
        spectUnAlignedOffsetONPhasescale[:,kk,mm] = spectUnAlignedOffsetONPhase[:,kk,mm]/scaleON
        
testDataOffsetOFFPhase = np.real(spectUnAlignedOffsetOFFPhasescale[512:-512,:,:])
testDataOffsetONPhase = np.real(spectUnAlignedOffsetONPhasescale[512:-512,:,:]) 

# Load Trained Model - Ph
model_Ph = tf.keras.models.load_model("Model_Ph.model")
prediction_Offset_Ph_OFF = np.empty((len(FidsOFF[0,0,:]), len(FidsOFF[0,:,0])))
prediction_Offset_Ph_ON = np.empty((len(FidsON[0,0,:]), len(FidsON[0,:,0])))
for mm in range(len(FidsOFF[0,0,:])):
    testTempOFF = np.transpose(np.squeeze(testDataOffsetOFFPhase[:,:,mm]))
    testTempON = np.transpose(np.squeeze(testDataOffsetONPhase[:,:,mm]))
    prediction_Offset_Ph_OFF[mm,:] = np.squeeze(model_Ph.predict(testTempOFF)) 
    prediction_Offset_Ph_ON[mm,:] = np.squeeze(model_Ph.predict(testTempON)) 

# Perform Phase Correction
FidsOffsetOFFCorrected = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
spectOffsetOFFCorrected = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
FidsOffsetONCorrected = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
spectOffsetONCorrected = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
for mm in range(len(FidsOFF[0,0,:])):
    for kk in range(len(FidsOFF[0,:,0])):     
        phasePartOFF = np.squeeze(np.exp(1j * prediction_Offset_Ph_OFF[mm,kk] * cm.pi / 180))
        phasePartON = np.squeeze(np.exp(1j * prediction_Offset_Ph_ON[mm,kk] * cm.pi / 180))
        
        FidsOffsetOFFCorrected[:,kk,mm] = np.multiply(FidsUnAlignedOffsetOFFPhase[:,kk,mm], phasePartOFF) 
        FidsOffsetONCorrected[:,kk,mm] = np.multiply(FidsUnAlignedOffsetONPhase[:,kk,mm], phasePartON) 
        
        spectOffsetOFFCorrected[:,kk,mm] = np.squeeze(fftshift(fft(FidsOffsetOFFCorrected[:,kk,mm])))
        spectOffsetONCorrected[:,kk,mm] = np.squeeze(fftshift(fft(FidsOffsetONCorrected[:,kk,mm])))

#Apply SR offsets and compute the SR FIDs for comparison
        
#Resulting offsets from Gannet
F0_SROFF = F0_SR[:, ::2]
F0_SRON = F0_SR[:, 1::2]
Ph_SROFF = Ph_SR[:, ::2]
Ph_SRON = Ph_SR[:, 1::2] 

time = np.arange(2048)/2000
spectSRcorrOFF = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]),len(FidsOFF[0,0,:])), dtype = np.complex_)
spectSRcorrON = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
FIDsSROFF = np.empty(spectSRcorrOFF.shape, dtype = np.complex_)
FIDsSRON = np.empty(spectSRcorrON.shape, dtype = np.complex_)
for kk in range(len(FidsON[0,:,0])):
    for mm in range(len(FidsON[0,0,:])):
        phasePartOFF = np.multiply(np.exp(1j * Ph_SROFF[mm,kk] * cm.pi / 180), np.exp(-1j * Ph_AddedOFF[mm,kk] * cm.pi / 180))
        freqPartOFF = np.multiply(np.exp(1j * time * F0_SROFF[mm,kk] * 2 * cm.pi), np.exp(-1j * time * F0_AddedOFF[mm,kk] * 2 * cm.pi))
        correctionOFF = np.multiply(freqPartOFF, phasePartOFF)
        
        phasePartON = np.multiply(np.exp(1j * Ph_SRON[mm,kk] * cm.pi / 180), np.exp(-1j * Ph_AddedON[mm,kk] * cm.pi / 180))
        freqPartON = np.multiply(np.exp(1j * time * F0_SRON[mm,kk] * 2 * cm.pi), np.exp(-1j * time * F0_AddedON[mm,kk] * 2 * cm.pi))
        correctionON = np.multiply(freqPartON,phasePartON)
        
        FIDsSROFF[:,kk,mm] = np.multiply(FidsOFF[:,kk,mm], correctionOFF)
        FIDsSRON[:,kk,mm] = np.multiply(FidsON[:,kk,mm], correctionON)
        
# Perform second Global Frequency and Phase Correction only for SR data
arrays = {}
f = h5py.File('GlobalSR_mediumAddedOffsets.mat')
for k, v in f.items():
    arrays[k] = np.array(v)   

freqCorrSR = arrays['F0SR']
phaseCorrSR = arrays['P0SR']

timeR = np.arange(2048)/2000

for mm in range(len(FidsOFF[0,0,:])):
    for kk in range(len(FidsOFF[0,:,0])):
                
        #No Correction for DL 
        spectOffsetOFFCorrected[:,kk,mm] = np.squeeze(fftshift(fft(FidsOffsetOFFCorrected[:,kk,mm])))
        spectOffsetONCorrected[:,kk,mm] = np.squeeze(fftshift(fft(FidsOffsetONCorrected[:,kk,mm])))

        phasePartOFF = np.exp(1j * phaseCorrSR[mm] * cm.pi / 180)
        freqPartOFF = np.exp(1j * timeR * freqCorrSR[mm] * 2 * cm.pi)
        correctionOFF = np.multiply(freqPartOFF, phasePartOFF)
        
        phasePartON = np.exp(1j * phaseCorrSR[mm] * cm.pi / 180)
        freqPartON = np.exp(1j * timeR * freqCorrSR[mm] * 2 * cm.pi)
        correctionON = np.multiply(freqPartON,phasePartON)
        
        spectSRcorrOFF[:,kk,mm] = np.squeeze(fftshift(fft(np.multiply(FIDsSROFF[:,kk,mm], correctionOFF))))
        spectSRcorrON[:,kk,mm] = np.squeeze(fftshift(fft(np.multiply(FIDsSRON[:,kk,mm], correctionON))))
                
diffNO_offset2 = np.squeeze(np.mean(spectOffsetON[:,:,:], axis=1).real) - np.squeeze(np.mean(spectOffsetOFF[:,:,:], axis=1).real)
diffDL_offset2 = np.squeeze(np.mean(spectOffsetONCorrected[:,:,:], axis=1).real) - np.squeeze(np.mean(spectOffsetOFFCorrected[:,:,:], axis=1).real)
diffSR_offset2 = np.squeeze(np.mean(spectSRcorrON[:,:,:], axis=1).real) - np.squeeze(np.mean(spectSRcorrOFF[:,:,:], axis=1).real)

# %% Compute results for large added offsets

# Load in vivo data - 33 datasets with 160 ON and 160 OFF
arrays = {}
f = h5py.File('InVivoData.mat')
for k, v in f.items():
    arrays[k] = np.array(v)    
    
FidsOFFtemp = np.transpose(arrays['OFFdata']) #OFF data
FidsONtemp = np.transpose(arrays['ONdata']) #ON data

# Construct the OFF and ON Fids
FidsOFF = np.empty((FidsOFFtemp.shape), dtype = np.complex_)
FidsON = np.empty((FidsONtemp.shape), dtype = np.complex_)
for pp in range(len(FidsON[:,0,0])):
    for kk in range(len(FidsON[0,:,0])):
        for jj in range(len(FidsON[0,0,:])):
            FidsOFF[pp,kk,jj] = FidsOFFtemp[pp,kk,jj][0] + 1j * FidsOFFtemp[pp,kk,jj][1]
            FidsON[pp,kk,jj] = FidsONtemp[pp,kk,jj][0] + 1j * FidsONtemp[pp,kk,jj][1]       

# Load Added offsets (large) and corresponding results from Gannet computed in Matlab
arrays = {}
f = h5py.File('InVivo_largeAddedOffsets.mat') # -10:-5 Hz, 5:10 Hz and -45:-20, 20:45 Deg
for k, v in f.items():
    arrays[k] = np.array(v)   

F0_Added = np.transpose(arrays['F0Addoffsets']) #Added frequency offsets, size [33, 320]
Ph_Added = np.transpose(arrays['PhAddoffsets']) #Added Phase offsets 

F0_AddedOFF = F0_Added[:, ::2] #[33, 160]
F0_AddedON = F0_Added[:, 1::2]
Ph_AddedOFF = Ph_Added[:, ::2] #[33, 160]
Ph_AddedON = Ph_Added[:, 1::2]

F0_SR = np.transpose(arrays['F0_SR']) #Added frequency offsets, size [33, 320]
Ph_SR = np.transpose(arrays['Ph_SR']) #Added Phase offsets, size [33, 320]     
        
#Apply artifical large offsets to in vivo data and prepare data for network
time = np.arange(2048)/2000
spectOffsetOFF = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]),len(FidsOFF[0,0,:])), dtype = np.complex_)
spectOffsetON = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
FidsOffsetOFF = np.empty(spectOffsetOFF.shape, dtype = np.complex_)
FidsOffsetON = np.empty(spectOffsetON.shape, dtype = np.complex_)
spectOffsetOFFscale = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]),len(FidsOFF[0,0,:])), dtype = np.complex_)
spectOffsetONscale = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
for kk in range(len(FidsON[0,:,0])):
    for mm in range(len(FidsON[0,0,:])):
        phasePartOFF = np.exp(-1j * Ph_AddedOFF[mm,kk] * cm.pi / 180)
        freqPartOFF = np.exp(-1j * time * F0_AddedOFF[mm,kk] * 2 * cm.pi)
        correctionOFF = np.multiply(freqPartOFF, phasePartOFF)
        
        phasePartON = np.exp(-1j * Ph_AddedON[mm,kk] * cm.pi / 180)
        freqPartON = np.exp(-1j * time * F0_AddedON[mm,kk] * 2 * cm.pi)
        correctionON = np.multiply(freqPartON,phasePartON)
        
        FidsOffsetOFF[:,kk,mm] = np.multiply(FidsOFF[:,kk,mm], correctionOFF)
        FidsOffsetON[:,kk,mm] = np.multiply(FidsON[:,kk,mm], correctionON)
        
        spectOffsetOFF[:,kk,mm] = np.squeeze(fftshift(fft(FidsOffsetOFF[:,kk,mm])))
        spectOffsetON[:,kk,mm] = np.squeeze(fftshift(fft(FidsOffsetON[:,kk,mm])))
        
        scaleOFF = np.max(np.squeeze(np.abs(spectOffsetOFF[:,kk,mm])))
        spectOffsetOFFscale[:,kk,mm] = spectOffsetOFF[:,kk,mm]/scaleOFF
        scaleON = np.max(np.squeeze(np.abs(spectOffsetON[:,kk,mm])))
        spectOffsetONscale[:,kk,mm] = spectOffsetON[:,kk,mm]/scaleON
  
#Prepare data for network        
testDataOffsetOFF = np.absolute(spectOffsetOFFscale[512:-512,:,:])
testDataOffsetON = np.absolute(spectOffsetONscale[512:-512,:,:])

# Load Trained Model - F0
model_F0 = tf.keras.models.load_model("Model_F0.model")

prediction_Offset_F0_OFF = np.empty((len(FidsOFF[0,0,:]), len(FidsOFF[0,:,0])))
prediction_Offset_F0_ON = np.empty((len(FidsON[0,0,:]), len(FidsON[0,:,0])))
for mm in range(len(FidsOFF[0,0,:])):
    testTempOFF = np.transpose(np.squeeze(testDataOffsetOFF[:,:,mm])) 
    testTempON = np.transpose(np.squeeze(testDataOffsetON[:,:,mm]))
    prediction_Offset_F0_OFF[mm,:] = np.squeeze(model_F0.predict(testTempOFF))
    prediction_Offset_F0_ON[mm,:] = np.squeeze(model_F0.predict(testTempON))

# Perform Frequency Correction
FidsUnAlignedOffsetOFFPhase = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
spectUnAlignedOffsetOFFPhase = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
FidsUnAlignedOffsetONPhase = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
spectUnAlignedOffsetONPhase = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
spectUnAlignedOffsetOFFPhasescale = np.empty(spectUnAlignedOffsetOFFPhase.shape, dtype = np.complex_)
spectUnAlignedOffsetONPhasescale = np.empty(spectUnAlignedOffsetONPhase.shape, dtype = np.complex_)
for mm in range(len(FidsOFF[0,0,:])):
    for kk in range(len(FidsOFF[0,:,0])):       
        freqPartOFF = np.squeeze(np.exp(1j * time * prediction_Offset_F0_OFF[mm,kk] * 2 * cm.pi))
        freqPartON = np.squeeze(np.exp(1j * time * prediction_Offset_F0_ON[mm,kk] * 2 * cm.pi))
        
        FidsUnAlignedOffsetOFFPhase[:,kk,mm] = np.multiply(FidsOffsetOFF[:,kk,mm], freqPartOFF) 
        FidsUnAlignedOffsetONPhase[:,kk,mm] = np.multiply(FidsOffsetON[:,kk,mm], freqPartON) 
        
        spectUnAlignedOffsetOFFPhase[:,kk,mm] = np.squeeze(fftshift(fft(FidsUnAlignedOffsetOFFPhase[:,kk,mm])))
        spectUnAlignedOffsetONPhase[:,kk,mm] = np.squeeze(fftshift(fft(FidsUnAlignedOffsetONPhase[:,kk,mm])))
        
        scaleOFF = np.max(np.squeeze(np.abs(spectUnAlignedOffsetOFFPhase[:,kk,mm])))
        scaleON = np.max(np.squeeze(np.abs(spectUnAlignedOffsetONPhase[:,kk,mm])))
        
        spectUnAlignedOffsetOFFPhasescale[:,kk,mm] = spectUnAlignedOffsetOFFPhase[:,kk,mm]/scaleOFF
        spectUnAlignedOffsetONPhasescale[:,kk,mm] = spectUnAlignedOffsetONPhase[:,kk,mm]/scaleON
        
testDataOffsetOFFPhase = np.real(spectUnAlignedOffsetOFFPhasescale[512:-512,:,:])
testDataOffsetONPhase = np.real(spectUnAlignedOffsetONPhasescale[512:-512,:,:]) 

# Load Trained Model - Ph
model_Ph = tf.keras.models.load_model("Model_Ph.model")
prediction_Offset_Ph_OFF = np.empty((len(FidsOFF[0,0,:]), len(FidsOFF[0,:,0])))
prediction_Offset_Ph_ON = np.empty((len(FidsON[0,0,:]), len(FidsON[0,:,0])))
for mm in range(len(FidsOFF[0,0,:])):
    testTempOFF = np.transpose(np.squeeze(testDataOffsetOFFPhase[:,:,mm]))
    testTempON = np.transpose(np.squeeze(testDataOffsetONPhase[:,:,mm]))
    prediction_Offset_Ph_OFF[mm,:] = np.squeeze(model_Ph.predict(testTempOFF)) 
    prediction_Offset_Ph_ON[mm,:] = np.squeeze(model_Ph.predict(testTempON)) 

# Perform Phase Correction
FidsOffsetOFFCorrected = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
spectOffsetOFFCorrected = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
FidsOffsetONCorrected = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
spectOffsetONCorrected = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
for mm in range(len(FidsOFF[0,0,:])):
    for kk in range(len(FidsOFF[0,:,0])):     
        phasePartOFF = np.squeeze(np.exp(1j * prediction_Offset_Ph_OFF[mm,kk] * cm.pi / 180))
        phasePartON = np.squeeze(np.exp(1j * prediction_Offset_Ph_ON[mm,kk] * cm.pi / 180))
        
        FidsOffsetOFFCorrected[:,kk,mm] = np.multiply(FidsUnAlignedOffsetOFFPhase[:,kk,mm], phasePartOFF) 
        FidsOffsetONCorrected[:,kk,mm] = np.multiply(FidsUnAlignedOffsetONPhase[:,kk,mm], phasePartON) 
        
        spectOffsetOFFCorrected[:,kk,mm] = np.squeeze(fftshift(fft(FidsOffsetOFFCorrected[:,kk,mm])))
        spectOffsetONCorrected[:,kk,mm] = np.squeeze(fftshift(fft(FidsOffsetONCorrected[:,kk,mm])))

#Apply SR offsets and compute the SR FIDs for comparison
        
#Resulting offsets from Gannet
F0_SROFF = F0_SR[:, ::2]
F0_SRON = F0_SR[:, 1::2]
Ph_SROFF = Ph_SR[:, ::2]
Ph_SRON = Ph_SR[:, 1::2] 

time = np.arange(2048)/2000
spectSRcorrOFF = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]),len(FidsOFF[0,0,:])), dtype = np.complex_)
spectSRcorrON = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
FIDsSROFF = np.empty(spectSRcorrOFF.shape, dtype = np.complex_)
FIDsSRON = np.empty(spectSRcorrON.shape, dtype = np.complex_)
for kk in range(len(FidsON[0,:,0])):
    for mm in range(len(FidsON[0,0,:])):
        phasePartOFF = np.multiply(np.exp(1j * Ph_SROFF[mm,kk] * cm.pi / 180), np.exp(-1j * Ph_AddedOFF[mm,kk] * cm.pi / 180))
        freqPartOFF = np.multiply(np.exp(1j * time * F0_SROFF[mm,kk] * 2 * cm.pi), np.exp(-1j * time * F0_AddedOFF[mm,kk] * 2 * cm.pi))
        correctionOFF = np.multiply(freqPartOFF, phasePartOFF)
        
        phasePartON = np.multiply(np.exp(1j * Ph_SRON[mm,kk] * cm.pi / 180), np.exp(-1j * Ph_AddedON[mm,kk] * cm.pi / 180))
        freqPartON = np.multiply(np.exp(1j * time * F0_SRON[mm,kk] * 2 * cm.pi), np.exp(-1j * time * F0_AddedON[mm,kk] * 2 * cm.pi))
        correctionON = np.multiply(freqPartON,phasePartON)
        
        FIDsSROFF[:,kk,mm] = np.multiply(FidsOFF[:,kk,mm], correctionOFF)
        FIDsSRON[:,kk,mm] = np.multiply(FidsON[:,kk,mm], correctionON)
        
# Perform second Global Frequency and Phase Correction only for SR data
arrays = {}
f = h5py.File('GlobalSR_largeAddedOffsets.mat')
for k, v in f.items():
    arrays[k] = np.array(v)   

freqCorrSR = arrays['F0SR']
phaseCorrSR = arrays['P0SR']

timeR = np.arange(2048)/2000

for mm in range(len(FidsOFF[0,0,:])):
    for kk in range(len(FidsOFF[0,:,0])):
                
        #No Correction for DL 
        spectOffsetOFFCorrected[:,kk,mm] = np.squeeze(fftshift(fft(FidsOffsetOFFCorrected[:,kk,mm])))
        spectOffsetONCorrected[:,kk,mm] = np.squeeze(fftshift(fft(FidsOffsetONCorrected[:,kk,mm])))

        phasePartOFF = np.exp(1j * phaseCorrSR[mm] * cm.pi / 180)
        freqPartOFF = np.exp(1j * timeR * freqCorrSR[mm] * 2 * cm.pi)
        correctionOFF = np.multiply(freqPartOFF, phasePartOFF)
        
        phasePartON = np.exp(1j * phaseCorrSR[mm] * cm.pi / 180)
        freqPartON = np.exp(1j * timeR * freqCorrSR[mm] * 2 * cm.pi)
        correctionON = np.multiply(freqPartON,phasePartON)
        
        spectSRcorrOFF[:,kk,mm] = np.squeeze(fftshift(fft(np.multiply(FIDsSROFF[:,kk,mm], correctionOFF))))
        spectSRcorrON[:,kk,mm] = np.squeeze(fftshift(fft(np.multiply(FIDsSRON[:,kk,mm], correctionON))))
                
diffNO_offset3 = np.squeeze(np.mean(spectOffsetON[:,:,:], axis=1).real) - np.squeeze(np.mean(spectOffsetOFF[:,:,:], axis=1).real)
diffDL_offset3 = np.squeeze(np.mean(spectOffsetONCorrected[:,:,:], axis=1).real) - np.squeeze(np.mean(spectOffsetOFFCorrected[:,:,:], axis=1).real)
diffSR_offset3 = np.squeeze(np.mean(spectSRcorrON[:,:,:], axis=1).real) - np.squeeze(np.mean(spectSRcorrOFF[:,:,:], axis=1).real)

# %% Compute results without added offsets

# Load in vivo data - 33 datasets with 160 ON and 160 OFF
arrays = {}
f = h5py.File('InVivoData.mat')
for k, v in f.items():
    arrays[k] = np.array(v)    
    
FidsOFFtemp = np.transpose(arrays['OFFdata']) 
FidsONtemp = np.transpose(arrays['ONdata']) 

# Construct complex FIDs
FidsOFF = np.empty((FidsOFFtemp.shape), dtype = np.complex_)
FidsON = np.empty((FidsONtemp.shape), dtype = np.complex_)
for pp in range(len(FidsON[:,0,0])):
    for kk in range(len(FidsON[0,:,0])):
        for jj in range(len(FidsON[0,0,:])):
            FidsOFF[pp,kk,jj] = FidsOFFtemp[pp,kk,jj][0] + 1j * FidsOFFtemp[pp,kk,jj][1]
            FidsON[pp,kk,jj] = FidsONtemp[pp,kk,jj][0] + 1j * FidsONtemp[pp,kk,jj][1]       

#Prepare data for networks
spectUnAlignedOFF = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]),len(FidsOFF[0,0,:])), dtype = np.complex_)
spectUnAlignedON = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
spectUnAlignedOFFscale = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]),len(FidsOFF[0,0,:])), dtype = np.complex_)
spectUnAlignedONscale = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
for kk in range(len(FidsON[0,:,0])):
    for jj in range(len(FidsON[0,0,:])):
        spectUnAlignedOFF[:,kk,jj] = np.squeeze(fftshift(fft(FidsOFF[:,kk,jj])))
        scaleOFF = np.max(np.squeeze(np.abs(spectUnAlignedOFF[:,kk,jj])))
        spectUnAlignedOFFscale[:,kk,jj] = np.squeeze(fftshift(fft(FidsOFF[:,kk,jj])))/scaleOFF
        spectUnAlignedON[:,kk,jj] = np.squeeze(fftshift(fft(FidsON[:,kk,jj])))
        scaleON = np.max(np.squeeze(np.abs(spectUnAlignedON[:,kk,jj])))
        spectUnAlignedONscale[:,kk,jj] = np.squeeze(fftshift(fft(FidsON[:,kk,jj])))/scaleON
        
testDataOFF = np.absolute(spectUnAlignedOFFscale[512:-512,:,:])
testDataON = np.absolute(spectUnAlignedONscale[512:-512,:,:])

# Load Trained Model - F0
model_F0 = tf.keras.models.load_model("Model_F0.model")
prediction_F0_OFF = np.empty((len(FidsOFF[0,0,:]), len(FidsOFF[0,:,0])))
prediction_F0_ON = np.empty((len(FidsON[0,0,:]), len(FidsON[0,:,0])))
for mm in range(len(FidsOFF[0,0,:])):
    testTempOFF = np.transpose(np.squeeze(testDataOFF[:,:,mm]))
    testTempON = np.transpose(np.squeeze(testDataON[:,:,mm]))

    prediction_F0_OFF[mm,:] = np.squeeze(model_F0.predict(testTempOFF))
    prediction_F0_ON[mm,:] = np.squeeze(model_F0.predict(testTempON))
    

# Perform Frequency Correction
FidsUnAlignedOFFPhase = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
spectUnAlignedOFFPhase = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
FidsUnAlignedONPhase = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
spectUnAlignedONPhase = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
spectUnAlignedOFFPhasescale = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
spectUnAlignedONPhasescale = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)

for mm in range(len(FidsOFF[0,0,:])):
    for kk in range(len(FidsOFF[0,:,0])):
        
        freqPartOFF = np.squeeze(np.exp(1j * timeR * prediction_F0_OFF[mm,kk] * 2 * cm.pi))
        FidsUnAlignedOFFPhase[:,kk,mm] = np.multiply(FidsOFF[:,kk,mm], freqPartOFF) 
        spectUnAlignedOFFPhase[:,kk,mm] = np.squeeze(fftshift(fft(FidsUnAlignedOFFPhase[:,kk,mm])))
        
        freqPartON = np.squeeze(np.exp(1j * timeR * prediction_F0_ON[mm,kk] * 2 * cm.pi))
        FidsUnAlignedONPhase[:,kk,mm] = np.multiply(FidsON[:,kk,mm], freqPartON) 
        spectUnAlignedONPhase[:,kk,mm] = np.squeeze(fftshift(fft(FidsUnAlignedONPhase[:,kk,mm])))
        
        scaleOFF = np.max(np.squeeze(np.abs(spectUnAlignedOFFPhase[:,kk,mm])))
        scaleON = np.max(np.squeeze(np.abs(spectUnAlignedONPhase[:,kk,mm])))
        
        spectUnAlignedOFFPhasescale[:,kk,mm] = spectUnAlignedOFFPhase[:,kk,mm]/scaleOFF
        spectUnAlignedONPhasescale[:,kk,mm] = spectUnAlignedONPhase[:,kk,mm]/scaleON


testDataOFFPhase = np.real(spectUnAlignedOFFPhasescale[512:-512,:,:])
testDataONPhase = np.real(spectUnAlignedONPhasescale[512:-512,:,:]) 

# Load Trained Model - Ph
model_Ph = tf.keras.models.load_model("Model_Ph.model")
prediction_Ph_OFF = np.empty((len(FidsOFF[0,0,:]), len(FidsOFF[0,:,0])))
prediction_Ph_ON = np.empty((len(FidsON[0,0,:]), len(FidsON[0,:,0])))
for mm in range(len(FidsOFF[0,0,:])):
    testTempOFF = np.transpose(np.squeeze(testDataOFFPhase[:,:,mm]))
    testTempON = np.transpose(np.squeeze(testDataONPhase[:,:,mm]))

    prediction_Ph_OFF[mm,:] = np.squeeze(model_Ph.predict(testTempOFF))
    prediction_Ph_ON[mm,:] = np.squeeze(model_Ph.predict(testTempON))
    
# Perform Phase Correction
FidsOFFCorrected = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
spectOFFCorrected = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
FidsONCorrected = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
spectONCorrected = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]), len(FidsOFF[0,0,:])), dtype = np.complex_)
for mm in range(len(FidsOFF[0,0,:])):
    for kk in range(len(FidsOFF[0,:,0])):
        
        phasePartOFF = np.squeeze(np.exp(1j * prediction_Ph_OFF[mm,kk] * cm.pi / 180))
        FidsOFFCorrected[:,kk,mm] = np.multiply(FidsUnAlignedOFFPhase[:,kk,mm], phasePartOFF) 
        
        phasePartON = np.squeeze(np.exp(1j * prediction_Ph_ON[mm,kk] * cm.pi / 180))
        FidsONCorrected[:,kk,mm] = np.multiply(FidsUnAlignedONPhase[:,kk,mm], phasePartON) 


# Load results from Gannet using no added offsets - Compute results from SR       
arrays = {}
f = h5py.File('inVivo_noAddedOffsets.mat')
for k, v in f.items():
    arrays[k] = np.array(v)    
    
F0_OFF = np.transpose(arrays['F0_OFF']) 
Ph_OFF = np.transpose(arrays['Ph_OFF']) 
F0_ON = np.transpose(arrays['F0_ON']) 
Ph_ON = np.transpose(arrays['Ph_ON']) 
       
spectrSROFF = np.empty((len(FidsOFF[:,0,0]), len(FidsOFF[0,:,0]),len(FidsOFF[0,0,:])), dtype = np.complex_)
spectrSRON = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
FIDsSROFF = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
FIDsSRON = np.empty((len(FidsON[:,0,0]), len(FidsON[0,:,0]), len(FidsON[0,0,:])), dtype = np.complex_)
for kk in range(len(FidsON[0,:,0])):
    for mm in range(len(FidsON[0,0,:])):
        phasePartOFF = np.exp(1j * Ph_OFF[mm,kk] * cm.pi / 180)
        freqPartOFF = np.exp(1j * timeR * F0_OFF[mm,kk] * 2 * cm.pi)
        correctionOFF = np.multiply(freqPartOFF, phasePartOFF)
        
        phasePartON = np.exp(1j * Ph_ON[mm,kk] * cm.pi / 180)
        freqPartON = np.exp(1j * timeR * F0_ON[mm,kk] * 2 * cm.pi)
        correctionON = np.multiply(freqPartON,phasePartON)
        
        FIDsSROFF[:,kk,mm] = np.multiply(FidsOFF[:,kk,mm], correctionOFF)
        FIDsSRON[:,kk,mm] = np.multiply(FidsON[:,kk,mm], correctionON)
        
# Perform second Global Frequency and Phase Correction

arrays = {}
f = h5py.File('GlobalSR_noAddedOffsets.mat')
for k, v in f.items():
    arrays[k] = np.array(v)   

freqCorrSR = arrays['F0SR']
phaseCorrSR = arrays['P0SR']

for mm in range(len(FidsOFF[0,0,:])):
    for kk in range(len(FidsOFF[0,:,0])):
        
        #No Global for DL
        spectOFFCorrected[:,kk,mm] = np.squeeze(fftshift(fft(FidsOFFCorrected[:,kk,mm])))
        spectONCorrected[:,kk,mm] = np.squeeze(fftshift(fft(FidsONCorrected[:,kk,mm])))

        phasePartOFF = np.exp(1j * phaseCorrSR[mm] * cm.pi / 180)
        freqPartOFF = np.exp(1j * timeR * freqCorrSR[mm] * 2 * cm.pi)
        correctionOFF = np.multiply(freqPartOFF, phasePartOFF)
        
        phasePartON = np.exp(1j * phaseCorrSR[mm] * cm.pi / 180)
        freqPartON = np.exp(1j * timeR * freqCorrSR[mm] * 2 * cm.pi)
        correctionON = np.multiply(freqPartON,phasePartON)
        
        spectrSROFF[:,kk,mm] = np.squeeze(fftshift(fft(np.multiply(FIDsSROFF[:,kk,mm], correctionOFF))))
        spectrSRON[:,kk,mm] = np.squeeze(fftshift(fft(np.multiply(FIDsSRON[:,kk,mm], correctionON))))
        

diffNO = np.squeeze(np.mean(spectUnAlignedON[:,:,:], axis=1).real) - np.squeeze(np.mean(spectUnAlignedOFF[:,:,:], axis=1).real)
diffDL = np.squeeze(np.mean(spectONCorrected[:,:,:], axis=1).real) - np.squeeze(np.mean(spectOFFCorrected[:,:,:], axis=1).real)
diffSR = np.squeeze(np.mean(spectrSRON[:,:,:], axis=1).real) - np.squeeze(np.mean(spectrSROFF[:,:,:], axis=1).real)

# %% Plot Figure and compute Q scores
 
freqRange = 2000/127.7
ppm = np.flip((np.arange(1,len(FidsOFF[:,0,0]) + 1)) / len(FidsOFF[:,0,0]) * freqRange + 4.68 - freqRange/2)
xlim = [1.8, 4.2]

fig1 = plt.figure(figsize=(17.6/2.54, 12/2.54))
plt.rcParams['mathtext.default'] = 'regular'
plt.rc('axes', linewidth=1)
ax = plt.subplot(241)    
ax.spines["top"].set_visible(False)       
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)   
ax.tick_params(width=1,color='k',length=2)
plt.xticks(np.arange(2,4.3,step=1), fontsize=9)
ax.set_yticklabels([])
ax.axes.get_yaxis().set_visible(False)  
plt.plot(ppm, diffNO+0.02,'k',linewidth=0.1)
plt.plot(ppm, diffDL,'k',linewidth=0.1)
plt.plot(ppm, diffSR-0.02,'k',linewidth=0.1)
plt.plot()
plt.xlabel('ppm', fontname="Arial", fontsize=9, fontweight='bold')
plt.xlim((xlim))
plt.gca().invert_xaxis()
plt.ylim((-0.03,0.035))
ax.text(4.2,0.024,"NO",fontsize=9,fontname="Arial",color='black')
ax.text(4.2,0.003,"DL",fontsize=9,fontname="Arial",color='black')
ax.text(4.2,-0.017,"SR",fontsize=9,fontname="Arial",color='black')
plt.title('No Added Offsets'
          "\n", fontname="Arial", fontsize=9, fontweight='bold')

plt.rc('axes', linewidth=1)
ax = plt.subplot(242)    
ax.spines["top"].set_visible(False)       
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)   
ax.tick_params(width=1,color='k',length=2)
plt.xticks(np.arange(2,4.3,step=1), fontsize=9)
ax.set_yticklabels([])
ax.axes.get_yaxis().set_visible(False)  
plt.plot(ppm, diffNO_offset1+0.02,'k',linewidth=0.1)
plt.plot(ppm, diffDL_offset1,'k',linewidth=0.1)
plt.plot(ppm, diffSR_offset1-0.02,'k',linewidth=0.1)
plt.plot()
plt.xlabel('ppm', fontname="Arial", fontsize=9, fontweight='bold')
plt.xlim((xlim))
plt.gca().invert_xaxis()
plt.ylim((-0.03,0.035))
plt.title(r'$ |\Delta f| \leq 5\/Hz\/$'
          "\n" 
          r'$\/|\Delta \phi| \leq 20\degree$', fontname="Arial", fontsize=9, fontweight='bold')

plt.rc('axes', linewidth=1)
ax = plt.subplot(243)    
ax.spines["top"].set_visible(False)       
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)   
ax.tick_params(width=1,color='k',length=2)
plt.xticks(np.arange(2,4.3,step=1), fontsize=9)
ax.set_yticklabels([])
ax.axes.get_yaxis().set_visible(False)  
plt.plot(ppm, diffNO_offset2+0.02,'k',linewidth=0.1)
plt.plot(ppm, diffDL_offset2,'k',linewidth=0.1)
plt.plot(ppm, diffSR_offset2-0.02,'k',linewidth=0.1)
plt.plot()
plt.xlabel('ppm', fontname="Arial", fontsize=9, fontweight='bold')
plt.xlim((xlim))
plt.gca().invert_xaxis()
plt.ylim((-0.03,0.035))
plt.title(r'$ 5 \leq  |\Delta f| \leq 10\/Hz\/$'
          "\n" 
          r'$ 20 \leq  |\Delta \phi| \leq 45\degree$', fontname="Arial", fontsize=9, fontweight='bold')

plt.rc('axes', linewidth=1)
ax = plt.subplot(244)    
ax.spines["top"].set_visible(False)       
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)   
ax.tick_params(width=1,color='k',length=2)
plt.xticks(np.arange(2,4.3,step=1), fontsize=9)
ax.set_yticklabels([])
ax.axes.get_yaxis().set_visible(False)  
plt.plot(ppm, diffNO_offset3+0.02,'k',linewidth=0.1)
plt.plot(ppm, diffDL_offset3,'k',linewidth=0.1)
plt.plot(ppm, diffSR_offset3-0.02,'k',linewidth=0.1)
plt.plot()
plt.xlabel('ppm', fontname="Arial", fontsize=9, fontweight='bold')
plt.xlim((xlim))
plt.gca().invert_xaxis()
plt.ylim((-0.03,0.035))

plt.title(r'$ 10 \leq  |\Delta f| \leq 20\/Hz\/$'
          "\n" 
          r'$ 45 \leq  |\Delta \phi| \leq 90\degree$', fontname="Arial", fontsize=9, fontweight='bold')

Q_DLSR = np.empty(diffDL[0,:].shape)         #No Added Offset: DL vs SR correction
Q_DLSR_offset1 = np.empty(diffDL[0,:].shape) #Offset 1 added: DL vs SR correction
Q_DLSR_offset2 = np.empty(diffDL[0,:].shape) #Offset 2 added: DL vs SR correction
Q_DLSR_offset3 = np.empty(diffDL[0,:].shape) #Offset 3 added: DL vs SR correction

for mm in range(len(Q_DLSR_offset1)):  
              
    freqInt = np.where((ppm <= 3.285) & (ppm >= 3.15))
    
    std_SA_DL = np.std(np.squeeze(diffDL[freqInt[0][0]:freqInt[0][-1],mm].real))
    std_SA_SR = np.std(np.squeeze(diffSR[freqInt[0][0]:freqInt[0][-1],mm].real))
    std_SA_DL_offset1 = np.std(np.squeeze(diffDL_offset1[freqInt[0][0]:freqInt[0][-1],mm].real))
    std_SA_SR_offset1 = np.std(np.squeeze(diffSR_offset1[freqInt[0][0]:freqInt[0][-1],mm].real))
    std_SA_DL_offset2 = np.std(np.squeeze(diffDL_offset2[freqInt[0][0]:freqInt[0][-1],mm].real))
    std_SA_SR_offset2 = np.std(np.squeeze(diffSR_offset2[freqInt[0][0]:freqInt[0][-1],mm].real))         
    std_SA_DL_offset3 = np.std(np.squeeze(diffDL_offset3[freqInt[0][0]:freqInt[0][-1],mm].real))
    std_SA_SR_offset3 = np.std(np.squeeze(diffSR_offset3[freqInt[0][0]:freqInt[0][-1],mm].real))  
       
    Q_DLSR[mm] = 1 - (np.square(std_SA_DL))/ (np.square(std_SA_SR) + np.square(std_SA_DL))
    Q_DLSR_offset1[mm] = 1 - (np.square(std_SA_DL_offset1))/ (np.square(std_SA_SR_offset1) + np.square(std_SA_DL_offset1))
    Q_DLSR_offset2[mm] = 1 - (np.square(std_SA_DL_offset2))/ (np.square(std_SA_SR_offset2) + np.square(std_SA_DL_offset2))
    Q_DLSR_offset3[mm] = 1 - (np.square(std_SA_DL_offset3))/ (np.square(std_SA_SR_offset3) + np.square(std_SA_DL_offset3))

#Plot the quality metrics  
ax = plt.subplot(245)    
ax.spines["top"].set_visible(False)       
ax.spines["right"].set_visible(False)     
ax.tick_params(width=1,color='k',length=2)
plt.xticks(range(9,34,10), [str(x) for x in range(10,35,10)], fontsize=9)
plt.yticks(np.arange(0,1.2,step=1), fontsize=9)
plt.ylabel('Q', fontname="Arial", fontsize=9, fontweight='bold')
plt.xlabel('Dataset', fontname="Arial", fontsize=9, fontweight='bold')
mSize = 1
for kk in range(len(Q_DLSR)):
        
    if Q_DLSR[kk] > 0.5: 
        if kk==1:
            plt.plot(kk, Q_DLSR[kk],'bo', label='DL > SR', markersize=mSize)
        else:
            plt.plot(kk, Q_DLSR[kk],'bo', label='_nolegend_', markersize=mSize)         
    else:
        if kk==2:
            plt.plot(kk, Q_DLSR[kk],'ro', label='DL < SR', markersize=mSize)
        else:
            plt.plot(kk, Q_DLSR[kk],'ro', label='_nolegend_', markersize=mSize)   
        
plt.plot(0.5*np.ones(Q_DLSR_offset1.shape),'k--',linewidth=1)
lgnd = plt.legend(loc='lower left',frameon=False,prop={'weight':'bold','size': 6})
lgnd.legendHandles[0]._legmarker.set_markersize(3)
lgnd.legendHandles[1]._legmarker.set_markersize(3)
plt.ylim((0,1))

ax = plt.subplot(246)    
ax.spines["top"].set_visible(False)       
ax.spines["right"].set_visible(False)     
plt.xticks(range(9,34,10), [str(x) for x in range(10,35,10)], fontsize=9)
plt.yticks(np.arange(0,1.2,step=1), fontsize=9)
plt.xlabel('Dataset', fontname="Arial", fontsize=9, fontweight='bold')
for kk in range(len(Q_DLSR_offset1)):
        
    if Q_DLSR_offset1[kk] > 0.5: 
        if kk==4:
            plt.plot(kk, Q_DLSR_offset1[kk],'bo', label='DL > SR', markersize=mSize)
        else:
            plt.plot(kk, Q_DLSR_offset1[kk],'bo', label='_nolegend_', markersize=mSize)         
    else:
        if kk==5:
            plt.plot(kk, Q_DLSR_offset1[kk],'ro', label='DL < SR', markersize=mSize)
        else:
            plt.plot(kk, Q_DLSR_offset1[kk],'ro', label='_nolegend_', markersize=mSize)   
        
plt.plot(0.5*np.ones(Q_DLSR_offset1.shape),'k--',linewidth=1)
lgnd = plt.legend(loc='lower left',frameon=False, fontsize=9,prop={'weight':'bold','size': 6}) 
lgnd.legendHandles[0]._legmarker.set_markersize(3)
lgnd.legendHandles[1]._legmarker.set_markersize(3)
plt.ylim((0,1))

ax = plt.subplot(247)    
ax.spines["top"].set_visible(False)       
ax.spines["right"].set_visible(False)     
plt.xticks(range(9,34,10), [str(x) for x in range(10,35,10)], fontsize=9)
plt.yticks(np.arange(0,1.2,step=1), fontsize=9)
plt.xlabel('Dataset', fontname="Arial", fontsize=9, fontweight='bold')
for kk in range(len(Q_DLSR_offset2)):
        
    if Q_DLSR_offset2[kk] > 0.5: 
        if kk==0:
            plt.plot(kk, Q_DLSR_offset2[kk],'bo', label='DL > SR', markersize=mSize)
        else:
            plt.plot(kk, Q_DLSR_offset2[kk],'bo', label='_nolegend_', markersize=mSize)         
    else:
        if kk==28:
            plt.plot(kk, Q_DLSR_offset2[kk],'ro', label='DL < SR', markersize=mSize)
        else:
            plt.plot(kk, Q_DLSR_offset2[kk],'ro', label='_nolegend_', markersize=mSize)   
        
plt.plot(0.5*np.ones(Q_DLSR_offset2.shape),'k--',linewidth=1)
lgnd = plt.legend(loc='lower left',frameon=False, fontsize=9,prop={'weight':'bold','size': 6}) 
lgnd.legendHandles[0]._legmarker.set_markersize(3)
lgnd.legendHandles[1]._legmarker.set_markersize(3)
plt.ylim((0,1))    

ax = plt.subplot(248)    
ax.spines["top"].set_visible(False)       
ax.spines["right"].set_visible(False)    
plt.xticks(range(9,34,10), [str(x) for x in range(10,35,10)], fontsize=9)
plt.yticks(np.arange(0,1.2,step=1), fontsize=9)
plt.xlabel('Dataset', fontname="Arial", fontsize=9, fontweight='bold')
for kk in range(len(Q_DLSR_offset3)):
        
    if Q_DLSR_offset3[kk] > 0.5: 
        if kk==0:
            plt.plot(kk, Q_DLSR_offset3[kk],'bo', label='DL > SR', markersize=mSize)
        else:
            plt.plot(kk, Q_DLSR_offset3[kk],'bo', label='_nolegend_', markersize=mSize)         
    else:
        if kk==1:
            plt.plot(kk, Q_DLSR_offset3[kk],'ro', label='DL < SR', markersize=mSize)
        else:
            plt.plot(kk, Q_DLSR_offset3[kk],'ro', label='_nolegend_', markersize=mSize)   
        
plt.plot(0.5*np.ones(Q_DLSR_offset3.shape),'k--',linewidth=1)
lgnd = plt.legend(loc='lower left',frameon=False, fontsize=9,prop={'weight':'bold','size': 6}) 
lgnd.legendHandles[0]._legmarker.set_markersize(3)
lgnd.legendHandles[1]._legmarker.set_markersize(3)
plt.ylim((0,1))     

temp = np.where(Q_DLSR_offset1 > 0.5)
percent_DLvsSR1 = len(temp[0])/len(FidsON[0,0,:]) * 100

temp = np.where(Q_DLSR_offset2 > 0.5)
percent_DLvsSR2 = len(temp[0])/len(FidsON[0,0,:]) * 100

temp = np.where(Q_DLSR_offset3 > 0.5)
percent_DLvsSR3 = len(temp[0])/len(FidsON[0,0,:]) * 100

print("DL vs SR 1: ", "{0:.2f}".format(round(percent_DLvsSR1,2)), "%")
print("DL vs SR 2: ", "{0:.2f}".format(round(percent_DLvsSR2,2)), "%")
print("DL vs SR 3: ", "{0:.2f}".format(round(percent_DLvsSR3,2)), "%")

print("DL vs SR 1 MEAN: ", "{0:.2f}".format(round(np.mean(Q_DLSR_offset1),2)))
print("DL vs SR 2 MEAN: ", "{0:.2f}".format(round(np.mean(Q_DLSR_offset2),2)))
print("DL vs SR 3 MEAN: ", "{0:.2f}".format(round(np.mean(Q_DLSR_offset3),2)))

plt.subplots_adjust(wspace=0.2, hspace=0.4,left=0.05,right=0.99, bottom=0.1, top=0.9)    

