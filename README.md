This repository contains the python code and data used in the project "deep learning based frequency and phase correction (FPC) of edited magnetic resonance spectroscopy (MRS) data". 

The training, validation, and in vivo data can be found here: https://www.dropbox.com/sh/7n9pdo1vyp683fz/AAByLeFO6uSQuclCCBTWrYWVa?dl=0

The training of the two networks model_F0.model and model_Ph.model was conducted using the DLFPC_Training.py script together with the TrainingData. 

The networks were validated using the DLFPC_Validation.py script together with the ValidationData.

The networks were further tested using the DLFPC_InVivoTest.py script, where different amounts of added offsets were added to regular Philips in vivo edited MRS data (Big GABA repository). In this step, the network performance were compared to conventional spectral registration computed using Gannet. The results from Gannet are also saved (inVivo_AddedOffset + GlobalSR) and used in the comparison. Networks performance was investigated when 1. With no added offsets, 2. Small added offsets, 3. Medium added offsets, 4. Large added offsets. 


Sofie Tapper
stapper2 at jh.edu
