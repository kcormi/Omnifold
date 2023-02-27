# <img src="img/omnifold_logo.png" width="100"> OmniFold
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ericmetodiev/OmniFold/master)

A method for universally unfolding collider data with machine learning-based reweighting. Check out the [demo](https://mybinder.org/v2/gh/ericmetodiev/OmniFold/master?filepath=OmniFold%20Demo.ipynb)!

OmniFold paper: https://arxiv.org/abs/1911.09107

Emily Dickinson, \#975  
>The Mountain sat upon the Plain  
>In his tremendous Chair&mdash;  
>His observation omnifold,  
>His inquest, everywhere&mdash;  
>  
>The Seasons played around his knees  
>Like Children round a sire&mdash;  
>Grandfather of the Days is He  
>Of Dawn, the Ancestor&mdash;  
# Omnifold
Changes to add the efficiency and acceptance into the unfolding (currently only implemented for the multifold).
Log into the tier 3 account.
```
source /cvmfs/sft.cern.ch/lcg/views/LCG_97py3cu10/x86_64-centos7-gcc7-opt/setup.sh
source /work/jinw/omnifold/ef/bin/activate
mkdir results_multifold_maxweight10_MCEPOS_unfoldCP1_1p3M_eff_acc
mkdir results_multifold_maxweight10_MCEPOS_unfoldCP1_1p3M_eff_acc/models
mkdir results_multifold_maxweight10_MCEPOS_unfoldCP1_1p3M_eff_acc/weights
```
To run in in the GPU:
```
sbatch train_gpu.sh
```
Otherwise
```
bash train_gpu.sh
```




