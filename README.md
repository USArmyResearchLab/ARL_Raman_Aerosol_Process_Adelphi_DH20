# UNDER CONSTRUCTION

# ARL_Raman_Aerosol_Process_Adelphi_DH20

---

Version 1.0.

13 December 2019

This software allows for the reproduction of the figures contained in Doughty and Hill 2020, entitled "Raman spectra of atmospheric particles measured in Maryland, USA over 22.5 hrs using an automated aerosol Raman spectrometer". 



# Motivation

---

Reproduce the results in Doughty and Hill 2020. We (Doughty and Hill) also believe that the code and data use to do science should be made easily available after publication of a paper. This way others can easily use it for their own research, and also potentially detect bugs or problems in the software.

# Description 

---

There are two main steps, preprocessing and analysis. As preprocessing can take a long time, we do this step separately from the 'analysis' step, and save the data in a .rdta, .rmta, and .rflu files, corresponding to data, meta, and fluorescence files, respectively. Preprocessing steps can include removal of bad pixels, removal of cosmic rays, calculation of background, removal of background, removal of fluorescence, removal of saturated spectra, and interpolation across bad pixels, and combination of replicates into an average, median, or summed spectrum. Preprocessing is done in python.  The file [rs_tools](rs_tools.py) includes routines to accomplish all of these tasks, as well as many other routines for plotting, and general processing of Raman spectral data. At the end of the preprocessing step, data are saved as two dimensional files, with each row (other than the header) containing a Raman spectrum (.rdta), a fluorescence spectrum (.rflu), or metadata associated with the associated spectra (.rmta).

There are many ways to analyze Raman data, we show only a few in this dataset. 

# Getting Started

---

These instructions will get you a copy of the project up and running on your local machine to reproduce the figures found in the main paper, and most of the supplement figures, for Doughty and Hill 2020.

## Prerequisites

To run the software, you need an installation of Python 3, and an installation of R. 

Python: It is recommended to use [Anaconda](https://www.anaconda.com/distribution/) or [Enthought](https://store.enthought.com/downloads/) python due to the inclusion of package managers. Packages needed include matplotlib, numpy, pandas, and scipy. 


## Installing and Configuration

Unzip the file or pull the file from the git server. Make sure that the rs_tools.py file is either in the working directory, or in a path that python searches. Make sure that the 'figures' directory is in your working directory. 

Download the data into the main directory. If you have placed the data correctly, the main directory should look like:
```
/20160825r13
/Heights
/figures
/*.py (represents all python files in this repo)
/directoryload.npy
/rn13_cal_20160825_paper.txt
```

# Running the software

---

There are several steps for both preprocessing and final analysis.

## 1. Preprocess data using different steps.


To generate initial data files:
```
python3 process_ARS_initial.py

```
This code can also be run within the python3 interpreter using:

```
exec(open("process_ARS_initial.py").read())

```
This creates .rdta, .rmta, and .rflu files with different amounts of preprocessing. These files are very large. This includes 

  - alldata_r13_TimeSeries_1_ClCs - Only cleaning and removal of saturation, and removal of cosmic rays 
  - alldata_r13_TimeSeries_2_ClCsBkr - All the previous steps plus removal of the background
  - alldata_r13_TimeSeries_3_ClCsBkrBrn - All the previous steps plus detection of charring
  - alldata_r13_TimeSeries_4_ClCsBkrBrnFle0 - All the previous steps plus removal of fluorescence with lambda = 1. This removes nearly all broad features 
  - alldata_r13_TimeSeries_5_ClCsBkrBrnFle6 - All the previous steps plus removal of fluorescence with lambda = 10^6. This leavesbroad features such as the D/G bands. 
  
This code takes a while to run.


## 2. Generate the figures for the paper

The following python file Generates the figures for the main paper. 

```
python GenerateFigures_DH20a_processing.py
```

Note that this code calls [generatefigures_dh19a_rdat_Fig5-10+FigS10-S12_BB.py](generatefigures_dh19a_rdat_Fig5-10+FigS10-S12_BB.py)

# 3. Public Domain Dedication

The text of the CC0 license can be found [here](https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt). A local copy of the license is in the file license.txt.
