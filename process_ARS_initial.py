###########################################################################
#
#   Process_ars_initial.py
#
#
#############################################################################
'''
    
This is the code to process the data for an ARS timeseries
	
Testforburning_v1 did it just the standard 'subtraction' way
Testforburning_v2 does it by subtracting - but then looking at the signal as a percentage of the average signal observed. If less than 0.1% - ignored. 
    
'''
################################################################################
#
#   Imports
#
#
#
import matplotlib  #Uncomment if you are on a linux system
matplotlib.use('Agg') # Uncomment if you are on a linux system

import rs_tools as rt
import matplotlib.pyplot as plt
import os
import numpy as np

from copy import deepcopy


outdir_1 = ''
outdir = 'figures'

day_list = ['20160825r13']

x_values = rt.get_rebs_calibration(cal_file='rn13_cal_20160825_paper.txt',cal_type='fit') 
x_values_wl = rt.get_rebs_calibration(cal_file='rn13_cal_20160825_paper.txt',cal_type='fit',return_wl=True) 

os.makedirs("figures",exist_ok=True)

#####
##
## Multipliers for bleach and replicate data
##  CAUTION! MAKE SURE THIS IS SET CORRECTLY!
##
#####
bbm = 1 #Bleach Multiplier
rbm = 1 #REplicate Multiplier

#Days to analyze 


#Initialize all three values

spec_date = np.array([])
spec_sum = np.array([])
spec_sum_fluosub = np.array([])

#Cl: This is cleaning, saturation, bad
#Cs: this is cosmic
#Bkr: Background
#Brn: Burning Removal
#Fluo_e0
#Fluo_e1

fname_base0 = 'alldata_r13_TimeSeries_0_Cl'
fname_data0 = fname_base0 + '.rdat'
fname_meta0 = fname_base0 + '.rmta'  

fname_base1 = 'alldata_r13_TimeSeries_1_ClCs'
fname_data1 = fname_base1 + '.rdat'
fname_meta1 = fname_base1 + '.rmta'  

fname_base2 = 'alldata_r13_TimeSeries_2_ClCsBkr'
fname_data2 = fname_base2 + '.rdat'
fname_meta2 = fname_base2 + '.rmta'  

fname_base3 = 'alldata_r13_TimeSeries_3_ClCsBkrBrn'
fname_data3 = fname_base3 + '.rdat'
fname_meta3 = fname_base3 + '.rmta'  

fname_base4 = 'alldata_r13_TimeSeries_4_ClCsBkrBrnFle0'
fname_data4 = fname_base4 + '.rdat'
fname_meta4 = fname_base4 + '.rmta'  

fname_base5 = 'alldata_r13_TimeSeries_5_ClCsBkrBrnFle6'
fname_data5 = fname_base5 + '.rdat'
fname_meta5 = fname_base5 + '.rmta'    
fname_fluo5 = fname_base5 + '.rflu'
path_to_output = outdir_1

rnno='13_'


path_data0 = os.path.join(path_to_output,fname_data0)
path_meta0 = os.path.join(path_to_output,fname_meta0)

path_data1 = os.path.join(path_to_output,fname_data1)
path_meta1 = os.path.join(path_to_output,fname_meta1)

path_data2 = os.path.join(path_to_output,fname_data2)
path_meta2 = os.path.join(path_to_output,fname_meta2)

path_data3 = os.path.join(path_to_output,fname_data3)
path_meta3 = os.path.join(path_to_output,fname_meta3)

path_data4 = os.path.join(path_to_output,fname_data4)
path_meta4 = os.path.join(path_to_output,fname_meta4)

path_data5 = os.path.join(path_to_output,fname_data5)
path_meta5 = os.path.join(path_to_output,fname_meta5)
path_fluo5 = os.path.join(path_to_output,fname_fluo5)

x_step = 2.0

starting_x_location=0
locnum = 0
isstart = 0

number_burning = np.array([])
number_cosmic = np.array([])
number_saturation = np.array([])
matrix_saturation = np.empty((0,5))

path_to_output = outdir

all_directories = np.load("directoryload.npy") #List of all the folderst to load. This can be different on different operating systems.

######Data 0--------------------------------------------------------->##Data 1--------------------------------------------------------->##Data 2--------------------------------------------------------->###Data 3 -------------------------------------------------------->###Data 4 -------------------------------------------------------->##Data 5 -------------------------------------------------------->##Data 6 -------------------------------------------------------->
with open(path_data0,'w') as f_data0, open(path_meta0,'w') as f_meta0,open(path_data1,'w') as f_data1, open(path_meta1,'w') as f_meta1, open(path_data2,'w') as f_data2, open(path_meta2,'w') as f_meta2, open(path_data3,'w') as f_data3, open(path_meta3,'w') as f_meta3, open(path_data4,'w') as f_data4, open(path_meta4,'w') as f_meta4, open(path_data5,'w') as f_data5, open(path_meta5,'w') as f_meta5, open(path_fluo5,'w') as f_fluo5:
    
    datastr = '%1.4f' + ',%1.4f'*1023 + '\n' 
    
    f_data0.write(datastr % tuple(x_values))
    f_meta0.write('DateTime,x,y,z,r,cn,fl,fla,rmx,p2,p3,p4,p5\n')
    
    f_data1.write(datastr % tuple(x_values))
    f_meta1.write('DateTime,x,y,z,r,cn,fl,fla,rmx,p2,p3,p4,p5\n')

    f_data2.write(datastr % tuple(x_values))
    f_meta2.write('DateTime,x,y,z,r,cn,fl,fla,rmx,p2,p3,p4,p5\n')
    
    f_data3.write(datastr % tuple(x_values))
    f_meta3.write('DateTime,x,y,z,r,cn,fl,fla,rmx,p2,p3,p4,p5\n')
    
    f_data4.write(datastr % tuple(x_values))
    f_meta4.write('DateTime,x,y,z,r,cn,fl,fla,rmx,p2,p3,p4,p5\n')
    
    f_data5.write(datastr % tuple(x_values))
    f_fluo5.write(datastr % tuple(x_values_wl))
    f_meta5.write('DateTime,x,y,z,r,cn,fl,fla,rmx,p2,p3,p4,p5\n')
                                                                
    
            
    for i,directory in enumerate(all_directories):
            
    #This one can be modified to handle multiple files    
        myCollection = rt.load_spotinfo(directory)
        #import pdb
        #pdb.set_trace()    
        #######################
        #
        #Process Collection
        #
        myCollection = rt.clean_collection(myCollection)
        myCollection = rt.use_bleach(myCollection) #COnverts the bleach into another replicate       
        myCollection = rt.remove_saturation(myCollection) #removes spectra that are saturated
        myCollection = rt.clean_badlocs(myCollection,rn=13) #Removes bad pixels or places where there is dust on the CCD
        myCollection = rt.add_binwn(myCollection,x_values) #Add the X values to the collection
        
        myCollection_0 = deepcopy(myCollection) 
    
        #######################
        #Uncomment for raw data figure dump
        #rt.output_rebs_images(myCollection,'figures_raw',bin_wavenumber=rt.get_rebs_calibration(13),note='')
        
        #######################
        #
        # Output Saturation Data
        #
        number_saturation = np.append(number_saturation,myCollection.Summary.NSaturation)
        mysat = myCollection.Summary.Saturation_Matrix

        
        if len(mysat) > 0:
            mysat[:,0] = i
        
        matrix_saturation = np.vstack((matrix_saturation,mysat))
        
        #######################
        #
        #   Uncomment for removal of cosmic rays 
        #
        myCollection = rt.remove_cosmic(myCollection,plot=False)    
        myCollection_1 = deepcopy(myCollection)                 
    
        ######################
        #   Uncomment for BKR Subtraction
        #            
        synthetic_bkr = rt.compute_bkr_collection(myCollection) 
        rt.plot_allspectra(synthetic_bkr,'syntheticbkr',x_values=x_values)
        myCollection = rt.collection_subtract_bkr(myCollection, synthetic_bkr)
        myCollection_2 = deepcopy(myCollection)  

        #####################
        #0
        #   Uncomment to check for burning
        #
        
        file_save = os.path.join(outdir,rnno +'banddiff'+ myCollection.Summary.Save_Name )
        #myCollection,banddiff= rt.extract_banddiff(myCollection,file_save=file_save,limits=[1300,1650],bin_wavenumber=x_values,vmax=0.1,vmin=-0.1,threshold=0.005)
        
        myCollection,banddiff= rt.detect_charring(myCollection,file_save=file_save,limits=[1300,1650],bin_wavenumber=x_values,vmax=0.1,vmin=-0.1,threshold_l=np.NaN)
        
        #def detect_charring(nrtd,limits=[-9999,-9999],limits_as_wavenumber = True, bin_wavenumber = None, make_plot=True,file_save='',vmax=0.1,vmin=-0.1,threshold_l=0,count_burning=True):
        
        number_burning = np.append(number_burning,myCollection.Summary.nBurning)
        
        if isstart == 0:
            myname = directory
            allbanddiffs = banddiff
            isstart==1
        else:
            myname.insert(directory)
            allbanddiffs = np.dstack(allbanddiffs,banddiff) 
    
        myCollection_3 = deepcopy(myCollection)
        
        number_cosmic = np.append(number_cosmic,myCollection.Summary.nCosmic)
        
        
        ####################
        #
        # Apply quality checks to spectrum
        #
        
        myCollection = rt.qc_spectrum(myCollection)
        myCollection_0 = rt.qc_spectrum(myCollection_0)
        myCollection_1 = rt.qc_spectrum(myCollection_1)
        myCollection_2 = rt.qc_spectrum(myCollection_2)
        myCollection_3 = rt.qc_spectrum(myCollection_3)
        #######################
        #UNcomment for Fluorescence removal
        #Notes: We create two collections to handle
        #       The fact that we will show results from two fluorescence removals.
        
        myCollection_4 = deepcopy(myCollection)
        
        myCollection = rt.remove_fluorescence(myCollection,p=0.001,lmb=1e6)
        
        myCollection_4 = rt.remove_fluorescence(myCollection_4,p=0.001,lmb=1)
        
        mywmin = 450
        mywmax = 3200
    
        #Integrate the spectra. 
        myCollection = rt.integrate_spectra(myCollection,wmin = mywmin)#,wmax=mywmax)
        myCollection_4 = rt.integrate_spectra(myCollection_4,wmin = mywmin)#,wmax=mywmax)
        
        #######################
        #
        # Uncomment to dump data to dircube
        #
        #dc,dx,dz,t,fl,rs,rx = rt.collection_process(myCollection,proc_fluo=True) 
                


        dc0,dx0,dz0,t0,fl0,fa0,rx0 = rt.collection_process(myCollection_0)    
        dc1,dx1,dz1,t1,fl1,fa1,rx1 = rt.collection_process(myCollection_1)  
        dc2,dx2,dz2,t2,fl2,fa2,rx2 = rt.collection_process(myCollection_2)  
        dc3,dx3,dz3,t3,fl3,fa3,rx3 = rt.collection_process(myCollection_3)  
        dc4,dx4,dz4,t4,fl4,fa4,rx4 = rt.collection_process(myCollection_4,proc_fluo=True)   
        dc5,dx5,dz5,t5,fl5,fa5,rx5 = rt.collection_process(myCollection,proc_fluo=True)  
        fc5,fx5,fz5,f5,ff5,fv5,fx5 = rt.collection_process_fl(myCollection)    
            
        #######################
        #
        #   Uncomment to dump images from dircube
        #
        #rt.output_rebs_images_dc(dc*5.0,outdir,basename_image=myCollection.Summary.Save_Name)
        
        #######################
        #
            #Uncomment to output images to file - needed for journal article
        # 
        rt.dc_output_sep(f_data0,dc0,f_meta0,t0,dx0,dz0,collection_num=i,Fsum=fl0,Fmax = fa0,Rmax=rx0)
        rt.dc_output_sep(f_data1,dc1,f_meta1,t1,dx1,dz1,collection_num=i,Fsum=fl1,Fmax = fa1,Rmax=rx1)
        rt.dc_output_sep(f_data2,dc2,f_meta2,t2,dx2,dz2,collection_num=i,Fsum=fl2,Fmax = fa2,Rmax=rx2)
        rt.dc_output_sep(f_data3,dc3,f_meta3,t3,dx3,dz3,collection_num=i,Fsum=fl3,Fmax = fa3,Rmax=rx3)
        rt.dc_output_sep(f_data4,dc4,f_meta4,t4,dx4,dz4,collection_num=i,Fsum=fl4,Fmax = fa4,Rmax=rx4)
        rt.dc_output_sep_fl(f_data5,dc5,f_fluo5,fc5,f_meta5,t5,dx5,dz5,collection_num=i,Fsum=fl5,Fmax = fa5,Rmax=rx5)
        #if i == 0:
            #break
        locnum=locnum + 1  
        plt.close('all')
        #please_stop_here()
                     
import pandas as pd
mydt = [os.path.basename(mydir).split(' ')[2][3::] for mydir in all_directories]
mydatetime = pd.to_datetime(mydt,format='%Y%m%d_%H%M%S')
d = {'DateTime':mydatetime,'nBurning':number_burning,'nCosmic':number_cosmic}
df=pd.DataFrame(data=d)
df_1 = df.sort_values(by=['DateTime'])
df_2 = df_1.set_index('DateTime')
df_2.to_csv('TS_burncos.csv')

np.save('saturations',matrix_saturation)


np.save('banddiffs',allbanddiffs)
#np.save('nBurning',number_cosmic)
#np.save('nCosmic',number_burning)
