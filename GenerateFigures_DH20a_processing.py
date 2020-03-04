'''
This is the code to generate the figures in 
Doughty and Hill (2020) on Raman Processing
Before you run, make the current working directory of python the directory of this file
before you run this, you need to run process_ARS_initial.py and have the .rdat, .rmta, .csv, and .npy outputs. 

'''

#Imports

########
## The following two imports should be uncommented to generate figures on a linux system.
#import matplotlib 
#matplotlib.use('Agg') 

#these imports are always needed

import matplotlib.pyplot as plt
import rs_tools as rt
import os
import numpy as np
import glob
import pandas as pd
import matplotlib.dates as dates

############# Helper Functions


def extract_banddiff(nrtd,limits=[-9999,-9999],limits_as_wavenumber = True, bin_wavenumber = None, make_plot=True,file_save='',vmax=0.1,vmin=-0.1,threshold=0):
    '''
    This extracts the difference between bands between the first and the last replicate. 
    I am going to go with the simple solution (first) - then we'll see what happens.
    Inputs:
    Outputs:
    Example: 
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    #######################
    #
    #   Determine limits
    #
    if limits[0] < 0 or limits[1] < 0:
        #If any limits are less than zero, assume you want the whole spectrum
        limits = [0,1023]
        limits_as_wavenumber = False #
    
    if limits_as_wavenumber==True:

        try:
            limit_upper = rt.find_nearest(limits[1],bin_wavenumber)
            limit_lower = rt.find_nearest(limits[0],bin_wavenumber)
            #print(limit_lower,limit_upper)
        except:
            print('Error (Extract_Banddiff): could not find bin index')
            print('You likely need to supply wavenumber values')
            return None                    
    else:
        limit_lower = limits[0]
        limit_upper = limits[1] + 1 #note we add one
    '''
    #############################
    #
    #   Determine Band Image Plot
    #                                                                                                                                  
    num_rows,num_wns,num_replicates = nrtd.Image[0].Replicate_Data.shape
    num_images = len(nrtd.Image)
    
    diffcube = np.zeros((num_rows,num_images))
    
    for i,Image in enumerate(nrtd.Image):
        
        #Average the replicate data
        repmean = np.nanmean(Image.Replicate_Data[:,limit_lower:limit_upper,:],axis=1)
        bmean = np.nanmean(Image.Bleach_Data[:,limit_lower:limit_upper],axis=1)
        #print repmean.shape
        #print bmean.shape
        allmeans = np.hstack((repmean,bmean[:,None]))
        meanvalue = np.nanmean(allmeans,axis=1)
        #print allmeans.shape
        #print meanvalue.shape
        
        nmm = np.nanmax(repmean,axis=1)

        diffcube[:,i] = np.divide(np.subtract(nmm,bmean),meanvalue)
        
        diffcube[meanvalue<=threshold,i] = np.NaN
    '''    
    #############################
    #
    #   Determine Band Image Plot
    #                                                                                                                                  
    num_rows,num_wns,num_replicates = nrtd.Image[0].Replicate_Data.shape
    num_images = len(nrtd.Image)
    
    diffcube = np.zeros((num_rows,num_images))
    
    for i,Image in enumerate(nrtd.Image):
        
        meanvs = np.nanmean(Image.Replicate_Data[:,limit_lower:limit_upper,:],axis=1)
        allmean = np.nanmean(meanvs,axis=1)

        diffcube[:,i] = np.subtract(meanvs[:,-1],meanvs[:,0])
        diffcube[allmean<=threshold,i] = np.NaN

    output_band_image = diffcube

    if make_plot:          
    
        fig = plt.figure()   
        ax = fig.add_subplot(111)   
    
        x,y = np.meshgrid(range(num_images),range(num_rows))
        if vmax<0:
            vmax=np.nanmax(output_band_image)
            vmin=-1*vmax
            
        imgplot= ax.pcolormesh(x,y,output_band_image,vmin=vmin,linewidth=0,cmap='bwr',vmax=vmax) 
        cbh = plt.colorbar(imgplot,orientation='vertical') 
        ax.set_title(nrtd.Summary.Save_Name + ' Rep2-Bleach' )                                
        ax.set_xlabel('Spectrum Number')
        ax.set_ylabel('Vertical Distance (micron)')
        ax.set_xlim([0,np.nanmax(x)])
        ax.set_ylim([0,np.nanmax(y)])
        cbh.set_label('Mean Raman Intensity Change 1300-1650 cm-1 (a.u.)')
        savename= file_save + '_dimg.png'
        plt.savefig(savename,transparent=True)
 
    return output_band_image
 
                      
   
def plot_loc(nrtd,row,column,myax):
    xv = np.zeros(len(nrtd.Image[0].Replicate_Name))

    for Image in nrtd.Image:
        xv[:] = Image.Replicate_Time
        
        results = myax.plot(xv,Image.Replicate_Data[row,column,:],'.k')
    return results

   
      
#############################################################################
#
#
#
#
#   Figure making Functions
#
#   
        
def make_Figure2():
    
    import pdb
    rt.plot_format(fs=10) 
    dir_plus = os.path.join("heights","*+*.csv")
    dir_minus = os.path.join("heights","*-*.csv")
    x_values = rt.get_rebs_calibration(cal_file='rn13_cal_20160825_paper.txt',cal_type='fit')
    
    allfiles_pl = glob.glob(dir_plus)
    allfiles_mn = glob.glob(dir_minus)
    
    allfiles_pl.sort()
    allfiles_mn.sort()
    allfiles_mn.reverse()
    allfiles = allfiles_mn+ allfiles_pl
    
    #pdb.set_trace()
    
    
    begin=True 
    
    legendvalue = []
    
    for myfile in allfiles:
        try:
            mydata = np.loadtxt(myfile,delimiter=',',skiprows=3)
        except:
            pdb.set_trace()
            
        if begin==True:
            alldata = mydata
            legendvalue = [myfile[26:29]]
            begin=False
        else:
            alldata = np.dstack((alldata,mydata))
            
            try:
                legendvalue.append(myfile[26:29])
            except:
                pdb.set_trace()
                
                
    mydata_zero = np.loadtxt(os.path.join("heights","RN14_20170508_H0.csv"),delimiter=',',skiprows=3)
    mydata_zero_plt = mydata_zero[40,:]
    
                   
    #pdb.set_trace()
    alldata_plot=alldata[40,:,:]  
    #alldata_plot=alldata[40,:,10:17]
    
    x,y = alldata_plot.shape
    #pdb.set_trace()
    
    
    
    #Part 3: Final Plots
    
    f,ax = plt.subplots(1,1,figsize=(4,4))
    cc = [plt.cm.jet(i) for i in np.linspace(0, 1, 9)]

    
    
    min_spc = 11
    max_spc = 18
    #TOp subplot
    line_below = ax.plot(x_values,alldata_plot[:,0:min_spc],color=cc[0],alpha=0.8,lw=1)
    
    
    
    h_line = [line_below[0]]
    
    for i,idx in enumerate(range(min_spc,max_spc)):
        myline = ax.plot(x_values,alldata_plot[:,idx],color=cc[i+1],alpha=0.8,lw=1)
        h_line.append(myline[0])
        
    line_above = ax.plot(x_values,alldata_plot[:,max_spc::],color=cc[-1],alpha=0.8,lw=1) 
    h_line.append(line_above[0]) 
    
    import pdb
    #pdb.set_trace()    
    

    from matplotlib.font_manager import FontProperties

    fontP = FontProperties()
    fontP.set_size('small')
    legendvalue = [mylv + ' $\mu$m' for mylv in legendvalue]
    legend_txt = ['<' + legendvalue[min_spc-1]] + legendvalue[min_spc:max_spc] + ['>'+legendvalue[max_spc]]
    ax.legend(h_line,legend_txt,loc='lower center',ncol=3,prop=fontP,handlelength=1)
    
    
    ax.set_xlabel('Raman Shift (cm$^{-1}$)')
    ax.set_ylabel('Raman Intensity (A.U.)')
    ax.set_xlim([400,3200])
    ax.set_ylim([0,2500])
    plt.tight_layout()
    
    '''
    #Lower subplot
    x,y = np.meshgrid(rcal_wl,range(nrows))
    imgplot = ax[1].pcolormesh(x,y,mdata,cmap='viridis')
    ax[1].set_xlim([min(rcal_wl),max(rcal_wl)])
    ax[1].set_ylim([0,127])
    ax[1].text(660,110,'b.',bbox=dict(facecolor='white'))
    cbh = plt.colorbar(imgplot,orientation='vertical')
    cbh.set_label('Intensity (A.U.)')
    ax[1].set_xlabel('Wavelength (nm)')
    ax[1].set_ylabel('CCD Vertical axis along\n laser line (pixel nmbr)')
    '''
    
    plt.savefig("DH19a_Processing_Figure02.png")
        
                
            
def make_Figure4():


    rt.plot_format(fs=10) 
    x_values = rt.get_rebs_calibration(cal_file='rn13_cal_20160825_paper.txt',cal_type='fit')
    
    myDirectory = os.path.join('20160825r13','REBS RN 13_20160825_001238')
    myCollection = rt.load_spotinfo(myDirectory)
    myCollection = rt.clean_collection(myCollection)
    myCollection = rt.use_bleach(myCollection)        
    myCollection = rt.remove_saturation(myCollection)
    myCollection = rt.clean_badlocs(myCollection,rn=13)
    myCollection = rt.add_binwn(myCollection,x_values)
    myCollection = rt.remove_cosmic(myCollection,plot=False)            
    synthetic_bkr = rt.compute_bkr_collection(myCollection) 
    myCollection = rt.collection_subtract_bkr(myCollection, synthetic_bkr)
    
    from copy import deepcopy
    myCollection_1 = deepcopy(myCollection)

    junk,banddiff = rt.detect_charring(myCollection_1,limits=[1300,1650],bin_wavenumber=x_values,make_plot=False)
    #banddiff = extract_banddiff(myCollection,limits=[1300,1650],bin_wavenumber=x_values,make_plot=False)
    num_rows,num_images = banddiff.shape
    #import pdb
    #pdb.set_trace() 
    f, (ax1,ax2,ax3) = plt.subplots(3, 1,figsize=(4.1,7))
         
    #A

    #=plt.figure(figsize=(4.1,7))
    #ax1 = plt.subplot2grid((3,6), (0,0), colspan=6)
    #ax2 = plt.subplot2grid((3,6), (1,0), colspan=6)
    #cax = plt.subplot2grid((3,6), (1,5))
    #ax3 = plt.subplot2grid((3,6), (2,0), colspan=6)
    myim = 6
    myrw = 19    
    ax1.plot(x_values,myCollection.Image[myim].Replicate_Data[myrw,:,:])
    ax1.legend(['0-10s','10-20s','20-30s'])
    ax1.set_xlabel('Raman Shift (cm$^{-1}$)')
    ax1.set_ylabel('Raman Intensity (A.U.)')
    ax1.set_xlim([300,3200])
    ax1.text(450,0.4,'(a)')
    
    #Download burning timeseries
    #import pdb
    #pdb.set_trace()
    
    import matplotlib.colors as mcolors


    colors1 = plt.cm.bwr(np.linspace(0, 1,220))
    colors2 = plt.cm.bwr(np.linspace(0, 1,22))
    colors = np.vstack((colors2[0:12,:],colors1[110::,:]))
    cmap = mcolors.LinearSegmentedColormap.from_list('colormap', colors)
    # generating a smoothly-varying LinearSegmentedColormap
    #cmap=plt.cm.jet
    #cmap.set_under([0.5,0.5,0.5])
    cmap.set_under('w')

    #FOr plotting purposes, normally we would do this:
    #x,y = np.meshgrid(range(num_images),range(num_rows))
    #im1= ax2.pcolormesh(x,y,banddiff,vmin=-0.05,linewidth=0,cmap='bwr',vmax=0.05) 
    #However, we need to represent that the x position spacing is two. 

    banddiff_nu = np.zeros((num_rows,num_images*2))
    
    for i in range(num_images):
        myind = i*2
        banddiff_nu[:,myind] = banddiff[:,i]
        banddiff_nu[:,myind+1] = -10

    num_rows,num_cols = banddiff_nu.shape
    import pdb
    #pdb.set_trace()
    x_nu,y_nu = np.meshgrid(range(num_cols),range(num_rows))
    im1= ax2.pcolormesh(x_nu,y_nu,banddiff_nu,vmin=-1,linewidth=0,cmap=cmap,vmax=11)
    '''
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    cax2 = inset_axes(ax2,
                    width="2%",  # width = 10% of parent_bbox width
                    height="60%",  # height : 50%
                    loc='center right',
                    borderpad = 6)
    '''

    
    

    #cbh = plt.colorbar(im1,orientation='vertical',cax=cax)                          
    cbh = plt.colorbar(im1,ax=ax2,orientation='vertical')                          
    #cax1.set_xtick([0.04,0,-0.04])
    #cbh.set_label('r',fontsize=10)
    ax2.set_xlabel('Stepping Axis ($\mu$m)')
    ax2.set_ylabel('Laser Long Axis ($\mu$m)')
    ax2.set_xlim([0,np.nanmax(x_nu)])
    ax2.set_ylim([0,np.nanmax(y_nu)])
    #ax2.arrow(myim*2.0+0.5,myrw+0.5,4,-5,head_starts_at_zero=True,head_width=0.01)
    ax2.annotate("",xy=(myim*2.0+0.5,myrw+0.5),xytext=(myim*2.0+3.5,myrw-3.5),arrowprops=dict(arrowstyle="simple",color='C2'))
    
    #ax2.scatter(np.array([myim*2.0 + 0.5]),np.array([myrw+0.5]),facecolors='none',edgecolors='k')
    ax2.text(1.8,35,'(b)')
    ax2.text(65,20,'r-1')
    #rectangle=plt.Rectangle((17.5,4.2),7,35,fill=False)
    #ax2.add_patch(rectangle)
   
    #C 
    import pandas as pd
    import matplotlib.dates as dates
    from scipy.signal import savgol_filter,medfilt

    my_df = pd.read_csv('TS_burncos.csv',parse_dates=[0])

    mydt = my_df['DateTime'].values
    vals_burn = my_df['nBurning'].values
    #vals_burn_smooth = savgol_filter(vals_burn,21,4)
    #vals_burn_smooth = medfilt(vals_burn,kernel_size=21)

    ax3.plot(mydt,vals_burn,'k')
    ax3.plot(mydt,vals_burn,'o',markerfacecolor='C2',markeredgecolor='k')
    #ax3.plot(mydt,vals_burn_smooth)
    ax3.text(pd.to_datetime('2016-08-25 1:15:00'),11,'(c)')
    ax3.set_xlim(pd.to_datetime(['2016-08-25','2016-08-26']))
    ax3.xaxis.set_major_locator(dates.HourLocator(interval=4))
    ax3.xaxis.set_major_formatter(dates.DateFormatter('%H'))
    ax3.set_xlabel('Hour of Day (EDT)')
    ax3.set_ylabel('# Charring Spectra')
    plt.subplots_adjust(hspace=0.1)
    
    brntxt_1 = 'Total Number of Charring Spectra:' + str(np.nansum(vals_burn))
    brntxt_2 = 'Average Number of Charring Spectra/collection:' + str(np.nanmean(vals_burn))
    plt.tight_layout()
    print(brntxt_1)
    print(brntxt_2)
    
    plt.savefig('DH19a_Processing_Figure04.png')

def make_Figure3():
    
    rt.plot_format(fs=10) 
    #directory = os.path.join("20160825r13","REBS RN 13_20160825_033821")
    directory = os.path.join("20160825r13","REBS RN 13_20160825_172824")
    myCollection = rt.load_spotinfo(directory)
    from copy import deepcopy
    myCollection_raw = deepcopy(myCollection)
    
    myCollection = rt.clean_collection(myCollection)
    myCollection = rt.use_bleach(myCollection)        
    myCollection = rt.clean_badlocs(myCollection,rn=13)      
    myCollection = rt.remove_cosmic(myCollection,plot=False)
    synthetic_bkr = rt.compute_bkr_collection(myCollection) 
    myCollection = rt.collection_subtract_bkr(myCollection, synthetic_bkr)
    dc,dx,dz,t,fl,ft,rx = rt.collection_process(myCollection)  
                
    #imgno=25
    #rowno = 33            
    imgno=20
    rowno = 37            
    #myspec = dc[33,:,25]
    myspec = dc[rowno,:,imgno]
    
    myfluo = rt.background_als_core_nu(myspec,handle_end=True,p=0.001,lmb=1e6)
    myfluo_1 = rt.background_als_core_nu(myspec,handle_end=True,p=0.001,lmb=1)
    
    import pdb
    
    f, ax = plt.subplots(3, sharex=True,figsize=(6,8))
    myColors = rt.nicePalette()
    #Top part of graph
    #This shows speectrum, and estimated fluorescence
    x_values = rt.get_rebs_calibration(cal_file='rn13_cal_20160825_paper.txt',cal_type='fit')
    myrepdata = myCollection_raw.Image[imgno].Replicate_Data[rowno,:,:]
    myblcdata = myCollection_raw.Image[imgno].Bleach_Data[rowno,:]
    ax[0].plot(x_values,myblcdata,'C0',alpha=0.8,linewidth=2)
    ax[0].plot(x_values,myrepdata[:,0],'C3',alpha=0.8,linewidth=1.2)
    ax[0].plot(x_values,myrepdata[:,1],'C2',alpha=0.8,linewidth=0.7)
    ax[0].axis([350,3175,0.01,0.07])
    ax[0].legend(['Rep 1','Rep 2','Rep 3'],loc='upper right',frameon=False,ncol=1,prop={'size':12})
    ax[0].text(1200,0.05,'(a) Raw Replicate Spectra')
    ax[1].plot(x_values,myspec,color='k')
    ax[1].plot(x_values,myfluo,color=myColors[1])
    ax[1].plot(x_values,myfluo_1,color=myColors[4])
    ax[1].fill_between(x_values,myfluo,facecolor=myColors[1], alpha=0.2,edgecolor="white")
    ax[1].fill_between(x_values,myfluo_1,facecolor=myColors[4], alpha=0.2,edgecolor="white")
    ax[1].axis([350,3175,0,0.029])
    ax[1].legend(['Processed','$\lambda$=10$^6$','$\lambda$=1'],loc='center right',frameon=False,ncol=1,prop={'size':12})
    ax[1].text(500,0.025,'(b) Processed Spectrum/Estimated Fluorescence')
    #ax[1].text(1200,0.0016,'Est. Fluorescence = $F_{rem}$')
    ax[1].set_ylabel('Raman Intensity (A.U.)')
    
    
    #Remove fluorescence
    specdiff = myspec-myfluo
    specdiff_1 = myspec-myfluo_1
    
    specdiff[0:94] = 0
    specdiff_1[0:94] = 0
    
    maxv = np.nanmax(specdiff)
    maxx = x_values[specdiff==maxv][0]
    ymainv = np.zeros(len(x_values))
    ymainv[:] = maxv
    
    maxv_1 = np.nanmax(specdiff_1)
    maxx_1 = x_values[specdiff_1==maxv_1][0]
    ymainv_1 = np.zeros(len(x_values))
    ymainv_1[:] = maxv_1
    
    #Make second part of plot showing the 
    #removed fluorescence and Rmax
    
    ax[2].plot(x_values,specdiff,color=myColors[0])
    ax[2].plot(x_values,specdiff_1,color=myColors[3])
   
    import pdb
    #pdb.set_trace()
    
    xloc_0 = np.where(specdiff==maxv)[0][0]
    arrloc = 900
    xloc_0_arr = rt.find_nearest(arrloc,x_values)

    
    xloc_1 = np.where(specdiff_1==maxv_1)[0][0]
    arrloc_1 = 1900
    xloc_1_arr = rt.find_nearest(arrloc_1,x_values)
       


    ax[2].plot(x_values[xloc_0_arr:xloc_0],ymainv[xloc_0_arr:xloc_0],color=myColors[0],alpha=0.5)
    ax[2].plot(x_values[xloc_1:xloc_1_arr],ymainv_1[xloc_1:xloc_1_arr],color=myColors[3],alpha=0.5)
    
    ax[2].axis([350,3175,0,0.024])
    
    ax[2].arrow(arrloc,0,0,maxv,length_includes_head=True,overhang=0.1,head_width=50,head_length=0.001,facecolor='k',zorder=2)
    ax[2].arrow(arrloc_1,0,0,maxv_1,length_includes_head=True,overhang=0.1,head_width=50,head_length=0.001,facecolor='k',zorder=2)
    ax[2].text(550,0.013,'$R_{max}$,\n $\lambda$=10$^6$') 
    ax[2].text(1850,0.004,'$R_{max}$,\n $\lambda$=1') 
    #pdb.set_trace()

    '''
    ax[2].annotate("$R_{max}$",
                xy=(maxx, maxv), xycoords='data',
                xytext=(1750, 0.012), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3"),
                )
    ax[2].annotate("$R_{max}$",
                xy=(maxx_1, maxv_1), xycoords='data',
                xytext=(1750, 0.012), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3"),
                )
    '''            
                
    #ax[2].set_ylabel('Fluorescence Removed \n Raman Intensity (A.U.)')            
    ax[2].text(500,0.021,'(c) Spectra after fluorescence subtraction')
    ax[2].legend(['$\lambda$=10$^6$','$\lambda$=1'],loc='center right',frameon=False,ncol=1,prop={'size':12})
    ax[2].set_xlabel('Wavenumber (cm$^{-1}$)')
    f.subplots_adjust(hspace=0.05)
    #pdb.set_trace() 
    plt.savefig('DH19a_Processing_Figure03.png')


def load_specdata_1(file2load):
    with open(file2load,'r') as f:
        myline = f.readline().strip().split()
        lenmax = len(myline)
        col_range = range(6,lenmax)
        data = np.loadtxt(file2load,usecols=col_range,delimiter='\t')
    return data

def get_spectrum(myDT,myX,myY,mymeta,mydata):
        myspecind= int(mymeta[(mymeta['DateTime'] == myDT) & (mymeta['x']==myX) & (mymeta['y'] == myY)].index.values)
        myspec = mydata[myspecind]
        return myspec,myspecind
    
def make_Figure11():
    import matplotlib.patches as patches
    import matplotlib as mpl
    
    ####################################
    #
    #   Grab base Data
    #
    #
    
    alldata_base = "alldata_r13_TimeSeries_2_ClCsBkr"
    file_rdat = alldata_base + ".rdat"
    file_rmta = alldata_base + ".rmta"
    rt.plot_format(fs=10) 
    rdat = np.loadtxt(file_rdat,delimiter=',')
    dt_meta = pd.read_table(file_rmta,sep=',')
    x_values = rdat[0,:]
    rdatdata = rdat[1::,:]
    #Loads the Fluorescence Removed Spectrum
    print("Loaded Non Fl-subtracted data")
    alldata_base = "alldata_r13_TimeSeries_5_ClCsBkrBrnFle6"
    file_rdat = alldata_base + ".rdat"
    file_rmta = alldata_base + ".rmta"
    file_rflu = alldata_base + ".rflu"
    rdat_f = np.loadtxt(file_rdat,delimiter=',')
    dt_meta_f = pd.read_table(file_rmta,sep=',')
    #x_values_f = rdat_f[0,:]
    rdatdata_f = rdat_f[1::,:]
    rfluo_f = np.loadtxt(file_rflu,delimiter=',')
    #x_values_fluo = rfluo_f[0,:]
    rdatfluo_f = rfluo_f[1::,:]
    
    #####################
    #
    #Subset the data
    #
    
    #data_fluo_ini = dt_meta_f['fl']
    data_rmax_ini = dt_meta_f['rmx'] 
    data_fmax_ini = dt_meta_f['fla']
    #data_fmax_ini = np.nanmax(rdatfluo_f[:,100::],axis=1)
    
    #These are the data to use in calculating for the vs plots!!
    data_rmax = data_rmax_ini[(data_rmax_ini>0)]
    data_fmax = data_fmax_ini[(data_rmax_ini>0)]
    
    from copy import deepcopy
    nufluo = deepcopy(data_fmax)
    nufluo[nufluo<0.0001] = 0.0001
    data_fmax[data_fmax<0.001] = 0.001
    data_rat = np.true_divide(data_rmax,data_fmax)
    
    #import pdb
    #pdb.set_trace() 
    
    #########################
    #
    #Get data for the subplots
    #
    
    spec_loc = ('2016-08-25 07:52:15',20,20)
    spec_b,ind_b = get_spectrum(spec_loc[0],spec_loc[1],spec_loc[2],dt_meta,rdatdata)
    spec_b_fsub = rdatdata_f[ind_b]
    spec_b_fluo = rdatfluo_f[ind_b]
    fl_b = data_fmax[ind_b]
    rmx_b = dt_meta_f['rmx'][ind_b]
    
    spec_loc = ('2016-08-25 10:46:14',8,10)
    spec_c,ind_c = get_spectrum(spec_loc[0],spec_loc[1],spec_loc[2],dt_meta,rdatdata)
    spec_c_fsub = rdatdata_f[ind_c]
    spec_c_fluo = rdatfluo_f[ind_c]
    fl_c = data_fmax[ind_c]
    rmx_c = dt_meta_f['rmx'][ind_c]
    
    spec_loc = ('2016-08-25 03:06:38',26,19)
    spec_f,ind_f = get_spectrum(spec_loc[0],spec_loc[1],spec_loc[2],dt_meta,rdatdata)
    spec_f_fsub = rdatdata_f[ind_f]
    spec_f_fluo = rdatfluo_f[ind_f]
    fl_f = data_fmax[ind_f]
    rmx_f = dt_meta_f['rmx'][ind_f]

    spec_loc = ('2016-08-25 01:16:20',14,13)
    spec_e,ind_e = get_spectrum(spec_loc[0],spec_loc[1],spec_loc[2],dt_meta,rdatdata)
    spec_e_fsub = rdatdata_f[ind_e]
    spec_e_fluo = rdatfluo_f[ind_e]
    fl_e = data_fmax[ind_e]
    rmx_e = dt_meta_f['rmx'][ind_e]
    
    spec_loc = ('2016-08-25 10:30:20',6,23)
    #spec_loc = ('2016-08-25 03:06:38',0,4)
    spec_d,ind_d = get_spectrum(spec_loc[0],spec_loc[1],spec_loc[2],dt_meta,rdatdata)
    spec_d_fsub = rdatdata_f[ind_d]
    spec_d_fluo = rdatfluo_f[ind_d]
    fl_d = data_fmax[ind_d]
    rmx_d = dt_meta_f['rmx'][ind_d]
    
    spec_loc = ('2016-08-25 03:06:38',0,4)
    #spec_loc = ('2016-08-25 10:30:20',6,23)
    spec_g,ind_g = get_spectrum(spec_loc[0],spec_loc[1],spec_loc[2],dt_meta,rdatdata)
    spec_g_fsub = rdatdata_f[ind_g]
    spec_g_fluo = rdatfluo_f[ind_g]
    fl_g = data_fmax[ind_g]
    rmx_g = dt_meta_f['rmx'][ind_g]
    
    
    ###################################
    #
    #   Make the plot!
    #
                            
    f, ax = plt.subplots(figsize=(8,8))
    
    
    
    #Main axis
    
    axloc_a = [0.075,0.3,0.60,0.68]
    ax_a = f.add_axes(axloc_a) 
    
    axloc_cb = [0.1,0.88,0.3,0.05]
    ax_cb = f.add_axes(axloc_cb)
    
    axloc_b = [0.69,0.80,0.25,0.2]
    ax_b = f.add_axes(axloc_b)
    
    axloc_c = [0.69,0.55,0.25,0.2]
    ax_c = f.add_axes(axloc_c)
    
    axloc_d = [0.69,0.30,0.25,0.2]
    ax_d = f.add_axes(axloc_d)
    
    axloc_e = [0.105,0.05,0.25,0.2]
    ax_e = f.add_axes(axloc_e)
    
    axloc_f = [0.415,0.05,0.25,0.2]
    ax_f = f.add_axes(axloc_f)
    
    axloc_g = [0.73,0.05,0.25,0.2]
    ax_g = f.add_axes(axloc_g)
    
    ax.axis('off')
    
    mycmap = 'viridis_r'
    Fig11Name='DH19a_Processing_Figure11.png'
    lc = 'k'# [0.3,0.3,0.3]
    la = 0.7
    mlw = 1
    myvmin = 0.3
    myvmax=5
    
    sct = ax_a.scatter(data_fmax,data_rmax,c=data_rat,cmap=mycmap,vmin=myvmin,vmax=myvmax,s=1,norm=mpl.colors.LogNorm())
    ax_a.text(-0.04,0.55,'R$_{max}$ (A.U.)',transform = ax_a.transAxes,horizontalalignment='center',rotation=90)
    ax_a.text(0.50,-0.04,'F$_{max}$ (A.U.)',transform = ax_a.transAxes,horizontalalignment='center')
    ax_a.text(0.075,0.96,'(a)',transform = ax_a.transAxes)         

    ax_a.set_yscale('log')
    ax_a.set_xscale('log')    
    ax_a.set_xlim([0.00095,1])
    ax_a.set_ylim([0.00095,1]) 
    ax_a.axvspan(0.00091,0.00103, alpha=0.5, color=[0.5,0.5,0.5])
    ax_a.text(0.0007,0.0007,'           ',bbox=dict(facecolor='white',edgecolor='white'))
    ax_a.text(0.0007,0.0007,'<=10$^{-3}$',bbox=dict(facecolor=[0.5,0.5,0.5], alpha=0.5))



    import pdb
    #pdb.set_trace()
    from scipy.signal import savgol_filter
    
       
    ax_b.plot(x_values,spec_b,alpha=0.8)
    ax_b.plot(x_values,spec_b_fsub,alpha=0.5)
    ax_b.plot(x_values,spec_b_fluo,alpha=0.5)

    ax_b.yaxis.tick_right()
    ax_b.yaxis.set_label_position("right")
    ax_b.set_xticks([500,1500,3000])
    ax_b.set_xlim([450,3150])
    ax_b.text(0.075,0.85,'(b)',transform = ax_b.transAxes)
    line = mpl.lines.Line2D([fl_b,10],[rmx_b,0.46],color=lc,alpha=la,lw=1,ls='--')
    plt.setp(ax_b.spines.values(), color=lc,alpha=la,lw=mlw)
    plt.setp([ax_b.get_xticklines(), ax_b.get_yticklines()], color=lc,alpha=la,lw=mlw)
    ax_a.add_line(line)
    line.set_clip_on(False)

    ax_c.plot(x_values,spec_c,alpha=0.8)
    ax_c.plot(x_values,spec_c_fsub,alpha=0.5)
    ax_c.plot(x_values,spec_c_fluo,alpha=0.5)
    ax_c.yaxis.tick_right()
    ax_c.yaxis.set_label_position("right")
    ax_c.set_xticks([500,3000])
    ax_c.set_xlim([450,3150])
    ax_c.text(0.075,0.85,'(c)',transform = ax_c.transAxes)
    plt.setp(ax_c.spines.values(), color=lc,alpha=la,lw=mlw)
    plt.setp([ax_c.get_xticklines(), ax_c.get_yticklines()], color=lc,alpha=la,lw=mlw)
    line = mpl.lines.Line2D([fl_c,10],[rmx_c,0.03],color=lc,alpha=la,lw=1,ls='--')
    ax_a.add_line(line)
    line.set_clip_on(False)

    ax_d.plot(x_values,spec_d,alpha=0.8)
    ax_d.plot(x_values,spec_d_fsub,alpha=0.5)
    ax_d.plot(x_values,spec_d_fluo,alpha=0.5)
    #ax_c.set_yticklabels('')
    ax_d.yaxis.tick_right()
    ax_d.yaxis.set_label_position("right")
    #ax_c.set_ylabel('Raman Intensity (A.U.)')
    ax_d.set_xticks([500,3000])
    ax_d.set_xticklabels(['         500','3000'])
    ax_d.set_xlim([450,3150])
    #ax_d.text(0.3,-0.25,'  Raman\nShift ($cm^{-1}$)',transform = ax_d.transAxes)
    ax_d.text(0.075,0.72,'(d)',transform = ax_d.transAxes)
    plt.setp(ax_d.spines.values(), color=lc,alpha=la,lw=mlw)
    plt.setp([ax_d.get_xticklines(), ax_d.get_yticklines()], color=lc,alpha=la,lw=mlw)
    line = mpl.lines.Line2D([fl_d,10],[rmx_d,0.001],color=lc,alpha=la,lw=1,ls='--')
    ax_a.add_line(line)
    line.set_clip_on(False)

    ax_e.plot(x_values,spec_e,alpha=0.8)
    ax_e.plot(x_values,spec_e_fsub,alpha=0.5)
    ax_e.plot(x_values,spec_e_fluo,alpha=0.5)
    spec_e_smoothed = savgol_filter(spec_e,15,2)
    ax_e.plot(x_values,spec_e_smoothed,alpha=0.8)
    #ax_d.set_yticklabels('')
    #ax_d.set_ylabel('Raman Intensity (A.U.)')
    ax_e.set_ylabel('Intensity (A.U.)')
    ax_e.set_xticks([500,3000])
    ax_e.set_xlim([450,3150])
    ax_e.text(0.3,-0.25,'  Raman\nShift ($cm^{-1}$)',transform = ax_e.transAxes)
    ax_e.text(0.075,0.85,'(e)',transform = ax_e.transAxes)
    ax_e.set_ylim([-.0009,0.002])
    plt.setp(ax_e.spines.values(), color=lc,alpha=la,lw=mlw)
    plt.setp([ax_e.get_xticklines(), ax_e.get_yticklines()], color=lc,alpha=la,lw=mlw)
    line = mpl.lines.Line2D([fl_e,0.002],[rmx_e,0.0005],color=lc,alpha=la,lw=1,ls='--')
    ax_a.add_line(line)
    line.set_clip_on(False)

    ax_f.plot(x_values,spec_f,alpha=0.8)
    ax_f.plot(x_values,spec_f_fsub,alpha=0.5)
    ax_f.plot(x_values,spec_f_fluo,alpha=0.5)
    #ax_e.set_yticklabels('')
    #ax_e.set_ylabel('Raman Intensity (A.U.)')
    ax_f.set_xticks([500,3000])
    ax_f.set_xlim([450,3150])
    ax_f.text(0.3,-0.25,'  Raman\nShift ($cm^{-1}$)',transform = ax_f.transAxes)
    ax_f.text(0.075,0.85,'(f)',transform = ax_f.transAxes)
    plt.setp(ax_f.spines.values(), color=lc,alpha=la,lw=mlw)
    plt.setp([ax_f.get_xticklines(), ax_f.get_yticklines()], color=lc,alpha=la,lw=mlw)
    line = mpl.lines.Line2D([fl_f,0.2],[rmx_f,0.0005],color=lc,alpha=la,lw=1,ls='--')
    ax_a.add_line(line)
    line.set_clip_on(False)

    ax_g.plot(x_values,spec_g,alpha=0.8)
    ax_g.plot(x_values,spec_g_fsub,alpha=0.5)
    ax_g.plot(x_values,spec_g_fluo,alpha=0.5)
    #ax_f.set_yticklabels('')
    #ax_f.set_ylabel('Raman Intensity (A.U.)')
    ax_g.set_xticks([500,3000])
    ax_g.set_xlim([450,3150])
    ax_g.text(0.3,-0.25,'  Raman\nShift ($cm^{-1}$)',transform = ax_g.transAxes)
    ax_g.text(0.8,0.85,'(g)',transform = ax_g.transAxes)
    plt.setp(ax_g.spines.values(), color=lc,alpha=la,lw=mlw)
    plt.setp([ax_g.get_xticklines(), ax_g.get_yticklines()], color=lc,alpha=la,lw=mlw)
    line = mpl.lines.Line2D([fl_g,3],[rmx_g,0.0003],color=lc,alpha=la,lw=1,ls='--')
    ax_a.add_line(line)
    line.set_clip_on(False)

    ax_cb.set_title('R$_{max}$/F$_{max}$')
    cbh = plt.colorbar(sct,cax=ax_cb,orientation='horizontal')
    #cbh.set_ticks([0.5,5])

    lc =  patches.Rectangle((1.25,0.007),0.05,0.001,linewidth=2,edgecolor='r',facecolor='none')
    ax.add_patch(lc)
    
    
    plt.savefig(Fig11Name)    




def cluster_in_cluster(input_file_base,input_file_comp,output_file):

    data_base = pd.read_csv(input_file_base)
    dt_base = pd.to_datetime(data_base["myspc.DateTime"].values)
    cv_base = data_base[["myspc.x","myspc.y","myspc.clusters"]].values
    
    
    data_comp = pd.read_csv(input_file_comp)
    dt_comp = pd.to_datetime(data_comp["myspc.DateTime"].values)
    cv_comp = data_comp[["myspc.x","myspc.y","myspc.clusters"]].values
    
    
    
    ml = cv_base.shape[0]
    uv_base = np.zeros(ml) #Number of values in the base without an analog in 'comp'
    uv_comp = np.zeros(cv_comp.shape[0]) #number of values in the comp wihtout an analog in 'base'
    
    #Really bad - we are dynamically allocating an array
    #Deal with it.
    dt_sec = np.array([])
    cv_sec = np.zeros(4)
    cv_sec[:] = np.nan
    
    
    for i in range(ml):
        #Loop over every value in CV_base    
        
        myloc = [(dt_base[i]==dt_comp) & (cv_base[i,0] == cv_comp[:,0]) & (cv_base[i,1] == cv_comp[:,1])]
        goodval = cv_comp[myloc[0],2]
        if len(goodval) == 1:
    
            dt_sec = np.append(dt_sec,dt_base[i])
            savearr = np.array([cv_base[i,0],cv_base[i,1],cv_base[i,2],goodval])
            cv_sec = np.vstack((cv_sec,savearr))
        elif len(goodval) == 0:
            uv_base[i] = 1
        elif len(goodval) > 1:
            raise ValueError('I found multiple matches - there is a problem with your archives')
            
    
    max_input_clusters=np.nanmax(cv_base[:,2])
    
    with open(output_file,"w") as f:
        f.write("Comparison of Clusters\nfrom file:\n" + input_file_base + " \nin file:\n" + input_file_comp + "\n\n")
        f.write("-----------------------------------------------\n")
        for i in range(max_input_clusters):
            mc = i+1
            
            matched_arr_2 = 0
            #Initial work: get number of clusters not matched, and number of clusters matched
            total_arr = np.where(cv_base[:,2] == mc)[0]
            matched_arr = np.where((uv_base == 0) & (cv_base[:,2] == mc))[0]
            unmatched_arr = np.where((uv_base == 1) & (cv_base[:,2] == mc))[0]
    
            
            f.write("Cluster " + str(mc) + "| Total:" + str(len(total_arr)) +  "| Matched:" + str(len(matched_arr)) + " | Unmatched:" + str(len(unmatched_arr)) + "\n")
            good_arr = np.where(cv_sec[:,2]==mc)
            cv_sec_nu = cv_sec[good_arr[0],:]
            good_subclusters = np.unique(cv_sec_nu[:,3])
            for msc in good_subclusters:
                nr,nc = cv_sec_nu[cv_sec_nu[:,3]==msc].shape
                f.write("\t" + str(int(msc)) + ":" + str(int(nr)))
                f.write("\n")
                matched_arr_2 = matched_arr_2 + nr
            
            #Get number of clusters that did not compare
            f.write("\tSum:"+ str(matched_arr_2))
            f.write("\n")        
    
            
            f.write("-----------------------------------------------\n")
            
	

def make_FigureS2():
    ''' Code to generate Figure S2: Cosmic Ray Removal
    Essentially it does the standard load, but when
    running the remove_cosmic subroutine, plot=True
    So it dumps out the plots. 
    
    #This is the code to generate cosmic_data
    master_folder = os.path.join('Data,'raw13')
    n_cosmic = []
    t_collection = []  
    
    
    day_list = ['20160825r13']
    
    for day in day_list:
        
        all_directories = glob.glob(os.path.join(master_folder,day,'REBS*'))
        
        for i,directory in enumerate(all_directories):
            print directory
            try:
        
                myCollection = rt.load_spotinfo(directory)
                myCollection = rt.clean_collection(myCollection)
                myCollection = rt.use_bleach(myCollection)        
                myCollection = rt.remove_saturation(myCollection)
                myCollection = rt.remove_cosmic(myCollection,plot=False)
                n_cosmic = np.append(n_cosmic,myCollection.Summary.nCosmic)
                t_collection = np.append(t_collection,myCollection.Summary.Imaging_Start)
            except:
                continue
            
    
        np.save('cosmic.txt',np.vstack((t_collection,n_cosmic)))


    '''
    from copy import deepcopy
    
    myDirectory = os.path.join('20160825r13','REBS RN 13_20160825_082403')
    myCollection = rt.load_spotinfo(myDirectory)
    myCollection = rt.clean_collection(myCollection)
    myCollection = rt.use_bleach(myCollection)        
    myCollection = rt.remove_saturation(myCollection)
    myCollection = rt.clean_badlocs(myCollection,rn=13)
    myCollection = rt.add_binwn(myCollection,rt.get_rebs_calibration(cal_file='rn13_cal_20160825_paper.txt',cal_type='fit'))
    
    #Code below here is copied and slightly edited from remove_cosmic. 
    #Get Image 7
    
    Image = myCollection.Image[7]
    
    j = 27
    k=2
    
    oldrows = deepcopy(Image.Replicate_Data[j,:,:])

    ##THis is the part that shows you where the cosmic rays are removed
    myRow = Image.Replicate_Data[j,:,k]
    myComparison = np.vstack((Image.Replicate_Data[j,:,:k].transpose(),Image.Replicate_Data[j,:,k+1:].transpose()))
    crl,sdb,sdc = rt.remove_cosmic_core(myRow,myComparison)
    all_crl = np.hstack((crl,np.add(crl,1),np.add(crl,-1)))
    Image.Replicate_Data[j,all_crl,k] = np.nan
    Image.Replicate_Data[j,:,k] = rt.lininterp_nan(Image.Replicate_Data[j,:,k])
    
    #Here we Generate the plot for Figure 1
    cgy = "#a0a0a0"
    co = "#f4a442"
    cgn = "#2f9143"
    
    x = rt.get_rebs_calibration(cal_file='rn13_cal_20160825_paper.txt',cal_type='fit')
    
    f, ax = plt.subplots(3,figsize=(3.5,9))   

    ax[0].plot(x,oldrows[:,k],color=co)
    ax[0].plot(x,Image.Replicate_Data[j,:,k],color=cgy)
    ax[0].plot(x[crl],oldrows[crl,k],'o', markerfacecolor="None",markeredgecolor='k')
    ax[0].set_ylim([0,np.nanmax(Image.Replicate_Data[j,100::,k])+0.01])
    ax[0].text(400,0.035,'(a)',fontsize=10)
    ax[0].set_ylabel('Intensity (A.U.)')

    ax[1].plot(x,sdb,color=cgn)
    ax[1].plot(x,sdc,color=cgy)
    ax[1].set_ylim([-0.002,0.002])
    ax[1].text(400,0.0015,'(b)',fontsize=10)
    ax[1].set_ylabel('Second Derivative (A.U.)')
    
    #remove_cosmic_core calcs
    dr = sdb-sdc
    z_score = rt.calculate_z_score(dr)               
    ax[2].plot(x,z_score,color=cgn)
    ax[2].plot([x[0],x[-1]],[-5,-5],'k')
    ax[2].set_ylim([-15,10])
    ax[2].text(400,5,'(c)',fontsize=10)
    ax[2].set_ylabel('Z-score')
    ax[2].set_xlabel('Wavenumber (cm$^{-1}$)')
    #Load the time-series of the cosmic rays removed
    #Plot it here. 
    

    plt.tight_layout()
    plt.savefig("Figure_S2.png")

def make_FigureS3():
    import pandas as pd
    import matplotlib.dates as mdates
    from scipy.signal import savgol_filter,medfilt
    import pdb
    my_df = pd.read_csv('TS_burncos.csv',parse_dates=[0])

    mydt = my_df['DateTime'].values
    vals_ncos = my_df['nCosmic'].values
    vals_ncos_smooth = savgol_filter(vals_ncos,21,4)
    #vals_burn_smooth = medfilt(vals_burn,kernel_size=21)
    	
    f,ax=plt.subplots(1)
    ax.plot(mydt,vals_ncos)
    ax.plot([mydt[0],mydt[-1]],[10.6,10.6],'k')
    ax.plot(mydt,vals_ncos_smooth,'k')
    
    #hour = mdates.HourLocator()
    hourFmt = mdates.DateFormatter('%H')
    #ax[3].xaxis.set_major_locator(hour)
    ax.xaxis.set_major_formatter(hourFmt)
    ax.set_ylabel('#Cosmic Rays Removed')
    ax.set_xlabel('Hour (EDT)')   
    plt.legend(['#Cos','Smooth','10.6'])
    plt.tight_layout()
    plt.savefig("Figure_S3.png")

	

def collection_process_dark(nrtd,manual_nreps = False, nreps=0):
    import numpy as np
    #Processes the collection data
    #Makes a 'dircube' This is a 3-d dataset corresponding to:
    # (laser_dimension,wavenumber_dimension,image number)
    #Used for Make_figure_s6
    #                                                               
    num_rows,num_wns,num_replicates = nrtd.Image[0].Replicate_Data.shape 
    num_images = len(nrtd.Image)
    
    final_dircube = np.zeros((num_rows,num_wns,num_images))
    final_time = np.zeros((num_images))

    for i,Image in enumerate(nrtd.Image):

        final_time[i] = Image.Replicate_Time
        if manual_nreps == True:
            output_mat = Image.Replicate_Data[:,:,0:nreps] 
        else:
            output_mat = Image.Replicate_Data          
            
        final_dircube[:,:,i] = np.nanmedian(output_mat,axis=2)
            
    return (final_dircube,final_time)                             
   
def plot_loc(nrtd,row,column,myax):
    xv = np.zeros(len(nrtd.Image[0].Replicate_Name))

    for Image in nrtd.Image:
        xv[:] = Image.Replicate_Time
        
        results = myax.plot(xv,Image.Replicate_Data[row,column,:],'.k')
    return results

   
def make_FigureS4():
    #Code to generate Figure S6: Dark Current
    
    myDirectory = 'TDS13_20170130_152928'
    print(myDirectory)
    myCollection = rt.load_summary(myDirectory)
    dc,t = collection_process_dark(myCollection)
    nrows,nwns,ntimes = dc.shape #Get number of dimensions
    plt.figure()
    
    #Preallocate Matrices:
    yint = np.zeros((nrows,nwns))
    slp = np.zeros((nrows,nwns))
    
    for i in range(nrows):
        for j in range(nwns):
            fit = np.polyfit(t[1::],dc[i,j,1::],1)
            yint[i,j] = fit[1]
            slp[i,j] = fit[0]
            #plt.plot(t,dc[i,j,:],color='k',alpha=0.7)
            
            
    ydiff = yint-dc[:,:,0]
    bins = np.arange(-0.0005 , 0.0005, 0.0001)
    bc =   np.arange(-0.00045, 0.00045,0.0001)
    yv,xv = np.histogram(ydiff.flatten(),bins=bins)
                    
    myrow = 20
    mycol = 500
    i=myrow
    j = mycol
    fit = np.polyfit(t[1::],dc[i,j,1::],1)
    
    slprange = (0,0.0006)
    yintrange = (0.005,0.006)
    
    #slprange = (np.nanmin(slp),np.nanmax(slp))
    #yintrange = (np.nanmin(yint),np.nanmax(yint))
    
    #Get number of outliers given slprange
    
    nslp_upper  = len(np.where(slp>slprange[1])[0])
    nslp_lower = len(np.where(slp<slprange[0])[0])
    
    nyint_upper  = len(np.where(yint>yintrange[1])[0])
    nyint_lower = len(np.where(yint<yintrange[0])[0])
    
    
    f,ax=plt.subplots(3,2,figsize=(7,9))
    
    h_rst, = plot_loc(myCollection,myrow,mycol,ax[0,0])
    h_med, = ax[0,0].plot(t,dc[myrow,mycol,:],color='r',alpha=0.7,lw=2)
    h_fit, = ax[0,0].plot(t,np.polyval(fit,t),color='b')
    ax[0,0].set_ylabel('Intensity (A.U.)')
    ax[0,0].set_xlabel('Imaging Time (s)')
    ax[0,0].legend([h_rst,h_med,h_fit],['Replicates','Median','Fit'],loc='lower right')
    ax[0,0].set_xlim([-0.1,65])
    ax[0,0].text(5,0.008,'a.')
    
    slpf = ax[1,0].pcolormesh(slp,vmin=slprange[0],vmax=slprange[1])
    f.colorbar(slpf,ax=ax[1,0])
    ax[1,0].set_xlim([0,1024])
    ax[1,0].set_ylim([0,44])
    ax[1,0].set_ylabel('Slope\nVertical CCD Bin Number')
    ax[1,0].text(100,38,'b.',color='w')
    #plt.colorbar()
    
    ax[1,1].hist(slp.flatten(),range=slprange)
    ax[1,1].set_yscale('log')
    ax[1,1].set_xlim(slprange)
    ax[1,1].set_ylabel('Slope: # Pixels')
    ax[1,1].yaxis.tick_right()
    ax[1,1].yaxis.set_label_position('right')
    ax[1,1].xaxis.set_ticks([0,0.0002,0.0004,0.0006])
    ax[1,1].text(0.00002,10000,'c.')
    ax[1,1].text(slprange[0],20,nslp_lower,color='r')
    ax[1,1].text(slprange[1]-0.00005,20,nslp_upper,color='r')
    
    yintf = ax[2,0].pcolormesh(yint,vmin=yintrange[0],vmax=yintrange[1])
    f.colorbar(yintf,ax=ax[2,0])
    ax[2,0].set_xlabel('Horizontal CCD Bin')
    ax[2,0].set_xlim([0,1024])
    ax[2,0].set_ylim([0,44])
    ax[2,0].set_ylabel('Y intercept\nVertical CCD Bin Number')
    ax[2,0].text(100,38,'d.',color='w')
    
    ax[2,1].hist(yint.flatten(),range=yintrange)
    ax[2,1].set_yscale('log')
    ax[2,1].set_xlim(yintrange)
    ax[2,1].set_ylabel('Y-intercept: # Pixels')
    ax[2,1].set_xlabel('Slope/Y Intercept Magnitude')
    ax[2,1].yaxis.tick_right()
    ax[2,1].yaxis.set_label_position('right')
    ax[2,1].text(0.00505,15000,'e.')
    ax[2,1].xaxis.set_ticks([0.005,0.0054,0.0058])
    ax[2,1].text(yintrange[0],1,nyint_lower,color='r')
    ax[2,1].text(yintrange[1]-0.0001,1,nyint_upper,color='r')
    
    plt.savefig('FigureS4.png')            
            
    
########################################################################################################



def make_FigureS6():
    cm = plt.cm.get_cmap('rainbow')
    
    satdata = np.load("saturations.npy")
    
    satvals = satdata[:,4]
    f,ax = plt.subplots(nrows=1,ncols=1)
    h2 = ax.hist(satvals,bins=np.arange(0,1000,5))
    ax.set_xlabel("Number of Saturated Pixels")
    ax.set_ylabel("Number of spectra")
    
    n,bins,patches=h2
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)
    
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    
    plt.savefig('DH19a_Processing_FigureS6.png')

########################################################################################################
#
#   Generate Figure 3: Background and SRM 2245 Picture
#

#Make Figure 3
#Download Background Files and Figures

#Part 1: Relative Heights

#Load Data
#print "Making Figure2"
make_Figure2()
########################################################################################################
#
#   Generate Figure 4: Burning Example
#
#print "Making Figure 4"
make_Figure4()
########################################################################################################
#
# Generate Figure 6: Fluorescence Removal
#
#print "Making Figure 5" 

make_Figure3()

########################################################################################################
#
# Generate Figure 11: Fluorescence Removal
#
make_Figure11()
########################################################################################################
#
#
#
#
#
#
#   Make Figures 7/8
#   This program was very long because of the peak finding code contained in it. 
#   Thus, it has been moved to a seperate program which we call. 
#
#   Also Makes Figure S7-S13 of Dh19a
#
#


plt.close('all')

exec(open("generatefigures_dh19a_rdat_Fig5-10+FigS10-S12_BB.py").read())



########################################################################################################
#
#
#           make Supplemental Figures
#
#   We don't remake the calibration figure - but data and code available upon request. 
#   We don't remake the dark current figure - but data available on request. The code is make_FigureS4.    

make_FigureS2()
make_FigureS3()
make_FigureS6()


########################################################################################################
#
#   Calculations
#
#
# Test time required for the different calculations on your computer
#
#print "Calculating Durations"
#calculate_Durations()


