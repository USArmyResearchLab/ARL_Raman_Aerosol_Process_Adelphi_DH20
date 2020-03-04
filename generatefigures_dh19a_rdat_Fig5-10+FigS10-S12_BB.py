# Make_Journal_Plots_DoughtyHillAMT2018
'''
This code generates Figures 7,8 of DH19, as well as Figures S7-S13
'''
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rs_tools as rt
import os
import numpy as np
import pandas as pd
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True}) 

                      
                        
def val2ind(x,val):
    #REturns index and value of elelment of vector X closest to val
    print (x)
    print (val)
    
    index = (np.abs(x-val)).argmin()  
    closestval = x[index] 
    return index,closestval    
    
def deriv(myvalue):
# First derivative of vector using 2-point central difference.
#  T. C. O'Haver, 1988.
    a = myvalue[:]
    d = np.zeros(len(a))
    n=len(a)
    d[0]=a[1]-a[0];
    d[n-1]=a[n-1]-a[n-2];
    for j in range(1,n-1):
        d[j]=(a[j+1]-a[j-1])/2.0
    return d

    
def fastsmooth(Y,w,typ=1,ends=0):
# fastbsmooth(Y,w,type,ends) smooths vector Y with smooth
#  of width w. Version 2.0, May 2008.
# The argument "type" determines the smooth type:
#   If typ=1, rectangular (sliding-average or boxcar)
#   If typ=2, triangular (2 passes of sliding-average)
#   If typ=3, pseudo-Gaussian (3 passes of sliding-average)
# The argument "ends" controls how the "ends" of the signal
# (the first w/2 points and the last w/2 points) are handled.
#   If ends=0, the ends are zero.  (In this mode the elapsed
#     time is independent of the smooth width). The fastest.
#   If ends=1, the ends are smoothed with progressively
#     smaller smooths the closer to the end. (In this mode the
#     elapsed time increases with increasing smooth widths).
# fastsmooth(Y,w,type) smooths with ends=0.
# fastsmooth(Y,w) smooths with type=1 and ends=0.
# Example:
# fastsmooth([1 1 1 10 10 10 1 1 1 1],3)= [0 1 4 7 10 7 4 1 1 0]
# fastsmooth([1 1 1 10 10 10 1 1 1 1],3,1,1)= [1 1 4 7 10 7 4 1 1 1]
#  T. C. O'Haver, May, 2008.
    if typ == 1:
        SmoothY=sa(Y,w,ends)
    elif typ == 2:
        SmoothY=sa(sa(Y,w,ends),w,ends)
    elif typ == 3:
        SmoothY=sa(sa(sa(Y,w,ends),w,ends),w,ends)
    else:
        print('Type not defined (should be 1,2, or 3)')
        SmoothY = np.nan(Y.shape)
    return SmoothY
          

def sa(Y,smoothwidth,ends):
    #This is the subroutine that actually does the smoothing
    L=len(Y);
    L_maxind = L-1
    w=np.round(smoothwidth)
    SumPoints=np.nansum(Y[0:w])
    s=np.zeros(L)
    halfw=np.round(w/2.0)

    import pdb
    
    for k in range(0,L-w):
        try:
            s[int(k+halfw-1)]=SumPoints
        except:
            pdb.set_trace()
        SumPoints=SumPoints-Y[k]
        SumPoints=SumPoints+Y[k+w]

    s[int(k+halfw)]=np.nansum(Y[L-w:L])
    SmoothY=np.true_divide(s,w)
    #Taper the ends of the signal if ends=1.
    if ends==1:
        startpoint=int(np.floor((smoothwidth)/2.0))
        SmoothY[0]=(Y[0]+Y[2])/2.0
        SmoothY[L_maxind]=(Y[L_maxind-1]+Y[L_maxind])/2.0
        for k in range(1,startpoint+1):
            SmoothY[k]=np.nanmean(Y[0:(2*k+1)])
            SmoothY[L_maxind-k]=np.nanmean(Y[L_maxind-2*k:L_maxind+1 ])    
    return SmoothY


def rmnan(input_value):
# Removes NaNs and Infs from vectors, replacing with PREVIOUS real numbers.
# 
#    In [51]: orig = [1, 2, 3, 4, np.nan, 6, np.inf, 7, 9]
#    In [52]:  rmnan(orig)
#    Out[52]: [1, 2, 3, 4, 4, 6, 6, 7, 9]


    a = input_value[:]
    la=len(a);
    if np.isnan(a[0]) | np.isinf(a[0]):
        a[0]=0
    for point in range(0,la):
        if np.isnan(a[point]) | np.isinf(a[point]):
            a[point]=a[point-1]
    return a
    

def gaussfit(myx,myy):
# Converts y-axis to a log scale, fits a parabola
# (quadratic) to the (x,ln(y)) data, then calculates
# the position, width, and height of the
# Gaussian from the three coefficients of the
# quadratic fit.  This is accurate only if the data have
# no baseline offset (that is, trends to zero far off the
# peak) and if there are no zeros or negative values in y.
#
# Example 1: Simplest Gaussian data set
# [Height, Position, Width]=gaussfit([1 2 3],[1 2 1]) 
#    returns Height = 2, Position = 2, Width = 2
#
# Example 2: best fit to synthetic noisy Gaussian
# x=50:150;y=100.*gaussian(x,100,100)+10.*randn(size(x));
# [Height,Position,Width]=gaussfit(x,y) 
#   returns [Height,Position,Width] clustered around 100,100,100.
#
# Example 3: plots data set as points and best-fit Gaussian as line
# x=[1 2 3 4 5];y=[1 2 2.5 2 1];
# [Height,Position,Width]=gaussfit(x,y);
# plot(x,y,'o',linspace(0,8),Height.*gaussian(linspace(0,8),Position,Width))
# Copyright (c) 2012, Thomas C. O'Haver
    
# Use  numpy.ndarray.flatten (method) ndarray.flatten(order='C')
#      Returns a copy of the array collapsed into one dimension.
    
    try:
        y = myy.flatten() #.flatten()
        x = myx.flatten() #.flatten()
    except:
        print('exception at 151 - - did not flatten')
        y = myy
        x = myx
    
    minimum_value = np.nanmin(y)
    if np.nanmin(y) <=0:
        y = y + np.abs(minimum_value) + 0.00001 
    
    maxy=np.nanmax(y)
    for p in range(0,len(y)):
        if y[p]<np.divide(maxy,100):
            y[p]=np.divide(maxy,100)

    z=np.log(y);
    try:
        coef=np.polyfit(x,z,2)
    except:
        pdb.set_trace()
        
    a=coef[2]
    b=coef[1]
    c=coef[0]
    #print(a,b,c,'a b c')
    Height=np.exp(a-c*(b/(2*c))**2)
    Position=np.divide(-b,(2*c))
    Width=np.divide(2.35482,(np.sqrt(2)*np.sqrt(-c)))
    return Height,Position,Width


def curve_gaus(x,a,x0,sigma):
    return a*np.exp(np.true_divide(-(x-x0)**2,(2*(sigma**2))))               

def get_spectrum(myDT,myX,myY,mymeta,mydata):
        myspecind= int(mymeta[(mymeta['DateTime'] == myDT) & (mymeta['x']==myX) & (mymeta['y'] == myY)].index.values)
        myspec = mydata[myspecind]
        return myspec,myspecind

def make_Fig_subplts_labels(iplt,ii,ptot,nameload, thr1,thr2,nxxmin,nxxmax,nzzmin,nzzmax,xlimmax,llow,lhi,px1,px2,px3,py1,py2,py3,zcrit,amthr,dxdthr,ilab,pklst,pkmat,fpks,fh,rdat_1b,rmta_1b,rdat_2bf0,rdat_3bf6,rflu_3bf6,filef,fpkdg,mxnpk):  
    import pandas as pd
    import matplotlib.dates as dates                
    import matplotlib.pyplot as plt
    import scipy
    from scipy.signal import savgol_filter,medfilt
    from scipy import optimize
    from scipy.optimize import curve_fit
    import matplotlib.patheffects as path_effects
    rt.plot_format(fs=10)    
    xrangemax = 628
    if ( xlimmax > 1800):
        xticks = [500,1000,1500,2000,2500,3000]
    if ( 751 < xlimmax < 1800):
        xticks = [500,750,1000,1250,1500]
        xrangemax = int(xrangemax * xlimmax / 3180)
    if ( xlimmax < 751):
        xticks = [500,550,600,650,700]
        xrangemax = int(xrangemax * xlimmax / 3180)
    x_values = rt.get_rebs_calibration(cal_file='rn13_cal_20160825_paper.txt',cal_type='fit')
    
    nameloadold = nameload
    nameload = nameloadold[8:26]
    popt = np.zeros(5)   
    
    nameloadtxt = nameload[3:7] + '-' + nameload[7:9] + '-' + nameload[9:11]+ ' ' + nameload[12:14] + ':' + nameload[14:16] + ':' + nameload[16:18]
    towrite1  = nameloadtxt
    ## Need if RN13 then lendc = 44???
    
    lendc=44
    directory = os.path.join("20160825r13",nameload)
    print(directory,'directory')

    for nxx in range(nxxmin,nxxmax):
        nzzmaxtemp = nzzmax        
        if ( lendc < nzzmaxtemp): nzzmaxtemp = lendc -1
        for nzz in range(nzzmin,nzzmaxtemp):                   
            figname = nameload + '_' + str(nxx) + '_' + str(nzz)   
            print(nxx,nzz,figname,'  nxx  nzz figname')

            
            #def get_spectrum(myDT,myX,myY,mymeta,mydata):
            #):#rdat_1b,rmta_1b,rdat_2bf0,rmta_2b0,rdat_3bf6,rmta_3bf6,rflu_3bf6):
            try:
                myspec,myspecind = get_spectrum(nameloadtxt,nzz*2,nxx,rmta_1b,rdat_1b)
            except:
                import pdb
                pdb.set_trace()
            #myspecind= int(mymeta[(mymeta['DateTime'] == myDT) & (mymeta['x']==myX) & (mymeta['y'] == myY)].index.values)
            #results = rmta_1b[(rmta_1b['DateTime'] == nameloadtxt) & (rmta_1b['x']==nxx*2) & (rmta_1b['y'] == nzz)].index.values
            specdiff = rdat_3bf6[myspecind]
            myfluo   = rflu_3bf6[myspecind]
            
            #myfluo_1 = rdat_2bf0[myspecind]
            specdiff_1 = rdat_2bf0[myspecind]
            specnotDG=specdiff_1
            

            '''            
            #print(' ')
            myspec = dc[nxx,:,nzz]
            ################################################################################################
            maxmyspec = np.nanmax(myspec)
            print(maxmyspec,' maxmyspec')

            myfluo=fc_f6[nxx,:,nzz] #1e6
            myfluo_1=fc_f0[nxx,:,nzz] #1e6
            #Remove fluorescence
            specdiff = dc_f6[nxx,:,nzz]
            specdiff_1 = dc_f0[nxx,:,nzz]      
            specnotDG = dc_f0[nxx,:,nzz]  
            '''
                      
            #initially include some numbers less than 449 so the second derivatives and smoothing can be calculated
            i0low = 104
            myspec[0:i0low] = 0
            specdiff[0:i0low] = 0
            specdiff_1[0:i0low] = 0
            specnotDG[0:i0low] = 0   
            myspec[i0low:119] = myspec[119]
            specdiff[i0low:119] = specdiff[119]
            specdiff_1[i0low:119] = specdiff_1[119]
            specnotDG[i0low:119] = specnotDG[119]
            
            maxmyspec = np.nanmax(myspec)            
            maxv = np.nanmax(specdiff)
            maxspecdiff = np.nanmax(specdiff)
            maxx = x_values[specdiff==maxv][0]
            ymainv = np.zeros(len(x_values))
            ymainv[:] = maxv    
            
            maxv_1 = np.nanmax(specdiff_1)
            maxfluo_1 = maxv_1
            maxx_1 = x_values[specdiff_1==maxv_1][0]
            ymainv_1 = np.zeros(len(x_values))
            ymainv_1[:] = maxv_1
            
            maxnotDG = np.nanmax(specnotDG)
            #Code dies here
            maxxnotDG = x_values[specnotDG==maxnotDG][0]
            avgspecnotDG = np.mean(specnotDG)  
            
            xv = x_values
            yv = myspec     
            lenxv = len(xv)
            maxxv = max(xv)
            num_spectra = lenxv
            av = yv            
            
            sumDG = 0.
            isumDG = 0    
            sumnotDG = 0.
            isumnotDG = 0
            Dpeak = 0
            Gpeak = 0
            iDG = 0
            sumCH = 0.
            isumCH = 0    
            sumnotCH = 0.
            isumnotCH = 0
            iCH = 0
            for i in range(120,1023):
                #if (abs(x_values[i]-1320)<35) or (abs(x_values[i]-1583)<27): 
                #    specnotCH[i]=0
                if (abs(x_values[i]-1320)<45) or (abs(x_values[i]-1583)<27): 
                    sumDG = sumDG + specdiff[i]
                    isumDG = isumDG + 1
                if (abs(x_values[i]-1120)<40) or (abs(x_values[i]-1490)<30) or (abs(x_values[i]-1640)<30): 
                    sumnotDG = sumnotDG + specdiff[i]
                    isumnotDG = isumnotDG + 1                   

                if abs(x_values[i]-1335) < 40: 
                    if specdiff[i] > Dpeak: Dpeak = specdiff[i]      
                if abs(x_values[i]-1590) < 25: 
                    if specdiff[i] > Gpeak: Gpeak = specdiff[i]
                
                if abs(x_values[i]-2940) < 160 : 
                    sumCH = sumCH + specdiff[i]
                    isumCH = isumCH + 1
                if ( x_values[i] < 2780) or (x_values[i] > 3100) : 
                    sumnotCH = sumnotCH + specdiff[i]
                    isumnotCH = isumnotCH + 1
                    
            avgDG    = sumDG/isumDG
            avgnotDG = sumnotDG/isumnotDG
            ratioDGtoother = avgDG/avgnotDG
            if ratioDGtoother > 1.4 : iDG = 1     
            GtoD = Gpeak / Dpeak
            DtoG = Dpeak / Gpeak
            towrite1=str(nameload)+' '+str(nxx)+' '+str(nzz)+' '+str(ratioDGtoother)+' '+str(DtoG)+' '+str(GtoD)+' '+str(Dpeak)+' '+str(Gpeak)+' DG/not-DG D/G G/D DGpk Gpk \n'
            fpkDG.write(towrite1)     

            avgCH    = sumCH/isumCH
            avgnotCH = sumnotCH/isumnotCH
            ratioCH = avgCH/avgnotCH
            if ratioCH > 1.15 : iCH = 1                       
              
            #import pdb
            #pdb.set_trace()
            
            ## Set av for myspec, specdiff or specdiff_1
            # temp test change
            #av = specdiff_1   
            av = myspec
            # end temp test change
            
            #print ('maxxv',maxxv)      
            maxy = max(av)
            x_at_maxy = xv[av==maxy]
            #print (x_at_maxy,maxy,'  x_at_maxy   maxy  ')
            minx = xv[0]
            maxx = max(xv)        
            miny = np.min(yv)
            xlimmin = 449
            

            #print(nxx,nzz,figname,'  nxx  nzz figname')
            towrite1='   \n'
            filef.write(towrite1)
            towrite1=str(nxx)+' '+str(nzz)+' '+str(figname)+'  \n'
            filef.write(towrite1)
            #for ii in range(399,417):
            #for ii in range(118,160):
            for ii in range(800,1000):  
                towrite1=str(ii)+'     '+str(av[ii])+'     '+str("{0:0.3f}".format(xv[ii]))+'  \n'
                filef.write(towrite1)
                ## print(ii,av[ii],xv[ii],'ii av xv')
            #print(maxnotDG,thr2,maxmyspec,maxv,maxv_1,'maxnotDG thr2 maxmyspec maxv max_1')
            fname = figname[8:18]    
            fnametime = fname[4:6]+':'+fname[6:8]+':'+fname[8:10]            
            
            nspk = len(pklst)
            ipks = np.zeros(nspk)                        

            ##################################################################
            ############  FIND PEAKS TO LABEL: Center Wavenumber and Height ##########################################
            ## Note: Problem with z_scores as found in FindpeaksG: these are relative to an overall average intensity
            
            # 1. Find second derivatives for spectra smoothed for three different widths (not optimized for widths used)
            # 2. Use findpeaksG to select spectra based on zero crossings of smoothed spectra 
            # 3. If peaks exceed thresholds (slope, 2nd derivative, etc.) use curve_fit optimize to minimize least sq.
            # 4. Then label peaks on figs.

            av[0:i0low] = 0   
            av[i0low:119] = av[119]   
            npk = 0
            #print (maxnotDG,'maxnotDG      thr2',thr2,npk,'npk')
            ifindpeaks = 1    
            if ( ( maxnotDG > thr2) and ( ifindpeaks == 1 ) ):              
                ### >>>>>>> FIND PEAK LOCATIONS < < < < < < < < < < < < < < < < < < < < 
                print ('  - - - - -  FIND PEAK LOCATIONS  - - - -- - ',nxx,nzz,figname)
                #* Calculate z-scores   
                #print (lenxv,' lenxv')
                #window_length = 11
                # temp test change
                window_length = 5
                window_length = 7
                #window_length = 21
                #window_length = 21
                # end temp test change
                av_smoothed = savgol_filter(av,window_length,polyorder=4)
                #print ('       av_smoothed     ********************************************************************')
                #z_score = scipy.stats.zscore(av_smoothed)            
                maxyv = np.max(av_smoothed)
                xv_at_maxyv = xv[av_smoothed==maxyv]        
                d2 = savgol_filter(av,window_length,deriv=2,polyorder=4)
                d2p = -d2
                d2p = d2p.clip(min=0) * window_length**1.5
                z_score_d2p = scipy.stats.zscore(d2p)        
                nzero = np.count_nonzero(av_smoothed)
                if nzero >= 1:
                    for j in range(window_length):
                        d2p[j] = 0.
                        z_score_d2p[j] = 0
                
                window_lengtho = 19
                av_smoothedo = savgol_filter(av,window_length=window_lengtho,polyorder=4)
                maxyv0 = np.max(av_smoothedo)
                xv_at_maxyv0 = xv[av_smoothedo==maxyv]                 
                d2o = savgol_filter(av,window_length=window_lengtho,deriv=2,polyorder=4)
                d2op = -d2o
                d2op = d2op.clip(min=0) * window_lengtho**1.5
                z_score_d2op = scipy.stats.zscore(d2op) 
                nzero = np.count_nonzero(av_smoothedo)
                if nzero >= 1:
                    for j in range(window_lengtho):
                        d2o[j] = 0.
                        z_score_d2op[j] = 0
                    
                window_lengthw = 37      
                av_smoothedw = savgol_filter(av,window_length=window_lengthw,polyorder=4) 
                d2w = savgol_filter(av,window_length=window_lengthw,deriv=2,polyorder=4)
                d2wp = -d2w
                d2wp  = d2wp.clip(min=0) * window_lengthw**1.5
                z_score_d2wp = scipy.stats.zscore(d2wp)      
                nzero = np.count_nonzero(av_smoothedw)
                if nzero >= 1:
                    for j in range(window_lengthw):
                        d2wp[j] = 0.
                        z_score_d2wp[j] = 0
                
                dxd = [0.] * lenxv        
                for j in range(lenxv):
                    if abs(xv[j]-xlimmax) < window_length/2.  : d2p[j]  = 0.
                    if abs(xv[j]-xlimmax) < window_lengtho/2. : d2op[j] = 0.
                    if abs(xv[j]-xlimmax) < window_lengthw/2. : d2wp[j] = 0.
                    #dxd[j] = 0.2* max(d2p[j]*window_length**2,d2op[j]*window_lengtho**2,d2wp[j]*window_lengthw**2) 
                    dxd[j] = max(d2p[j],d2op[j],d2wp[j]) 
                    #dxd[j] = d2p[j]
                    
                av_d2p = np.average(d2p)
                av_d2wp = np.average(d2wp)
                av_d2op = np.average(d2op)                
                #dxd = d2p + d2op + d2wp        
                lendxd = len(dxd) 
                maxdxd = max(dxd)
                
                ## print (av_d2p,av_d2wp,av_d2op,' av_d2p  av_d2wp d2op')
                z_scored2p   = scipy.stats.zscore(d2p)          
                z_scored2wp  = scipy.stats.zscore(d2wp)          
                z_scorewd2op = scipy.stats.zscore(d2op)          
                z_scoredxd   = scipy.stats.zscore(dxd)          
                z_scorew = scipy.stats.zscore(av_smoothedw)          
                offset = 0.07*maxy         
                                      
                ##### completed this figure
                                 
                z_scorew = scipy.stats.zscore(av_smoothedw)          
                #line7, = ax[2].plot(xv,offset*3+5*d2op,color="blue",lw=.7)
                maxd2wp = max(d2wp)
                d2wp = d2wp / maxd2wp  
                #print('- FIND PEAKS - - USE FINDPEAKSG. THEN SELECT VALID PEAKS, I.E., PEAKS WITH PARAMETERS ABOVE THRESHOLDS FOR DXD, AMPL, etc - ')

                ##  Now that smoothing has been done and z-scores calculated, zero all y values at wavenumbers smaller than 449 cm-1
                myspec[0:i0low] = 0
                specdiff[0:i0low] = 0
                specdiff_1[0:i0low] = 0
                specnotDG[0:i0low] = 0   
                av_smoothed[0:i0low] = 0
                maxmyspec = np.nanmax(myspec)  
                minx = x_values[120]
                            
                amp_thrsh = np.nanpercentile(av,96)                             
                slp_thrsh = 0.00005
                amp_thrsh = 0.00005
                smooth_wdth = 1
                pkgrp = 5

                xmin = 459
                xmax = 3180

                Peak_Result = findpeaksG3(xv,av_smoothed,xmin,xmax,slp_thrsh,amp_thrsh,smooth_wdth,pkgrp)
                lenpeak = len(Peak_Result[:,2]) 
                #print('>> Aft findpeaksG        lenpeak=',lenpeak,'          smooth_wdth = ',smooth_wdth)
                
                #print('Peak_Result',Peak_Result)
#                smooth_wdth = 7                
#                av_smoothed = savgol_filter(av,smooth_wdth,polyorder=4)
#                Peak_Result = findpeaksG3(xv,av_smoothed,xmin,xmax,slp_thrsh,amp_thrsh,smooth_wdth,pkgrp)
#                lenpeak = len(Peak_Result[:,2]) 
#                print('>> Aft findpeaksG lenpeak',lenpeak,'   smooth_wdth = ',smooth_wdth)               
#                
                
                #print('Now print j, Peak_Result[j,1] ,Peak_Result[j,2]')
                for j in range(lenpeak):
                    #print(j,Peak_Result[j,1],Peak_Result[j,2],'  j,Peak_Result[j,1],Peak_Result[j,2]')
                    if (Peak_Result[j,1]) < xmin:
                        Peak_Result[j,1] = xmin
                        Peak_Result[j,2] = 0.0
                        

                #print('Peak_Result',Peak_Result)


                peak_value_strings = ['%.2f' % value for value in Peak_Result[:,1]]                      
                number_peaks = len(peak_value_strings)
                #qt = sorted(Peak_Result, key=itemgetter(1)) 
                max_number_peaks = 280
                jn = [0] * max_number_peaks
                zs = [0.] * max_number_peaks
                px = [0.] * max_number_peaks
                py = [0.] * max_number_peaks         # peak height in specdiff_1 - -  best for finding peaks 
                pyraw = [0.] * max_number_peaks      # peak height in raw        - -  best for labeling peaks in myspec 
                dxdj = [0.] * max_number_peaks
                iokj = [0] * max_number_peaks
                xwnj = [0.] * max_number_peaks
                heightpj = [0] * max_number_peaks
                dxdxr = [0] * max_number_peaks
                lsq = [0] * max_number_peaks
                heightpsj = [0] * max_number_peaks
                xwnpsj = [0] * max_number_peaks
                pxsj = [0] * max_number_peaks
                
                npk = 0            
        
                ## Select peaks found with FindpeaksG and having 2nd derivs exceeding thresholds     ######################################
                ## Select peaks found with FindpeaksG and having 2nd derivs exceeding thresholds
                
                for j in range(0,len(peak_value_strings)-1):
                    PR = Peak_Result[j,1]
                    PRY = Peak_Result[j,2]
                    #if ( ( 449 <= PR <= 1290.) or (1340 <= PR <= 1560) or (1600<=PR<=1740) or (2715<=PR)):      
                    if ( ( 459 <= PR <= 1740.) or (2715<=PR)):           
                        goodloc=rt.find_nearest(xv,Peak_Result[j,1])
                        id2_good = 0
                        z_score_good = 0
                        d2p_goodloc = 0
                        if (dxd[goodloc] > dxdthr):
                            if ( (d2p[goodloc] > d2op[goodloc] ) and (d2p[goodloc] > d2wp[goodloc] ) ):  id2_good = 1
                            if ( (d2op[goodloc] > d2p[goodloc] ) and (d2op[goodloc] > d2wp[goodloc] ) ): id2_good = 2
                            if ( (d2wp[goodloc] > d2p[goodloc] ) and (d2wp[goodloc] > d2op[goodloc] ) ): id2_good = 3    
                            z_scoregood = 0    
                            if ( id2_good == 1): 
                                z_score_good = z_score_d2p[goodloc]
                                d2p_goodloc = d2p[goodloc]
                                widthpix = ( window_length + 1) / 2
                            if ( id2_good == 2): 
                                z_score_good = z_score_d2op[goodloc]
                                d2p_goodloc = d2op[goodloc]                                
                                widthpix = ( window_lengtho + 1) / 2
                            if ( id2_good == 3):
                                z_score_good = z_score_d2wp[goodloc]
                                d2p_goodloc = d2wp[goodloc]
                                widthpix = ( window_lengthw + 1) / 2
                            #print(j,PR,goodloc,z_score_good,d2p_goodloc,widthpix,' j,PX,goodloc,z_score_good,d2p_goodloc,widthpix',nxx,nzz,figname)
                            #if id2_good != 1 :print(id2_good,'id2_good',widthpix,' widthpix')
                            
                            towrite1=str(nxx)+' '+str(nzz)+' '+str(j)+' '+str(np.rint(PR))+' '+str(np.rint(10000*PRY))+' '+str(goodloc)+' '+str(np.rint(1000*z_score_good))+' '+str(np.rint(1000*d2p_goodloc))+' '+str(widthpix)+' \n'
                            fpks.write(towrite1)
                            #print(towrite1)
                        if (z_score_good > zcrit) and (Peak_Result[j,2] > amthr):
                            #print(' ')
                            #print (j,goodloc,z_score_good,Peak_Result[j,2],d2p_goodloc,Peak_Result[j,1],'j goodloc=jn[npk],z_score,PRj2,d2p,PRj1')
                            dxdj[npk] = d2p_goodloc
                            zs[npk] = z_score_good
                            jn[npk] = goodloc
                            px[npk] = Peak_Result[j,1]
                            py[npk] = Peak_Result[j,2]
                            dxdxr[npk] = dxdj[npk] * np.sqrt(z_score_good)
                            if dxdxr[npk] > dxdthr * 100.:
                                iokj[npk] = 1                                   
                            
                                indexpxraw = min(range(len(xv)), key=lambda ii: abs(xv[ii]-px[npk]))
                                pyraw[npk] = myspec[indexpxraw]
                                #print (indexpxraw,xv[indexpxraw],dxdxr[npk],'   ',pyraw[npk],py[npk],px[npk],'indexpxraw xv[indexpxraw] dxdxr[npk] pxraw[npk] py[npk]')                          
                                                            
                                towrite1='                             '+str(Peak_Result[j,1])+' '+str(Peak_Result[j,2])+'Passed z_score and Py test  \n'
                                fpks.write(towrite1)
                                
                                ## OPTIMIZE CURVE FITS ####################################
                                ## OPTIMIZE CURVE FITS ####################################                              
                                #errfunc1 = lambda p, x, y: (one_gaussian(x, *p) - y)**2
                                #errfunc2 = lambda p, x, y: (two_gaussians(x, *p) - y)**2
                                #errfunc3 = lambda p, x, y: (three_gaussians(x, *p) - y)**2
                                
                                # widthpix is the half-width in pixels rounded up
                                # width is the half-width in oscillations / um
                                # want height to be height above background
                                jlo = int(jn[npk]+1-widthpix*.8)
                                jhi = int(jn[npk]+1+widthpix*.8)
                                if jhi >= 1024: jhi = 1024
                                minpynpk = 99999.
                                for jtmp in range (jlo ,jhi): 
                                    if av_smoothed[jtmp] <= minpynpk: minpynpk = av_smoothed[jtmp]
                                                               
                                height = py[npk] - minpynpk
                                offset = minpynpk  
                                center = px[npk]/10000.
                                width = widthpix * 1. / 10000.
                                horig = height
                                corig = center
                                worig = width
                                oorig = offset
                                dummy = 0.
                                guess1 = [height,center,width,offset,dummy]
                                xvvv = [0.] * len(x_values)
                                xvvv = x_values[jlo:jhi]
                                lennn = len(xvvv)
                                onegfin = one_gaussian(x_values[jlo:jhi],height,center,width,offset,dummy)                           
                                residual = onegfin - myspec[jlo:jhi]
                                lenspd = len(myspec)
                                if jhi > lenspd-1: jhi = lenspd-1                            
                                
                                try:
                                    popt, pcov = curve_fit(one_gaussian_slope,x_values[jlo:jhi]/10000.,av_smoothed[jlo:jhi],p0=guess1)
                                    rt_error = 0
                                except RuntimeError:
                                    #print("Error - curve_fit failed fitting with slope")
                                    popt[0] = 0.
                                    popt[1] = 0.0003
                                    popt[2] = 20.
                                    popt[3] = 0.        
                                    popt[4] = 0.                   
                                    rt_error = 1
    
                                if rt_error == 1:
                                    height = py[npk]       
                                    center = px[npk]/10000.
                                    width = widthpix * 5. / 10000.
                                    offset = 0.00001       
                                    horig = height
                                    corig = center
                                    worig = width*4.
                                    oorig = offset
                                    dummy = 0.
                                    guess1 = [height,center,width,offset,dummy]
                                    jlo = int(jn[npk]+1-widthpix*.8)
                                    jhi = int(jn[npk]+1+widthpix*.8)
                                    xvvv = [0.] * len(x_values)
                                    xvvv = x_values[jlo:jhi]
                                    lennn = len(xvvv)
                                    onegfin = one_gaussian(x_values[jlo:jhi],height,center,width,offset,dummy)                           
                                    residual = onegfin - myspec[jlo:jhi]
                                    lenspd = len(myspec)
                                    if jhi > lenspd-1: jhi = lenspd-1       
                                    try:
                                        popt, pcov = curve_fit(one_gaussian_slope,x_values[jlo:jhi]/10000.,av_smoothed[jlo:jhi],p0=guess1)
                                        rt_error = 0
                                    except RuntimeError:
                                        #print("Error - curve_fit failed fitting with slope")
                                        popt[0] = 0.
                                        popt[1] = 0.0003
                                        popt[2] = 20.
                                        popt[3] = 0.        
                                        popt[4] = 0.                   
                                        rt_error = 1
    
                                if rt_error == 0:                                
                                    height = popt[0]
                                    center = popt[1]
                                    width = popt[2]
                                    offset = popt[3]
                                    px[npk] = center * 10000.
                                    py[npk] = height
                                    slope = popt[4]      
                                    
                                    heightthr = 0.015 * maxmyspec  
                                    #print(px[npk],height,heightthr,'px height heighthr   after fit')
                                    dxdxr[npk] = 0.
                                    if height > heightthr :
                                        dxdxr[npk] = dxdj[npk] * np.sqrt(z_score_good) * height / maxmyspec
                                        dxdxr[npk] = np.sqrt(dxdxr[npk])
                                        #print(center*10000,npk,dxdxr[npk],dxdthr,height,heightthr,'cent_wn npk,dxdthr[npk],dxdthr,height,heightthr')
                                        if dxdxr[npk] >= dxdthr:          # Increment npk only if height found > heightthr AND dxdr[npk] >= dxdthr 
                                            npk = npk + 1                                
                       
                #print(npk,'npk')                                                                       

				## Remove any peaks with px < xmin
                for j in range(0,npk-1):
                    ## print(j,px[j],py[j],dxdxr[j],'j,px,py,dxdxr  - before sorting 1')
                    if px[j] < xmin: 
                        dxdxr[j] = 0.
                        
                ## print('  ')                                        
#                for j in range(0,npk-1):
#                    ##  print(j,px[j],py[j],dxdxr[j],'j,px,py,dxdxr  - before sorting 2')
                #print('  ')                        
                for j in range(0,npk-1):
                    for jk in range(0,npk-1):
                        if jk != j:
                            if abs(px[j]-px[jk]) < 3.:
                                if dxdxr[j] > dxdxr[jk]: dxdxr[jk] = 0.
                                if dxdxr[j] < dxdxr[jk]: dxdxr[j] = 0.      
                                #npk = npk - 1
                ## print('  ')
#                for j in range(0,npk-1):
#                    ## print(j,px[j],py[j],dxdxr[j],'j,px,py,dxdxr  - before sorting 3')                  

                #print(npk,' = npk aft clean up 1')                
                for j in range(0,npk-1):
                    for jk in range(0,npk-1):
                        if jk != j:
                            if abs(px[j]-px[jk]) < 3.:
                                if dxdxr[j] > dxdxr[jk]: dxdxr[jk] = 0.
                                if dxdxr[j] < dxdxr[jk]: dxdxr[j] = 0.      
                                #npk = npk - 1          
                 #print('  ')                
#                for j in range(0,npk-1):
#                    ## print(j,px[j],py[j],dxdxr[j],'j,px,py,dxdxr  - immediately before sorting')
                
                towrite1=str(npk)+'    '+str(px[j])+' '+str(py[j])+' \n'
                fpks.write(towrite1)                      
                
                print(npk,' = npk  before sorting ')              
                
                if npk >= 1:
                    isort = [idxd[0] for idxd in sorted(enumerate(dxdxr), reverse=True, key=lambda iz:iz[1])]    
                    for kt in range(0,npk-1):
                        j = isort[kt]
                        #print(kt,j,px[j],py[j],dxdxr[j],'kt,j,px,py,dxdxr after sorting')
                        #print(' ')     
                        towrite1=str(kt)+' '+str(j)+'  '+str(px[j])+'    '+str(py[j])+'   '+str(dxdxr[j])+' after sorting \n'
                        fpks.write(towrite1)                     
                        
                        
                        
                ## Quick/simple attempt to avoid too many peaks to plot 
                    npkplot = 0
                    ratdx = 0.001
                    for j in range(1,npk-1):
                        ratiodx = dxdxr[isort[j]] / dxdxr[isort[0]]
                        if ratiodx >= ratdx:
                            npkplot = j
                                                
                        #print(kt,j,px[j],py[j],dxdxr[j],'kt,j,px,py,dxdxr   After sorting')
                        #print(' ')     
                        towrite1=str(kt)+' '+str(j)+'  '+str(px[j])+'    '+str(py[j])+'   '+str(dxdxr[j])+'  \n'
                        fpks.write(towrite1)


#                    if npkplot >= 8: 
#                        if (dxdxr[isort[7]]/dxdxr[isort[0]]) <= 0.01: npkplot = 7
#                    if npkplot >= 7:
#                        if (dxdxr[isort[6]]/dxdxr[isort[0]]) <= 0.01: npkplot = 6                   
#                    if npkplot >= 6:
#                        if dxdxr[isort[5]]/dxdxr[isort[0]] <= 0.01: npkplot = 5  
#                    if npkplot >= 5:
#                        if dxdxr[isort[4]]/dxdxr[isort[0]] <= 0.01: npkplot = 4      
#                    #if npkplot >= 4:
#                    #    if dxdxr[isort[3]]/dxdxr[isort[0]] <= 0.01: npkplot = 3                                  
#                    if npkplot <= 2 : npkplot = 3
                    if npkplot > 8: npkplot = 8		
                                 # Set iokj[] to zero (so do not plot that jth peak) for some of the peaks which have dxdxr derivatives below a threshold
                # and are too close to another peak to plot easily.
                minspacesmall = 15
                minspacelarge = 70
        
                dxdxchthr = 0.001
                if npk > 10: npk = 10
                for kt in range(2,npk-1):
                    j = isort[kt]
                    if (( kt > 2 ) and ( dxdxr[j] < dxdxchthr ) ): iokj[j]=0
                    if kt <= 3: minspace = minspacesmall
                    if kt > 3 : minspace = minspacelarge
                        #print (j,px[j],dxdxr[j],py[j],iokj[j],'j px dxdxr py iokj[j]')
                    if iokj[j] ==1:
                        for kp in range(0,npk):
                            jj = isort[kp]
                            if iokj[jj] == 1:
                                if j != jj:
                                    #print (px[j],px[jj],j,jj,iokj[j],iokj[jj],'px px - -  going to compare now')
                                    if abs(px[j]-px[jj]) < minspace: 
                                        #print (j,jj,dxdxr[j],dxdxr[jj],' j,jj,dxdxr[j],dxdxr[jj]')
                                        if ( (py[jj]/maxy)<0.6) and ((py[j]/maxy)<0.6):
                                            if dxdxr[jj] < dxdxr[j]: 
                                                #print (px[j],px[jj],j,jj,'px px setting iokj[jj] to 0')
                                                iokj[jj] = 0
                                            if dxdxr[j]  < dxdxr[jj]: 
                                                #print (px[j],px[jj],j,jj,'px px setting iokj[j] to 0')
                                                iokj[j] = 0
                
                                        if (dxdxr[jj]< dxdthr*1.5) and (dxdxr[j]< dxdthr*1.5):
                                            if dxdxr[jj] < dxdxr[j]: 
                                                #print (px[j],px[jj],j,jj,'px px setting iokj[jj] to 0')
                                                iokj[jj] = 0
                                            if dxdxr[j]  < dxdxr[jj]: 
                                                #print (px[j],px[jj],j,jj,'px px setting iokj[j] to 0')
                                                iokj[j] = 0
                ptot = ptot + 1
                if npk >= 1:                    
                    fpks.write('   \n')
                for kt in range(0,npk-1):
                    j = isort[kt]                    
                    #print (j,px[j],py[j],dxdxr[j],iokj[j],'j px py dxdxr py iokj[j]',py[j]/maxv,'py/maxv')  
                    towrite1=str(nxx)+' '+str(nzz)+' '+str(kt)+' '+str(j)+' '+str(np.rint(px[j]))+' '+str(np.rint(10000*py[j]))+'   '+str(iokj[j])+'   '+str(dxdxr[j])+' '+str(dxdxr[j])+' '+str(ptot)+nameload+' \n'
                    fpks.write(towrite1)
                    
                    if kt == 0:
                        j = isort[kt]        
                        for jspk in range(0,nspk-1):
                            ipks[jspk] = 0
                            if abs(px[j]-pklst[jspk]) < 2.5:
                                myfh = fh[jspk]
                                myfh.write(towrite1)
                                ipks[jspk] = 1
                                
#                        if iDG ==1:
#                            fpkDG.write(towrite1)
                #pxsw = sorted(Peak_Resultw, key=itemgetter(1))
                #pxs = sorted(Peak_Result, key=itemgetter(1))
                #[izsw[0] for izsw in sorted(enumerate(zsw), reverse=True, key=lambda izw:izw[1])]    
                #plt.scatter(Peak_Result[:,1],Peak_Result[:,2])                           
            
            ###############################################################################################################################
            ############### Have now completed peak finding and selecting the ones with both dxdxr > dxdxsqthr and not too close to another peak
            
            ## -----------------------------------------------------------------------------------------------------------------------------
            ########   MAKE THE PLOTS WITH PEAKS LABELED   **************************************************************************
            
            #print(maxmyspec,npk,'maxmyspec  npk')
            if (maxmyspec > thr1) and ( npk > 0) :
                #print (maxmyspec,nxx,nzz,'= maxmyspec exceeds threshold')
                #f, ax = plt.subplots(2, sharex=True,figsize=(6.,6.))
                #plt.figure(1)
                #plt.tight_layout(h_pad=-.8,w_pad=3.0)
                #myColors = rt.nicePalette()
#                ax[0].plot(x_values,myspec,color='k')
#                
#                ax[0].axis([450,3175,0,maxmyspec*1.05])
                maxmyspec2 = np.nanmax(myspec) 
                #print(maxmyspec2,maxmyspec,' maxmyspec2 maxmyspec aft plot myspec')
#                ax[0].plot(x_values,myfluo,color=myColors[1])
               
#                ax[0].plot(x_values,myfluo_1,color=myColors[4])
#                
                window_length = 29
                av_smoothed_myspec = savgol_filter(myspec,window_length,polyorder=4)
#                ax[0].plot(x_values,av_smoothed_myspec,color='turquoise',lw=1.)                   
#                ax[0].fill_between(x_values,myfluo,facecolor=myColors[1], alpha=0.2,edgecolor="white")
#                ax[0].fill_between(x_values,myfluo_1,facecolor=myColors[4], alpha=0.2,edgecolor="white")
#                ax[0].plot(x_values,specdiff,color='blue',lw=0.8)
#                ax[0].plot(x_values,specdiff_1,color='purple',lw=0.7)
#                
#                #Make second part of plot showing the removed fluorescence and Rmax                
#                ax[1].plot(x_values,specdiff,color='black',lw=0.8)
#                ax[1].plot(x_values,specdiff_1*0.99*maxspecdiff/maxfluo_1,color=myColors[3],lw=0.8)      
#                
#                window_length = 19
                av_smoothed_specdiff = savgol_filter(specdiff,window_length,polyorder=4)
                window_length = 9
                av_smoothed_specdiff_1 = savgol_filter(specdiff_1,window_length,polyorder=4)
#                ax[1].plot(x_values,av_smoothed_specdiff,color='brown',lw=0.6)                   
#                ax[1].plot(x_values,0.6*av_smoothed_specdiff_1*maxspecdiff/maxfluo_1,color='blue',lw=0.7)    
#                ax[1].axis([450,3175,0,maxspecdiff*1.05])  
#                
#                ax[0].set_ylabel('Raw Raman \nIntensity (A.U.)')
#                ax[1].set_ylabel('Raman Intensity (A.U.) \n After Fluorescence Removed')                            
#                #ax[1].legend(['$\lambda$=10e6','$\lambda$=1'],loc='upper right',frameon=False,ncol=1,prop={'size':9})
#                ax[1].set_xlabel('Wavenumber (cm$^{-1}$)')
#                nxxzz = str(nxx)+' '+str(nzz)
#                ax[0].text(2700,0.92*maxmyspec2,fname)
#                ax[0].text(2850,0.84*maxmyspec2,str(nxxzz))
                
                temp = 0.8*maxspecdiff/maxfluo_1
                rincrease = 'specdiff_1 X '+str(np.rint(temp))
                
#                ax[0].text(2620,0.99*maxmyspec2,rincrease)
#          
                #######   LABEL PEAKS OF FOUR PANEL FIGS  #############################################################            
                ####   Some parameters used in where to plot labels - - not relevant to peak finding                
                if xlimmax > 1800:
                    maxd = 50
                    sl = 30
                if 751 < xlimmax < 1800: 
                    sl = 15  
                    maxd = 30            
                if xlimmax < 751:      
                    pkgrp = 9
                    pkgrpw = 15
                    maxd= 20
                    sl = 5
                ####  End of some parameters used in plotting but not relevant to peak finding
 
                spacer  = 0        
                #print (' ---- Find locations for lables of peaks  - - - - - - - - - - - - - - - - - - - - - - ')
               
                
                if npk > mxnpk : npk = mxnpk
                npkplot = npk
                
                if npk >= 1:
                    for kt in range(0,npkplot):
                        j = isort[kt]
                        #print (' ')
                        #print (kt,j,px[j],'kt   j  px')
                        if px[j] < 449:
                            #print('px[j]',px[j],j)
                            if px[j] < 447: iokj[j] = 0
                            px[j] = 449.
                        iprnt = 0
                        if iokj[j] == 1: iprnt = 1
                        # py[j] = height above the background - - found on specdiff_1
                        heightp0 = py[j] / maxmyspec
                        heightp  = heightp0
                        xwn0 = ( px[j] - xlimmin) / (xlimmax-xlimmin) 
                        xwn  = xwn0
                        #print (xwn,heightp,kt,j,px[j],py[j],dxdxr[j],'xwn heightp  kt   j  px  py  dxdxr')    
                        
                        
                        if iprnt == 1:
                            if heightp/maxmyspec < 0.65:    #If pk(j)/max < 0.65, set the initial label location on top of peak
                                xwn = xwn - 0.02    
                                if px[j] >= 1000: heightp = heightp0 + 0.25
                                if px[j] <  1000: heightp = heightp0 + 0.2
                                
                                for kp in range(0,kt):    #Compare with all previous peak labels to see if the above is ok
                                    jp = isort[kp]
                                    #print (xwn0,xwnj[jp],'xwn0 xwnj[jp]')
                                    if ( abs(xwn0-xwnj[jp]) < 0.03):   #There is a peak with higher dxdxr close by: look and and move if needed
                                    
                                        #print (xwn0,xwnj[jp],'xwn are close to each other - - xwn0 xwnj[jp]')
                                        if heightpj[jp] > 0.6:         
                                            print ('h>.6',kt,j,jp,px[j],px[jp],xwn,xwnj[jp],heightp,heightpj[jp],' kt,j,jp,px xwn,xwnj[jp],heightp,heightj[jp]')
                                            if ( xwn0 > xwnj[jp] ):       # the big peak is to the left of peak j: move j down and to rt
                                                xwn = xwn0 + 0.01
                                                print ('> > > xwn shifted to rt',xwn)
                                                if (heightpj[jp]-heightp) < 0.35:
                                                    heightp = heightp - 0.25           # shift new peak (j) down and to right
                                            if ( xwn0 < xwnj[jp] ):       # the big peak is to the right of peak j: move j down and to rt
                                                xwn = xwn0 - 0.035
                                                print ('< < <xwn shifted to left',xwn)
                                                if (heightpj[jp]-heightp) < 0.2:
                                                    heightp = heightp - 0.2      # shift new peak (j) down and to left
                                                    
                                        if heightpj[jp] <= 0.6: 
                                            print ('h<.6',kt,j,jp,px[j],px[jp],xwn,xwnj[jp],heightp,heightpj[jp],' kt,j,jp,px xwn,xwnj[jp],heightp,heightj[jp]')
                                            if ( xwn0 > xwnj[jp] ):       # the big peak is to the left of peak j: move j down and to rt
                                                xwn = xwn0 + 0.01
                                                print ('> >  xwn shifted to rt',xwn)
                                                if (heightpj[jp]-heightp) < 0.35:
                                                    heightp = heightp - 0.25           # shift new peak (j) down and to right
            
                                            if ( xwn0 < xwnj[jp] ):       # the big peak is to the right of peak j: move j down and to rt
                                                xwn = xwn0 - 0.03
                                                print ('< < xwn shifted to left',xwn)
                                                if (heightpj[jp]-heightp) < 0.2:
                                                    heightp = heightp - 0.2      # shift new peak (j) down and to left
        
                            if py[j]/maxmyspec > 0.65:  
                                heightp = 0.89    
                                xwn = xwn0 + 0.005
                            if ( xwn < 0.002): xwn = 0.002
                    
                            if iokj[j] == 1:
                                xwnp = xwn * (xlimmax-xlimmin) + xlimmin
                                pytmp = pyraw[j]
                                if (pytmp>maxmyspec*0.94): pytmp = maxmyspec * 0.94
                                #ax[0].text(px[j]-30,pytmp*1.02,int(np.rint(px[j])),rotation=90,color='red')
                                pytmp = py[j]
                                if (pytmp>0.94*maxspecdiff): pytmp = 0.94*maxspecdiff
                                if (pytmp<0.4*maxspecdiff): pytmp = 0.4*maxspecdiff
                                #ax[1].text(px[j]-30,pytmp,int(np.rint(px[j])),rotation=90,color='red')
                                #print('xwnp',xwnp,heightp*maxmyspec*1.5,'=heightp*maxv   ')
                                #print (kt,j,px[j],heightp,heightp*maxmyspec,'kt  j  px[j] heightp heightp*maxmyspec-aft ax.text(. . .)')
                                xwnj[j] = xwn
                                heightpsj[j] = heightp
                                xwnpsj[j] = xwnp
                                pxsj[j] = px[j]                                
                
                #print ('ilab before 1467 flabel.axes[].plot(xv,av)',ilab)
                #flabel.axes[ilab].plot(xv,av,color="C1",lw=0.7)    
                from matplotlib.ticker import MultipleLocator, FormatStrFormatter
                minorLocator = MultipleLocator(50)
                #flabel.axes[ilab].xaxis.set_minor_locator(minorLocator)


            ########   MAKE THIRD PLOT OF TYPE TO BE A SUBPLOT IN FIGURE 7     ***********************************
            #print(maxmyspec,npk,'maxmyspec  npk')
            if (maxmyspec > thr1) and ( npk > 0) :
                #print (maxmyspec,nxx,nzz,'= maxmyspec exceeds threshold')
                #f, ax = plt.subplots(1, sharex=True,figsize=(7.,4.))
                #plt.tight_layout(h_pad=-.8,w_pad=3.0)
                #myColors = rt.nicePalette()
                #print('                                                                         iplt=',iplt)
                #print('MAKE THIRD PLOT',maxmyspec,'maxmyspec - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
                
                ax[iplt].axis([449,xlimmax,0,maxmyspec*1.05])
                ax[iplt].plot(x_values,myspec,color='k')
                maxmyspec2 = np.nanmax(myspec) 
                #print(maxmyspec2,maxmyspec,' maxmyspec2 maxmyspec aft plot myspec')
                ax[iplt].plot(x_values,myfluo,color=myColors[1])
                
                #ax[iplt].plot(x_values,myfluo_1,color=myColors[4])
                
                window_length = 29
                av_smoothed_myspec = savgol_filter(myspec,window_length,polyorder=4)
                #ax[iplt].plot(x_values,av_smoothed_myspec,color='turquoise',lw=1.)                   
                ax[iplt].fill_between(x_values,myfluo,facecolor=myColors[1], alpha=0.2,edgecolor="white")
                #ax[iplt].fill_between(x_values,myfluo_1,facecolor=myColors[4], alpha=0.2,edgecolor="white")
                ax[iplt].plot(x_values,specdiff,color='grey',lw=0.8)
                ax[iplt].plot(x_values,specdiff_1,color='purple',lw=0.7)
                
                #ax[iplt].plot(x_values,av_smoothed_specdiff_1,color='darkgreen',lw=0.8)

                maxmyspec2 = np.nanmax(myspec) 
                if iplt == 3: ax[iplt].set_xlabel('Wavenumber (cm$^{-1}$)')                
                ax[iplt].set_ylabel('Raman \nIntensity (A.U.)')
                nxxzz = str(nxx)+' '+str(nzz)
                ax[iplt].text(2780,0.98*maxmyspec2,fnametime)
                ax[iplt].text(2930,0.9*maxmyspec2,str(nxxzz))
                
                from matplotlib.ticker import MultipleLocator, FormatStrFormatter
                minorLocator = MultipleLocator(50)
                #flabel.axes[ilab].xaxis.set_minor_locator(minorLocator)

                #print(' ')               
                print('npkplot',npkplot)
                pxused = np.zeros(npkplot+20)                        
                pyused = np.zeros(npkplot+20)
                nijk = 0
                
                for kt in range(0,npkplot-1):
                    j = isort[kt] 
                    pxplus = 0.   
                    pyplus = 0.
                    indexx = (np.abs(x_values - px[j])).argmin()
                    #print(indexx,x_values[indexx],myspec[indexx],'indexx,x_values[indexx],myspec[indexx]')
                    pyy = myspec[indexx]
                    if iokj[j] == 1:
                        pxplus = 0.
                        if kt == 0: 
                            if px[j] <= 520: pxplus = 520 - px[j] 
                            pytmp = 0.99*maxmyspec      
                            if px[j] < 2400:
                                text = ax[iplt].text(px[j]+pxplus,pytmp,"{0:0.1f}".format(px[j]),color='blue',alpha=1.,ha='center',va='center',size=13)
                            if px[j] >= 2400: 
                                pytmp = 0.55 * maxmyspec                            
                                text = ax[iplt].text(px[j]+pxplus,pytmp,int(np.rint(px[j])),color='blue',alpha=1.,ha='center',va='center',size=13)
                            text.set_path_effects([path_effects.Stroke(linewidth=0.1, foreground='black',alpha=0.4),path_effects.Normal()])
                            pxused[kt] = px[j]                      
                            pyused[kt] = py[j]
                            nijk = nijk + 1   
                        if (kt == 1) and (abs(px[j] - px[isort[0]])) > 15:
                            if pyy > 0.9*maxmyspec and (abs(px[j] - px[isort[0]]) > 150):               # set the peak label low to miss peak label 1
                                if px[j] <= 520: pxplus = 520 - px[j]       
                                pytmp = pyy -0.03*maxmyspec
                                if pytmp > 0.98 * maxmyspec: pytmp = 0.98 * maxmyspec 
                                if px[j] >= 2700: 
                                    pytmp = pyy - 0.4 * maxmyspec  
                                    if (pytmp+pyplus) < 0.38*maxmyspec: 
                                        pytmp = 0.38*maxmyspec
                                        pyplus = 0                                        
                                text = ax[iplt].text(px[j]+pxplus,pytmp,int(np.rint(px[j])),color='red',alpha=1.,ha='center',va='center',size=13)
                                text.set_path_effects([path_effects.Stroke(linewidth=0.1, foreground='black',alpha=0.4),path_effects.Normal()]) 
                                pxused[kt] = px[j]+pxplus                    
                                pyused[kt] = pytmp
                                nijk = nijk + 1   
                            if pyy <= 0.9*maxmyspec:
                                #pytmp = pyy - 0.25*maxmyspec
                                #if pytmp > 0.6  * maxmyspec: pytmp = 0.6  * maxmyspec
                                #if pytmp < 0.45 * maxmyspec: pytmp = 0.45 * maxmyspec
                                pytmp = 0.7 * maxmyspec
                                if px[j] <= 495: pxplus = 495 - px[j]    
                                if px[j] >= 2700 :
                                    pytmp = pyy - 0.2 * maxmyspec  
                                    if pytmp < 0.32*maxmyspec: pytmp = 0.32*maxmyspec     
                                if abs(px[j] - px[isort[0]]) < 60: 
                                    pytmp = 0.8 * maxmyspec
                                text = ax[iplt].text(px[j]+pxplus,pytmp,int(np.rint(px[j])),color='red',alpha=1.,ha='center',va='center',size=12,rotation=90)
                                text.set_path_effects([path_effects.Stroke(linewidth=0.1, foreground='black',alpha=0.4),path_effects.Normal()])  
                                pxused[kt] = px[j]+pxplus                    
                                pyused[kt] = pytmp
                                nijk = nijk + 1   
                        if kt > 1:
                            pyplus = 0.
                            ireject = 0
                            for ijk in range(0,nijk):
                                #print(j,ijk,nijk,'j,ijk,nijk')
                                if abs(px[j]-pxused[ijk])<6: ireject = 1
                                if abs(px[j]-pxused[ijk])>6: 
                                    if abs(px[j]-pxused[ijk])< 18:
                                        if dxdxr[ijk]/dxdxr[0] <= 0.04: ireject = 1      
                                        if dxdxr[ijk]/dxdxr[0] > 0.04:
                                            pyplus = -0.38*maxmyspec
                                    #if abs(px[j]-pxused[ijk]) > 18:
                                    #    pyplus = 0.13*maxmyspec                               
                            if ireject == 0:
                                if px[j] <= 496: pxplus = 496 - px[j]       
                                if (pyy >= 0.67*maxmyspec): pytmp = pyy - 0.2*maxmyspec
                                if (pyy <  0.67*maxmyspec): pytmp = pyy + 0.22*maxmyspec
                                if px[j] >= 2700: pytmp = pyy - 0.34 * maxmyspec   
                                if pyplus !=0: pytmp = pyy + pyplus   
                                if px[j] >= 2700: 
                                    if pytmp < 0.3*maxmyspec: 
                                        pytmp = 0.3*maxmyspec
                                        pxplus = 0
                                text = ax[iplt].text(px[j]+pxplus,pytmp,int(np.rint(px[j])),color='red',alpha=1.,ha='center',va='center',size=12,rotation=90)
                                text.set_path_effects([path_effects.Stroke(linewidth=0.1, foreground='black',alpha=0.4),path_effects.Normal()])                        
                                pxused[kt] = px[j] + pxplus                     
                                pyused[kt] = pytmp
                                nijk = nijk + 1
                                #ax[iplt].text(px[j]-30,pytmp,int(np.rint(px[j])),color='white',ha='center',va='center',size=13,rotation=90)                                  
                        #ax[iplt].text(px[j]-30,pytmp,int(np.rint(px[j])),rotation=90,color='red',fontsize='12') 
                                                    
                #print(' ')
                #print(figname,Dpeak,Gpeak,Gpeak/Dpeak,'Dpeak Gpeak Gpeak/Dpeak   < < < < < < < < < < ')                                       
                #print(' ')
                #print(' ')
                #####################################################################################################
                '''
                #ax[iplt].axis([450,3175,0,maxmyspec*1.05])
                ax[iplt].axis([450,xlimmax,0,maxmyspec*1.05])
                h0=ax[iplt].plot(x_values,myspec,color='k')
                maxmyspec2 = np.nanmax(myspec) 
                h1=ax[iplt].plot(x_values,myfluo,color=myColors[1])
                
                #h2=ax[iplt].plot(x_values,myfluo_1,color=myColors[4])
                
                window_length = 29
                av_smoothed_myspec = savgol_filter(myspec,window_length,polyorder=4)
                #Doughty commented this line
                #ax[iplt].plot(x_values,av_smoothed_myspec,color='turquoise',lw=1.)                   
                ax[iplt].fill_between(x_values,myfluo,facecolor=myColors[1], alpha=0.2,edgecolor="white")
                #ax[iplt].fill_between(x_values,myfluo_1,facecolor=myColors[4], alpha=0.2,edgecolor="white")
                h3=ax[iplt].plot(x_values,specdiff,color='C2',lw=1.5)
                h4=ax[iplt].plot(x_values,specdiff_1,color=[0.5,0.5,0.5],lw=1.5)
                #Doughty commented this line
                #ax[iplt].plot(x_values,av_smoothed_specdiff_1,color='darkgreen',lw=0.8)
                #plt.legend([h1,h2,h3,h4],['Processed','Fl(e6)','Fl(1)','Proc-Fl(e6)','Proc-Fl(e1)'])
                maxmyspec2 = np.nanmax(myspec) 
                if iplt == 3: ax[iplt].set_xlabel('Wavenumber (cm$^{-1}$)')                
                ax[iplt].set_ylabel('Raman \nIntensity (A.U.)')
                nxxzz = str(nxx)+' '+str(nzz)
                #ax[iplt].text(2700,0.98*maxmyspec2,fname)
                #ax[iplt].text(2850,0.9*maxmyspec2,str(nxxzz))
                
                from matplotlib.ticker import MultipleLocator, FormatStrFormatter
                minorLocator = MultipleLocator(50)
                #flabel.axes[ilab].xaxis.set_minor_locator(minorLocator)
               
                for kt in range(0,npkplot):
                    j = isort[kt] 
                    if iokj[j] == 1:
                        pytmp = py[j] * 3.2
                        if (pytmp>0.99*maxmyspec): pytmp = 0.99*maxmyspec
                        if (pytmp<0.3*maxmyspec): pytmp = 0.3*maxmyspec
                        ax[iplt].text(px[j]-30,pytmp,int(np.rint(px[j])),rotation=90,color='red',fontsize='12') 
                '''   
                ############################################################################################################
                                                    
                ba = str(maxmyspec)
                fignamepng = 's'+ba[2:6]+'_'+figname+'.png'
                if iDG ==1: fignamepng = 'RvF'+ba[2:6]+'_'+figname+'_'+str(int(100*ratioDGtoother))+'DG.png'
                for jspk in range(0,nspk-1):
                    if ipks[jspk] == 1:                
                        fignamepng = 'RvF'+ba[2:6]+'_'+figname+'_'+str(pklst[jspk])+'_'+str(pkmat[jspk])+'.png'
                if iCH ==1: fignamepng = 'RvF'+ba[2:6]+'_'+figname+'_'+str(int(100*ratioCH))+'CH.png'
                for jspk in range(0,nspk-1):
                    if ipks[jspk] == 1:                
                        fignamepng = 'RvF'+ba[2:6]+'_'+figname+'_'+str(pklst[jspk])+'_'+str(pkmat[jspk])+'.png'
                ## FINISHED THIRD MAIN PLOT - - - ONES TO ILLUSTRATE RMAX vs FMAX

                ilab = 999
                clusternum_actual = 0
                myordn = ord('a') 
                if ilab < llow:
                    py1w = py1 + 0.19
                    #print(px1,py1w,'px1  py1w before fig f text ')
                    #print ('before plot  center=',center)
                    f.axes[1].text(px1,py1w*maxy,center,rotation=90,color='red',fontweight='bold')
                    f.axes[1].text(px1,py1w*maxy,center,rotation=90,color='red')
                    #print('after: f.axes')
                    #flabel.axes[ilab].text(px1,py1w*maxy,center,rotation=90,color='blue',fontweight='bold')
                    #print('after: flabel.axes')
                    #ax[1].text(px1,py1w,center,transform = ax.transAxes,color='blue',fontweight='bold')
                    #ax[1].text(px1+0.03,py1,' n=' + str(num_spectra),transform = ax.transAxes,color='blue',fontweight='bold')

    return ptot
           
 
# function to use with scipy.optimize 
def one_gaussian(x, height,center,width,offset,dummy):
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset


# function to use with scipy.optimize      
def one_gaussian_slope(x, height,center,width,offset,slope):
    #print(height,center,width,offset,slope,'in one_gauss_slope: h cent wid offs sl')
    #yyy = height*np.exp(-(x - center)**2/(2*width**2)) + offset + slope*(x - center)
    #print(x,yyy,'x yyy in one_gaussian_slope')
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset + slope*(x - center)
                
                
def findpeaksG3(x,y,xmin,xmax,slope_Threshold=-999,amp_Threshold=-999,smooth_width=1,smooth_type=1,peakgroup=5):

    auto_amp_pctl = 25
    auto_slop_thrsh = 1
    
    P = np.empty((1,5))    
    
    smooth_type = 3 if smooth_type > 3 else smooth_type
    smooth_type = 1 if smooth_type < 1 else smooth_type
    smooth_type = 1 if smooth_width< 1 else smooth_type

    #If user does not specify slope and amplitude thresholds
    amp_Threshold = np.percentile(y,auto_amp_pctl) if amp_Threshold == -999 else amp_Threshold   
    slope_Threshold = auto_slop_thrsh if slope_Threshold == -999 else slope_Threshold
    
    smoothwidth=np.round(smooth_width)
    peakgroup=np.round(peakgroup)
    
    #IF we have a reasonable smooth width, then we smooth
    if smoothwidth>1:
        d=fastsmooth(deriv(y),smoothwidth,smooth_type)
    else:
        d=deriv(y)
        
    sign_d = np.sign(d)
        
    n = round(peakgroup/2.0+1)
    vectorlength = len(y)
    
    amp_test=amp_Threshold;
    peak=0;
    #Now loop through all points outside of smoothwidth
    
    jmin = int(2*np.around(smoothwidth/2)-2)
    jmax = len(y)-smoothwidth-1
    #print ('jmin jmax',jmin,jmax)
    
    #Here we loop through all 'good' locations
    for j in range(jmin,jmax):
        #Here we determine if we have a 'peak'
        #print(j,x[j],xmin,xmax,'j xj  xmin xmax')
        #if sign_d[j] > sign_d[j+1]:')
        if (sign_d[j] > sign_d[j+1]) and (xmin <= x[j]) and (xmin <= xmax):
            if d[j]-d[j+1] > slope_Threshold:
                if y[j]>amp_test or y[j+1] > amp_test:
                    
                    #Construct 'mini arrays' which we will use to fit our functions
                    xx = np.zeros((peakgroup,1))
                    yy = np.zeros((peakgroup,1))
                    
                    for k in range (0,peakgroup):
                        groupindex = j + k - n + 2
                        groupindex = 1 if groupindex < 1 else groupindex
                        groupindex = vectorlength if groupindex > vectorlength else int(groupindex)
                        xx[k] = x[groupindex]
                        yy[k] = y[groupindex]
                        
                    if peakgroup > 2:
                        [Height,Position,Width] = gaussfit(xx,yy)
                        PeakX = Position
                        PeakY = Height
                        MeasuredWidth = Width
                    else:
                        pdb.set_trace()
                        

                        try:
                            PeakY = np.nanmax(yy)
                            pindex = val2ind(yy,PeakY)
                            PeakX = xx(pindex(1))
                            MeasuredWidth = 0
                        except:
                            pdb.set_trace()
                            continue
                        
                    if np.isnan(PeakX) or np.isnan(PeakY) or PeakY < amp_Threshold:
                        pass
                    else:
                        new_pk_array = np.array([peak,PeakX,PeakY,1.0646*PeakY*MeasuredWidth])
                        #P[peak,:] = np.array([peak,PeakX,PeakY,MeasuredWidth,1.0646*PeakY*MeasuredWidth])
                        #P = np.append(P,np.zeros((1,5)),axis=0)
                        if peak == 0:
                            P = new_pk_array
                        else:
                            P = np.vstack((P,new_pk_array))
                        peak=peak+1; # Move on to next peak
    shp = P.shape
    if len(shp) == 1:
        #Hanel scenario
        P = P[None,:]
                    
    return P                
                

   
########################################################################################################
########################################################################################################
    
#####################
#
#   Load the data - comment out after we are done
#

#,rdat_1b,rmta_1b,rdat_2bf0,rdat_3bf6,rflu_3bf6): 
### To run more quickly the following ''' . . . '''   can be commented out if these lines have been run once and the kernal has not been restarted
### the first time, make sure the next 27 lines are not commented out


#alldata_base = "alldata_r13_TimeSeries_2_ClCsBkr"
alldata_base = "alldata_r13_TimeSeries_3_ClCsBkrBrn"
file_rdat = alldata_base + ".rdat"
print('file_rdat',file_rdat)
file_rmta = alldata_base + ".rmta"
rdat = np.loadtxt(file_rdat,delimiter=',')
dt_meta = pd.read_table(file_rmta,sep=',')
x_values = rdat[0,:]
rdatdata = rdat[1::,:]
#Loads the Fluorescence Removed Spectrum

alldata_base = "alldata_r13_TimeSeries_5_ClCsBkrBrnFle6"
file_rdat = alldata_base + ".rdat"
file_rmta = alldata_base + ".rmta"
file_rflu = alldata_base + ".rflu"
rdat_f6= np.loadtxt(file_rdat,delimiter=',')
dt_meta_f6 = pd.read_table(file_rmta,sep=',')
rdatdata_f6 = rdat_f6[1::,:]
rfluo_f6 = np.loadtxt(file_rflu,delimiter=',')
rdatfluo_f6 = rfluo_f6[1::,:]

alldata_base = "alldata_r13_TimeSeries_4_ClCsBkrBrnFle0"
file_rdat = alldata_base + ".rdat"
file_rmta = alldata_base + ".rmta"
rdat_f0= np.loadtxt(file_rdat,delimiter=',')
dt_meta_f0 = pd.read_table(file_rmta,sep=',')
rdatdata_f0 = rdat_f0[1::,:]



#="REBS RN 13_20160825_033821"
# THE FOLLOWING ARE IN PYTHON NOTATION FOR INDEXING WHICH STARTS AT 0:
# SO, SUBTRACT ONE FROM THE NUMBERS BELOW TO GET THE ACTUAL NUMBERS FOR Nx and Nz
# IF 11, 23 is desired you must enter: 12, 24 (11+1, 23+1)


fh = []
fnamespks = []

pklst = [463,     495,        503,        507,     514,        609,    613,    660,       699,      910,   972,    980,  1000, 1008,   1016,   1020, 1025,   1042,   1045,   1060,  1067,    1085,   1098,     1356,       1386,      1442, 1465,      1470,   1475,   2857, 2932]
pkmat = ['quartz','stilbite','anorthite','albite','orthoclase','FeO','rutile','grunerite','jadeite','CAS','AmSO4','MSO4','Ba','gypsum','CaSO4','unk','FeSO4','AmNO3','MNO3','trona','NaNO3','CaCO3','dolomite','NaFormate','CaFormat','CC','whewell','FeMgOx','weddel','CH1','CaAc']
nspk = len(pklst)
for j in range(0,nspk-1):
    filenamepeaks = 'peaks'+str(pklst[j])+'.txt'
    fnamespks.append(filenamepeaks)
    myfh = open(filenamepeaks,"w")
    fh.append(myfh)   
    


filenameload = open("namesloaded.txt","w")
#flabel, axes = plt.subplots(nrows=3, ncols=2,figsize=(6,8), sharex=True)
ilab = 0
ptot = 0
iii = 0
ilaab = 0

nametimelist = [

"REBS RN 13_20160825_172824_15_15",   # gypsum                     FIg. 5
"REBS RN 13_20160825_035418_31_11",   # anhydrite + BC
"REBS RN 13_20160825_104614_10_04",   # calcite
"REBS RN 13_20160825_103020_17_19",   # dolomite     S/N 

"REBS RN 13_20160825_203814_41_01",   # jadeite 699                Fig. 6
"REBS RN 13_20160825_190312_33_05",   # othroclase 513 475
"REBS RN 13_20160825_113352_22_05",   # hematite
#"REBS RN 13_20160825_094326_16_18",   # iron oxide
"REBS RN 13_20160825_174422_06_14",   # Quartz 465

"REBS RN 13_20160825_132338_40_22",   # oxalate 1465               Fig. 7
"REBS RN 13_20160825_070433_39_09",   # oxalate 1467   Large s/n
"REBS RN 13_20160825_061652_20_20",   # oxalate 1468   Large s/n
"REBS RN 13_20160825_152129_33_11",   # oxalate 1469

"REBS RN 13_20160825_014758_12_23",   # oxalate 1470              Fig. 8
"REBS RN 13_20160825_075215_22_09",   # oxalate 1470
"REBS RN 13_20160825_080809_34_15",   # oxalate 1473 strong weddelite
"REBS RN 13_20160825_132338_23_00",   # oxalate 1476 

"REBS RN 13_20160825_183143_28_13",   # BC with D >> G - clean    Fig. 9
"REBS RN 13_20160825_033821_33_05",   # BC with D >> G - clean
"REBS RN 13_20160825_123647_22_14",   # BC D > G but similar
"REBS RN 13_20160825_191905_11_01",   # BC  D = G

"REBS RN 13_20160825_030638_05_00",   # CH                          Fig. 10
"REBS RN 13_20160825_064847_26_23",   # CH + 1443, 1308 1048
"REBS RN 13_20160825_171234_35_03",   # CH  similar to wood 2930
"REBS RN 13_20160825_193501_12_03",   # CH  similar to wood 2934

# BEGIN SUPPLEMENTAL FIGS

"REBS RN 13_20160825_172824_17_15",   # gypsum                     Fig S10
"REBS RN 13_20160825_172824_14_16",   # gypsum
"REBS RN 13_20160825_103020_16_19",   # dolomite     S/N
"REBS RN 13_20160825_103020_28_02",   # dolomite     S/N
 
"REBS RN 13_20160825_130806_13_01",   # quartz + BC                Fig S11
"REBS RN 13_20160825_203814_15_06", 
"REBS RN 13_20160825_095858_10_24",   # quartz (very weak)
"REBS RN 13_20160825_010021_19_09",   # quartz (very weak)

"REBS RN 13_20160825_203814_40_00",   # jadeite 699              Fig. S12
"REBS RN 13_20160825_103020_16_19",   # dolomite     S/N 
"REBS RN 13_20160825_200631_29_19",   # BC  D >> G
"REBS RN 13_20160825_094326_14_08",

# EXTRA FIGURES

"REBS RN 13_20160825_103020_17_19",   # dolomite     S/N         Fig. E14
"REBS RN 13_20160825_030638_05_00",   # CH   
"REBS RN 13_20160825_080809_34_15",   # oxalate 1473 strong weddelite?     
"REBS RN 13_20160825_190312_33_05",   # othroclase 513 475   

"REBS RN 13_20160825_103020_17_19",   # dolomite     S/N         Fig. E15
"REBS RN 13_20160825_030638_05_00",   # CH   
"REBS RN 13_20160825_080809_34_15",   # oxalate 1473 strong weddelite? 
"REBS RN 13_20160825_113352_22_05",   # hematite  

"REBS RN 13_20160825_203814_13_09",   # 522 + fluor + D/G       Fig. E16
"REBS RN 13_20160825_103020_28_03",   # anorthite
"REBS RN 13_20160825_203814_29_05",   # 503 anorthite - CaAlSi2O8 
"REBS RN 13_20160825_063250_31_13",   # oxalate 1467
"REBS RN 13_20160825_070433_39_09",   # oxalate 1467
"REBS RN 13_20160825_103020_39_04",   # oxalate 1470
"REBS RN 13_20160825_132338_25_00",   # weddellite
"REBS RN 13_20160825_033821_33_25",   # anhydrite + BC

"REBS RN 13_20160825_103020_39_04",   # dolomite
"REBS RN 13_20160825_033821_33_03",
"REBS RN 13_20160825_215713_18_08",
"REBS RN 13_20160825_114940_35_15",

"REBS RN 13_20160825_215713_19_08",  
"REBS RN 13_20160825_075215_22_09",  
"REBS RN 13_20160825_103020_38_02",
"REBS RN 13_20160825_203814_40_00",

"REBS RN 13_20160825_200631_29_04", 
"REBS RN 13_20160825_064847_22_07",
"REBS RN 13_20160825_203814_15_06",   # 469 is highest peak _ BC
"REBS RN 13_20160825_150531_35_02",

"REBS RN 13_20160825_014758_32_21",  
"REBS RN 13_20160825_063250_38_17",  ##,, OK
"REBS RN 13_20160825_010021_13_16", 
"REBS RN 13_20160825_103020_16_12",

]


plt.show()
plt.close('all')
print('   ')

f7c, ax = plt.subplots(4, sharex=True,figsize=(8.,10.))

myColors = rt.nicePalette()    
ii = -1
ifigold = 0

fignames = [
            #"DH19a_Processing_FigTest1.png",
            #"DH19a_Processing_Figure05pre.png",
            "DH19a_Processing_Figure05.png",
            "DH19a_Processing_Figure06.png",
            "DH19a_Processing_Figure07.png",
            "DH19a_Processing_Figure08.png",
            "DH19a_Processing_Figure09.png",
            "DH19a_Processing_Figure10.png",
            "DH19a_Processing_FigureS10.png",
            "DH19a_Processing_FigureS11.png",
            "DH19a_Processing_FigureS12.png",
            "DH19a_Processing_FigureE14.png",
            "DH19a_Processing_FigureE15.png",
            "DH19a_Processing_FigureE16.png",
            "DH19a_Processing_FigureE17.png",
            "DH19a_Processing_FigureE18.png",
            "DH19a_Processing_FigureE19.png",
            "DH19a_Processing_FigureE20.png",
            "DH19a_Processing_FigureE21.png",
            "DH19a_Processing_FigureE22.png",
            "DH19a_Processing_FigureE23.png",
            "DH19a_Processing_FigureE24.png",
            "DH19a_Processing_FigureE25.png",
            "DH19a_Processing_FigureE26.png",
            "DH19a_Processing_FigureE27.png",
            "DH19a_Processing_FigureE28.png",
            "DH19a_Processing_FigureE29.png",
            "DH19a_Processing_FigureE30.png",
            "DH19a_Processing_FigureE31.png",
            "DH19a_Processing_FigureE32.png",
            "DH19a_Processing_FigureE33.png",
            "DH19a_Processing_FigureE34.png",
            "DH19a_Processing_FigureE35.png",
            "DH19a_Processing_FigureE36.png",
            "DH19a_Processing_FigureE37.png",
            "DH19a_Processing_FigureE38.png",
            "DH19a_Processing_FigureE39.png",
            "DH19a_Processing_FigureE40.png",
            "DH19a_Processing_FigureE41.png",
            "DH19a_Processing_FigureE42.png",
            "DH19a_Processing_FigureE43.png",
            "DH19a_Processing_FigureE44.png",
            "DH19a_Processing_FigureE45.png",
            "DH19a_Processing_FigureE46.png",
            "DH19a_Processing_FigureE47.png",
            ]


#fignames = [
#            "DH19a_Processing_FigTest1.png",
#           [
filef = open("filef.txt","w")
filet = open("ratios.txt","w")
fpkDG = open("pksDG.txt","w")  
filewdt = '104614_09_03.txt'
filewd = open(filewdt,"w")  
           
 
for nametimeload in nametimelist:
    #print(nametimeload,' nametimeload')
    ii = ii + 1
    ifig = int(ii/4)
    #print(ii,ifig,'ii,ifig')
    if ifig > ifigold:
        #print('ifig > ifigold')
        towrite1 = '   \n '
        fpkDG.write(towrite1)
        ifigold = ifig
        #fname = 'Fig7c_'+str(ifigold-1)+'.png'
        fname=fignames[ifigold-1]
        f7c.subplots_adjust(wspace=0, hspace=0)    #If figures appear as if hspace not equal to zero, check the anaconda/ipython/matplotlib graphics settings.
                              # Something is overriding this command. Most likely related to graphics backend.  Or forced use of tight layout for inline plots. 
                              # Backend should be "automatic", not "inline"
        plt.savefig(fname)
        plt.close('all')
        #closed old, now open new
        #print(ii,' ii')
        f7c, ax = plt.subplots(4, sharex=True,figsize=(8.,10.))
        
        #plt.tight_layout(h_pad=-.8,w_pad=3.0)
        myColors = rt.nicePalette()    
    filenameload.write(nametimeload)
    nameload = nametimeload[0:26]
    nxxmin = int(nametimeload[27:29])
    ipm = 1
    nxxmax = nxxmin + ipm
    nzzmin = int(nametimeload[30:32])
    nzzmax = nzzmin + ipm  
    #print(nameload,nxxmin,nxxmax,nzzmin,nzzmax)
    #print('ptot =',ptot,'               nxxmin nzzmin =',nxxmin,nzzmin,'    <<-----------------------------------------------------------------')   
#    nxxmin = 32   #  0  for full run
#    nxxmax = 34   # 43
#    nzzmin = 24   # 0 for full run 
#    nzzmax = 26   # 60 for full run
#
#    if nxzall == 1:
#        nxxmin = 0   #  0  for full run
#        nxxmax = 43   # 43
#        nzzmin = 0   # 0 for full run 
#        nzzmax = 60   # 60 for full run    
    thr1 = 0.0001  #  was 0.0015  # was .007   (later 0.0015)
    thr2 = 0.0015  # was 0.002
#    thr1 = 0.005  # was .007   (later 0.0015)
#    thr2 = 0.005  # was 0.002
    xlimmax = 3150
    llow = 40 ## was 14
    lhi = 8888 ## was 18
    px1 = 0.55
    px2 = 0.05
    px3 = 0.52
    py1 = 0.62
    py2 = 0.62
    py3 = 0.62
    amthr  = 0.000001
    zcrit  = 0.00004
    dxdthr = 0.0001
    fnpks = 'peaks_'+nameload+'.txt'  
    fpks = open(fnpks,"w")        
    print('ptot bef call make_Fig_subplts_labels =',ptot,'  with nameload=',nameload,'    where  nxxmin and nzzmin are ',nxxmin,nzzmin,'    <<----------')
    iplt = ii - 4*ifig
    if iplt == 6: zlimmax = 1050
        
    mxnpk = 7
    if ii == 0: mxnpk = 3    # Fig. 5
    if ii == 1: mxnpk = 4
    if ii == 2: mxnpk = 3
    if ii == 3: mxnpk = 5
    if ii == 4: mxnpk = 5    # Fig. 6
    if ii == 5: mxnpk = 5
    if ii == 6: mxnpk = 6
    if ii == 7: mxnpk = 7
    if ii == 8: mxnpk = 8    # Fig. 7
    if ii == 9: mxnpk = 8
    if ii == 10: mxnpk = 8
    if ii == 11: mxnpk = 8
    if ii == 12: mxnpk = 8   # Fig. 8
    if ii == 13: mxnpk = 8
    if ii == 14: mxnpk = 7
    if ii == 15: mxnpk = 8
    if ii == 16: mxnpk = 3   # Fig. 9
    if ii == 17: mxnpk = 4
    if ii == 18: mxnpk = 4
    if ii == 19: mxnpk = 3
    if ii == 20: mxnpk = 8   # Fig. 10
    if ii == 21: mxnpk = 7
    if ii == 22: mxnpk = 7
    if ii == 23: mxnpk = 7
    if ii == 24: mxnpk = 4   # Fig. S10
    if ii == 25: mxnpk = 4
    if ii == 26: mxnpk = 4
    if ii == 27: mxnpk = 4
    if ii == 28: mxnpk = 7   # Fig. S11
    if ii == 29: mxnpk = 6
    if ii == 30: mxnpk = 5
    if ii == 31: mxnpk = 4
    if ii == 32: mxnpk = 6   # Fig. S12
    if ii == 33: mxnpk = 4
    if ii == 34: mxnpk = 5
    if ii == 35: mxnpk = 4
    if ii == 36: mxnpk = 5    # Fig. E14
    if ii == 37: mxnpk = 6
    if ii == 38: mxnpk = 6
    if ii == 39: mxnpk = 6
    if ii == 40: mxnpk = 6    # Fig. E15
    if ii == 41: mxnpk = 7
    if ii == 42: mxnpk = 7
    if ii == 43: mxnpk = 6
    if ii == 44: mxnpk = 6    # Fig. E16
    if ii == 45: mxnpk = 5
    if ii == 46: mxnpk = 6
    if ii == 47: mxnpk = 7
    if ii == 48: mxnpk = 7    # Fig. E17
    if ii == 49: mxnpk = 4
    if ii == 50: mxnpk = 5
    if ii == 51: mxnpk = 5
    
        #print(ifig,' = ifig',ii,' = ii    fname = ',fignames[ifig],'   + + + + + + + + + + + + + +  mxnpk =',mxnpk)
        
    ptot=make_Fig_subplts_labels(iplt,ii,ptot,nameload,thr1,thr2,nxxmin,nxxmax,nzzmin,nzzmax,xlimmax,llow,lhi,px1,px2,px3,py1,py2,py3,zcrit,amthr,dxdthr,ilab,pklst,pkmat,fpks,fh,rdatdata,dt_meta,rdatdata_f0,rdatdata_f6,rdatfluo_f6,filef,fpkDG,mxnpk)
     
    print('ptot on ret from make_Fig_subplts_lables ',ptot,'   fname = ',fignames[ifig],'   + + + + + + + + + + + + + + + + + + + + + + + + + +')
    print( ' ')
    print('  ')
    filewd.close()           

f7c.subplots_adjust(wspace=0, hspace=0)
fname=fignames[ifig]
#print('completed fig = ',fname)
#fname = 'Fig7c_'+str(ifig)+'.png'
#plt.show()
plt.savefig(fname)
plt.close('all')

filef.close()
filet.close()
fpkDG.close()
fpks.close()
filewd.close()
fpkDG.close()
  
for j in range(0,nspk-1):
    myfh = fh[j]
    myfh.close()
