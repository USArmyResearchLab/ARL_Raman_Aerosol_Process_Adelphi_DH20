###########################
#
#   SizeTools
#
#   Tools to work with data of different sizes
#

import pandas as pd
import numpy  as np    
import re

##################
#
#   Stats and Computations
#
#
def subsample_time(myPSD,start_time,end_time):
    good_samples = (myPSD.sampleTimes > start_time) & (myPSD.sampleTimes < end_time)
    shorteneddata_PSD = myPSD.dNdlogDp[good_samples,:]  
    return shorteneddata_PSD
    
def compute_stats_sampletime(myPSD,start_time,end_time):
    shorteneddata = subsample_time(myPSD,start_time,end_time)
    sd_mean = np.nanmean(shorteneddata,axis=0)
    sd_std = np.nanstd(shorteneddata,axis=0)
    return sd_mean,sd_std
    
##################
#
#   Plotting
#
#    
def nicePalette():
    nicepalette = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    return nicepalette

def plot_format(fs=16,ls=10):
    import matplotlib.pyplot as plt
    plt.rcdefaults()

    fSize = fs
    fName = 'Arial'
    fWght = 'bold'
    defLW = 2
    #Format the plots
    
    font = {'family' : 'normal',
    'weight' : fWght,
    'size'   : fSize}
    
    
    plt.rc('font', **font)
    
    plt.rc('legend',fontsize=10)
    
    
    plt.rc('axes',linewidth=defLW)
    plt.rc('axes',labelsize=fSize)
    plt.rc('axes',labelweight=fWght)
    
    #plt.rc('axes',edgecolor=[0.1,0.1,0.1])#,color='black')
    plt.rc('lines',linewidth = defLW)

def plot_3dTS(myPSD,savename):
    
    import matplotlib.pyplot as plt

    plt.figure()
    #plt.pcolormesh(myPSD.Time_HOD,myPSD.binEdges,myPSD.data.transpose(),vmin=0,vmax=5)
    plt.pcolormesh(myPSD.Time_HOD,myPSD.binEdges,myPSD.dNdlogDp.transpose(),vmin=0,vmax=50)
    plt.colorbar()
    plt.yticks(myPSD.binCenters)             
    plt.ylim([0.25,12])    
    plt.xlim([np.nanmin(myPSD.Time_HOD),np.nanmax(myPSD.Time_HOD)])   
    plt.xlabel('Hour of Day (EST)')
    plt.ylabel('Size (um)')
    plt.title('Concentration (Dn/DlogDp), #/cm^-3')  
    plt.savefig(savename)         

                        
##################
#
#   Loading
#
#                           
def load_APS(file_load,delim=','):
    myTSI = TSI()

    df = pd.read_table(file_load,header=6,parse_dates=[[1,2]],index_col=0,delimiter=delim)
    
    #Handle Bin Edges    
    bin_edges_aps = np.array([0.4870,0.5228,0.5624,0.6039,0.6492,0.6975,0.7495,0.8055,0.8663,0.9309,1.0004,1.0746,1.1547,1.2406,1.3332,1.4335,1.5396,1.6544,1.7779,1.9110,2.0538,2.2071,2.3711,2.5486,2.7387,2.9432,3.1622,3.3985,3.6522,3.9242,4.2165,4.5320,4.8696,5.2333,5.6230,6.0426,6.4941,6.9784,7.4993,8.0588,8.6598,9.3061,10.0035,10.7463,11.5470,12.4055,13.3316,14.3349,15.3960,16.5439,17.7787,19.1099,20.5353])
    dLogDp = np.diff(np.log10(bin_edges_aps))
    myTSI.binEdges = bin_edges_aps
    logbe = np.log10(myTSI.binEdges)
    bdiff = np.divide(np.diff(logbe),2)
    logbc = logbe[0:-1] + bdiff
    myTSI.binCenters = np.power(10,logbc)
    
    #Handle Times
    myTSI.sampleTimes = pd.to_datetime(df.index)
    myTSI.Time_HOD = myTSI.Time_HOD  = np.array([(time.hour + np.true_divide(time.minute,60) + np.true_divide(time.second,3600)) for time in myTSI.sampleTimes])
    
    #Check type and compute data values
    if df['Aerodynamic Diameter'][1] == 'dN':
        myTSI.data=df.ix[:,2:54].as_matrix()
        myTSI.dNdlogDp = np.divide(myTSI.data,dLogDp)
    
    return myTSI
    
    
def load_3330(input_file):

    myTSI = TSI()
    
    with open(input_file,'r') as f:
        myTSI.instrName = f.readline().strip().split(',')[1]
        myTSI.modelNum = int(f.readline().strip().split(',')[1])
        myTSI.serialNum = int(f.readline().strip().split(',')[1])
        myTSI.firmware = f.readline().strip().split(',')[1]
        myTSI.calDate = f.readline().strip().split(',')[1]
        myTSI.protocolName = f.readline().strip().split(',')[1]
        myTSI.testStartTime = f.readline().strip().split(',')[1]
        myTSI.testStartDate = f.readline().strip().split(',')[1]
        myTSI.testLength = f.readline().strip().split(',')[1]
        myTSI.sampleInterval= f.readline().strip().split(',')[1]    
        myTSI.numChannels= int(f.readline().strip().split(',')[1])+1   
        myTSI.ChannelNum = np.arange(0,myTSI.numChannels,1)
        myTSI.cutPoint = np.zeros(myTSI.numChannels)
        for channel in myTSI.ChannelNum:
            myTSI.cutPoint[channel] = float(f.readline().strip().split(',')[1])
        myTSI.alarm = f.readline().strip().split(',')[1]
        myTSI.Density = float(f.readline().strip().split(',')[1])    
        myTSI.refractiveIndex = f.readline().strip().split(',')[1]
        myTSI.sizeCorrFac = float(f.readline().strip().split(',')[1]) 
        myTSI.flowCal = float(f.readline().strip().split(',')[1]) 
        myTSI.deadTimeCorrFac = float(f.readline().strip().split(',')[1]) 
        myTSI.errors = f.readline().strip().split(',')[1]  
        myTSI.numSamples = int(f.readline().strip().split(',')[1])
        test = f.readline().strip()
        
        if test != ',':
            print("Error on the read")
            print(test)
    
        myTSI.columnNames = f.readline().strip().split(',')   
        
        #PreAllocateNumpyArrays:
        myTSI.elapsedTime = np.zeros(myTSI.numSamples)
        myTSI.rawdata = np.zeros((myTSI.numSamples,myTSI.numChannels))
        myTSI.deadTime = np.zeros(myTSI.numSamples)  
        myTSI.T = np.zeros(myTSI.numSamples)  
        myTSI.RH = np.zeros(myTSI.numSamples)      
        myTSI.P = np.zeros(myTSI.numSamples)   
        myTSI.alarms = []
        myTSI.errors = [] 
        nC = myTSI.numChannels
        nS = myTSI.numSamples
        myTSI.rawdata[:] = np.NaN
        datastr = '%i,' + ' %04d,'*nC + '%i,%04d,%04d,%04d,%s,%s'  
        for i in range(0,myTSI.numSamples):
            data = f.readline().strip().split(',')  
            myTSI.elapsedTime[i] = int(data[0])
            myTSI.rawdata[i,:] = np.asarray(data[1:nC + 1],dtype=np.float32)
            myTSI.deadTime[i] = float(data[nC + 1])
            myTSI.T[i] = float(data[nC + 2]   )     
            myTSI.RH[i] = float(data[nC + 3])           
            myTSI.P[i] = float(data[nC + 4])
            myTSI.alarms.append(data[nC + 5])
            myTSI.errors.append(data[nC + 6])
            
        
        myTSI.binEdges = myTSI.cutPoint
        #Note: We do not do any serious work with the largest bin
        #We do not know how big the particles are
        
        ##Test analysis:
        #this uses the equation in 5.5.
        #We currently do not applythe deadtime correction factor. 
        
        #Concentration Factor
        #This is the concentration!
        #Get sample interval in seconds
        siv = np.asarray(myTSI.sampleInterval.split(':'),dtype='float')
        samp_time = siv[0]*3600+siv[1]*60+siv[2]
        
        samp_time_corr = np.subtract(samp_time,np.multiply(myTSI.deadTimeCorr,myTSI.deadTime))
        concentration_factor = np.multiply(myTSI.flowRate,samp_time)
        myTSI.data = np.divide(myTSI.rawdata,concentration_factor)
        
        sumparts = np.nansum(myTSI.data[:,:],axis=1)
    
        #Now for the good stuff   
        #Convert sizes to DnDlogDp
        #Discard the last size bin because it counts but does not size particles larger than 10 micron
        
        #Not actually bin center
        myTSI.binCenters = myTSI.cutPoint[0:-1] + np.divide(np.diff(myTSI.cutPoint),2)
        logvs = np.log10(myTSI.cutPoint)
        dlogDp = np.diff(logvs)
        myTSI.dNdlogDp = np.divide(myTSI.data[:,0:-1],dlogDp)
    
        leftloc = myTSI.cutPoint[0:-1]
        width = np.diff(myTSI.cutPoint)
        myTSI.startDateTime = pd.to_datetime(myTSI.testStartDate + ' ' + myTSI.testStartTime)
        myTSI.sampleTimes = myTSI.startDateTime + pd.to_timedelta(myTSI.elapsedTime,'s')
        
        myTSI.Time_HOD  = np.array([(time.hour + np.true_divide(time.minute,60) + np.true_divide(time.second,3600)) for time in myTSI.sampleTimes])
        return myTSI
                            
def load_EDM164(input_file):
    myGRM = GRIMM()
    average=False
    nC = 0
    nc = 0
    current_time = []

    grimm_sample_duration = pd.Timedelta('6 seconds')
    
    with open(input_file,'r') as f:
        alldata = f.readlines()
        for line in alldata:
        #for line in f.readlines():
        #    print line
            data = line.strip()
            #print data
            
            #Handle scenarios where there's no data
            if len(data) == 0:
                #no data in line
                continue
            if (not current_time) & (data[0] != 'P'):
                #we started the file in the middle of a read and don't know the time
                continue
            if data == 'POWER OFF':
                #Power turned off. Could confuse with the 'P' command
                continue
                
            if data[0] == 'P':
                #It is a new measurement, parse the new measurement
                #Clean, strip the 'p , split by tabs'
                p_clean = re.sub(r'[P_]','',data).split()
                #Date and date string
                #print p_clean
                datestr = p_clean[1] + '/' + p_clean[2] + '/20' +p_clean[0] + ' ' +p_clean[3] + ':' +p_clean[4] 
                current_time = pd.to_datetime(datestr)
                measurement_counter = 0
                #pdb.set_trace()
    
                #print datetime
            elif data[0] == 'N':
                #This means we have a new measurement 
                #We split the PM mass into three. 
                #If we go above category then we handle this problem
                n_clean = re.split('[ ,]+',data.strip())
                try:
                    pm1 = np.divide(float(n_clean[3]),10)
                except:
                    pm1 = np.NaN
                try:
                    pm25  = np.divide(float(n_clean[2]),10)
                except:
                    pm25 = np.NaN
                try:
                    pm10  = np.divide(float(n_clean[1]),10)     
                except:
                    pm10 = np.NaN
                 
                myGRM.PM1= np.append(myGRM.PM1,pm1)
                myGRM.PM10= np.append(myGRM.PM10,pm10)
                myGRM.PM25= np.append(myGRM.PM25,pm25)
                myGRM.PM_Time=np.append(myGRM.PM_Time,current_time)
                
                c_values = np.array([])
                C_values = np.array([])
                nc=0
                nC = 0       
            elif data[0] == 'C':
                #C is full power laser
                nC = nC + 1     
                #Parse data
                C_clean = data.strip(':;').split()
                C_values = np.append(C_values, np.asarray(C_clean[1::],dtype=np.float32))
                
                            
            elif data[0] == 'c': 
                #c is low power laser
                nc = nc + 1     
                #Parse data
                c_clean = data.strip(':;').split()
                #Note that we skip the last value in this case. The reason for this is that we wish to avoid problems. 
                c_values = np.append(c_values, np.asarray(c_clean[1:-1],dtype=np.float32))
    
                
                if nc == 2 & nC ==2:
    
                    #compute cross checks
                    len_c = len(c_values)
                    len_C = len(C_values)
                    
                    if len_c != len_C & len_C != 8:
                        #Data Length Cross Check
                        continue
                    
                    if int(C_values[15]) != int(c_values[0]):
                        #LaserPowerPM CrossCheck
                        continue
                        
                    #Concatenate
                    
                    short_concentration  = np.append(C_values,c_values[1::])
                    #Convert to cubic centimeter:
                    cubic_centimeters_per_liter = 1000.0
                    short_concentration = np.divide(short_concentration,cubic_centimeters_per_liter)
                                    
                    #Here ae manually add six seconds every time the Grimm measures. Use Caution. 
                    current_sample_time = current_time + measurement_counter*grimm_sample_duration
                    measurement_counter = measurement_counter + 1
                    
                    myGRM.sampleTimes = np.append(myGRM.sampleTimes,current_sample_time)
                    #note: GRIMM does not log seconds (not quite sure why)....but 
                    #I do not trust the computer time for this. Thus, I am going to treat it as if it
                    #Is accurate down to the second, but I don't know if this is in fact correct. 
                    
                    if len(myGRM.rawdata) == 0:
                        #Raw concentration data
                        myGRM.rawdata = short_concentration
                    else:
                        myGRM.rawdata = np.vstack((myGRM.rawdata,short_concentration))
                
    #Fix Data Values
    #myGRM.cleanedData = np.zeros(np.data.shape)
    myGRM.data = np.copy(myGRM.rawdata)
    myGRM.data[:,0:-1] = np.fliplr(np.diff(np.fliplr(myGRM.rawdata)))
                    
    #Calculate DnDlogDp
    myGRM.dNdlogDp = np.divide(myGRM.data,myGRM.dlogDp)
    myGRM.Time_HOD  = np.array([(time.hour + np.true_divide(time.minute,60) + np.true_divide(time.second,3600)) for time in myGRM.sampleTimes])
    
    return myGRM
    
def load_3330_usb(input_file):
    return 'werwer'
    
def load_APS_AIM(input_file):
    return 'werwer'    
    
class GRIMM:
    #This is a setup data for the GRIMM instrument
    #

    def __init__(self):
        import numpy as np

        self.instrName=[]
        self.modelNum=[]
        self.serialNum=[]
        self.firmware=[]
        self.calDate=[]
        self.protocolName = ''
        self.testStartTime = ''
        self.testStartDate = ''
        self.testLength = ''
        self.sampleInterval = ''
        self.numChannels = 31 #Total number of channels 
        self.ChannelNum = np.arange(0,32,1) #The number for each channel
        cpv = np.array([0.28,0.30,0.35,0.40,0.45,0.50,0.58,0.65,0.70,0.80,1.0,1.3,1.6,2.0,2.5,3.0,3.5,4.0,5.0,6.5,7.5,8.5,10.0,12.5,15.0,17.5,20.0,25.0,30.0,32.0,36.0]) #This is the upper cut point of each channel
        sbl = np.array([0.25,0.28,0.30,0.35,0.40,0.45,0.50,0.58,0.65,0.70,0.80,1.0,1.3,1.6,2.0,2.5,3.0,3.5,4.0,5.0,6.5,7.5,8.5,10.0,12.5,15.0,17.5,20.0,25.0,30.0,32.0,36.0]) #This is the upper cut point of each channel
        self.cutPoint = cpv #Note: this is the 'upper' cut for each bin. 
        self.binEdges = sbl #This is the edges of all of the bins.  
        self.binCenters  = np.divide(np.diff(sbl),2) + sbl[0:-1] 
        self.dlogDp = np.diff(np.log10(sbl))
        self.density=0.0        # Used Density
        self.refractiveIndex = []    # Refractive Index
        self.sizeCorrFac = 0.00      #Size correction factor
        self.flowCal = 1.000         #Flow Calibration
        self.errors = []
        self.PM1=[]
        self.PM10=[]
        self.PM25=[]
        self.PM_Time=[]
        self.gravimetricFactor = 0.0   
        self.Time_HOD = []          
        self.columnNames = [] 
        self.numSamples = 0
        self.elapsedTime = []
        self.rawdata = []
        self.data = []
        self.deadTime = []
        self.alarms = []
        self.Errors = []
        self.dNdlogDp = []
        self.startDateTime = []     
        self.sampleTimes = []        

class TSI:
     
    def __init__(self):
        

        self.flowRate = 16.67 #cubic centimeters per second
        self.instrName=[]
        self.modelNum=[]
        self.serialNum=[]
        self.firmware=[]
        self.calDate=[]
        self.protocolName = ''
        self.testStartTime = ''
        self.Time_HOD = []  
        self.testStartDate = ''
        self.testLength = ''
        self.sampleInterval = ''
        self.numChannels = 0 #Total number of channels including the 'smaller than this' size 
        self.ChannelNum = 0 #The number for each channel
        self.cutPoint = [] #This is the upper cut point of each channel
        self.binEdges = []
        self.alarm = 0.0        #Alarm
        self.density=0.0        # Used Density
        self.refractiveIndex = []    # Refractive Index
        self.sizeCorrFac = 0.00      #Size correction factor
        self.flowCal = 1.000         #Flow Calibration
        self.deadTimeCorr = 1.000    # Deat Time correction factor
        self.errors = []             #Number of samples in this measurement period
        self.columnNames = [] 
        self.numSamples = 0
        self.elapsedTime = []
        self.rawdata = []
        self.data = []
        self.deadTime = []
        self.T = []
        self.RH = []
        self.P = []
        self.alarms = []
        self.Errors = []
        self.binCenters  = [] 
        self.dNdlogDp = []
        self.startDateTime = []     
        self.sampleTimes = [] 


def append_TSI(data_1,data_2):
    from copy import deepcopy
    output_data = deepcopy(data_1)
    output_data.Time_HOD =np.append(output_data.Time_HOD,data_2.Time_HOD)
    output_data.errors =np.append(output_data.errors,data_2.errors)
    output_data.rawdata =np.append(output_data.rawdata,data_2.rawdata,axis=0)
    output_data.data  =np.append(output_data.data ,data_2.data,axis=0)
    output_data.dNdlogDp =np.append(output_data.dNdlogDp,data_2.dNdlogDp,axis=0)
    output_data.startDateTime =np.append(output_data.startDateTime,data_2.startDateTime)
    output_data.TsampleTimes =np.append(output_data.sampleTimes,data_2.sampleTimes) 
    output_data.numSamples = np.sum(output_data.numSamples + data_2.numSamples)
     
    return output_data  
def append_Grimm(data_1,data_2):
    pass    
