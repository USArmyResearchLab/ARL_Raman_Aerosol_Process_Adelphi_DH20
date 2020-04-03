#rs_tools.py
################################################################################
#
#           RS_tools version 0.3.1
#
###############################################################################
#
#
#   Section 1: Classes
#
#   This section contains the Classes that are used in analysis of line-scanning RS data
#   These classes were written in-house, and any internet resources were used as guides
#
#
    
class Parameters():
    import numpy as np
    pass
            
class Collection:
    '''
	This is a collection class, that describes one hyperspectral 'map'.  
    '''
    def __init__(self):
        self.Summary=Summary()
        self.Environment=Environment()
        self.Collector=Collector() 
        self.Image=Image()      
        #self.tape_advance=[]
        #self.find_tape=[]
        #self.identification=[]
        #self.focus=[]
        #self.Image=[]
    
    def build_Summary(self,root,Directory = ''):
        #This builds the summary structure
        self.Summary.System=root.attrib['System']
        self.Summary.Date=root.attrib['Date']
        self.Summary.Image_Start=self.Collector.Start_Time
        self.Summary.Image_End = self.Collector.Date
        self.Summary.ID = self.Collector.ID
        self.Summary.Directory = Directory
        self.Summary.binwn = []

class Environment:
    def __init__(self):
        self.Temperature = []
        self.RelativeHumidity=[]
        self.Date=[]
        self.ID=[]
        
    def load_Environment(self,Info):
        self.Date=Info.attrib['Date']
        self.ID = Info.attrib['ID']  
        for Metric in Info.iter('Metric'):
            if Metric.attrib['Name'] == 'Temperature (C)':
                self.Temperature=float(Metric.text)
                #print float(Metric.text)
            elif Metric.attrib['Name'] == 'Relative Humidity':
                self.RelativeHumidity=float(Metric.text)
                #print float(Metric.text)  

class Summary:
    def __init__(self):
        self.System = []
        self.Date=[]
        self.Imaging_Start=[]
        self.Save_Name=''
        self.Imaging_End=[]
        self.Collection_Start=[]
        self.Collection_End=[]
        self.Identification_Time = []
        self.Collection_Verified = False
        self.Background_Subtracted = False
        self.Cosmic_Removed = False
        self.nCosmic = 0
        self.nSaturation = 0
        self.Saturation_Matrix=[]
        self.nBurning = 0
        self.Spectra_Quality= []
        self.Cleaned = False
        self.QC_applied = False
        self.ID=[]
        self.Directory=[]
        self.Background_Data=[]
        self.Bin_Wavenumber=[]
        self.Error=False
        self.Bleach=False
        
class Collector:
    def __init__(self):
        self.Data_File = []
        self.Start_Time=[]
        self.Average_Current_uamps=[]
        self.Perc_Average_Current_in_Target=[]
        self.Perc_in_Target=[]
        self.Average_Voltage_kV=[]
        self.Last_Ceiling_Voltage_kV=[]
        self.Perc_Voltage_Limited=[]
        self.Quenches=[]
        self.Voltage_Mismatches=[]
        self.Resets=[]
        self.Over_Temps=[]
        self.Date=[]
        self.ID=[]
        
    def load_Collector(self,Info):
        self.Date=Info.attrib['Date']
        self.ID = Info.attrib['ID']  
        for Metric in Info.iter('Metric'):
            if Metric.attrib['Name'] == 'Data File':
                self.Data_File=Metric.text
            elif Metric.attrib['Name'] == 'Start Time':
                self.Start_Time=Metric.text
            elif Metric.attrib['Name'] == 'Average Current (uamps)':
                self.Average_Current_uamps=float(Metric.text) 
            elif Metric.attrib['Name'] == '% Average Current in Target':
                self.Perc_Average_Current_in_Target=float(Metric.text)                
            elif Metric.attrib['Name'] == '% in Target':
                self.Perc_in_Target=float(Metric.text)                                
            elif Metric.attrib['Name'] == 'Average Voltage (kV)':
                self.Average_Voltage_kV=float(Metric.text)
            elif Metric.attrib['Name'] == 'Last Ceiling Voltage (kV)':
                self.Last_Ceiling_Voltage_kV=float(Metric.text)
            elif Metric.attrib['Name'] == '% Voltage Limited':
                self.Perc_Voltage_Limited=float(Metric.text)
            elif Metric.attrib['Name'] == 'Quenches':
                self.Quenches=float(Metric.text)
            elif Metric.attrib['Name'] == 'Voltage Mismatches':
                self.Voltage_Mismatches=float(Metric.text)
            elif Metric.attrib['Name'] == 'Resets':
                self.Resets=float(Metric.text)
            elif Metric.attrib['Name'] == 'Over Temps':
                self.Over_Temps=float(Metric.text)   
                


class Image:
    #One "Image" for every "Image" on the CCD
    def __init__(self):
        #These are the variables read in by the load_Image pogram
        #These are read in by the XML file. 
        self.Date=[] #Date and time of the image
        self.ID=[]
        self.Bleach_Name=[]
        self.Spot_UID=[] #Generally empty, but we have this in case you want a unique collection Hash identifier
        self.x_location = 0.
        self.z_location = 0.        
        self.Replicate_Data=[]
        self.Replicate_Name=[]
        self.Bleach_Data=[]
        self.Target_Data=[]
        self.Fluorescence_Data=[]
        self.Fluorescence_Sum=[]
        self.Fluorescence_Max=[]
        self.Raman_Sum = []
        self.Raman_Max = []
        self.Clip_Ind = []
        self.Replicate_Time = [] #This will tell you about the replicate time
        #These could be used to tell you if QC is performed on the image.
        self.Cleaned=False
        self.Background=False
        self.Fluorescence=False
        self.Good = True 
    
################################################################################
################################################################################
################################################################################
#
#
#   Section 2: Loading/Outputting Data Files
#
#
#
    
def load_summary(directory_load,QC=False,QC_Dir = '',QC_Name='colldata.txt'):         
    import glob
    import os
    import numpy as np
    from pandas import Timedelta
    from re import sub
    import ntpath
    import pdb
    from pandas import to_datetime
    
    #pdb.set_trace()    
    allfile2load = glob.glob(os.path.join(directory_load,'*summary.txt'))
    

    if len(allfile2load) == 1:
        file2load = allfile2load[0]
        myRebs = Collection()
        myRebs.Image = []    
        #myRebs.Summary.Background_Data=load_bkr(directory_load)
     
        #Loop through the 'info' categories, fill xml data as required
        #myRebs[i].Summary.load_Summary(root.iter('Summary'))
            
        with open(file2load,'r') as f:
            #Load the header
            stl = f.readline()
            #Get the start date in 'nice' string format
            stl = f.readline().strip().split(' ')#Contains the date in nice 'string format'
            myRebs.Summary.Save_Name = stl[3]
            f.readline() #Blank Line in file
            f.readline() #Run Notes
            f.readline() #Machine
            f.readline() #X steps
            f.readline() #X step size
            f.readline() #z range
            f.readline() #Timage - t9ime of each image
            f.readline() #Number of replicates

            
            #Get start time
            stl=f.readline().strip().split(' ') #Start Time
            try:
                myRebs.Summary.Imaging_Start=to_datetime(stl[3] + ' ' + stl[4] + ' ' + stl[5])   
            except:
                myRebs.Summary.Imaging_Start=to_datetime(stl[-3] + ' ' + stl[-2] + ' ' + stl[-1])
            f.readline() #Number of replicates
            f.readline() #Number of replicates
            
            #Now we have finished loading the header.             
            ##########
            #
            #   NOw step through and load the images
            #
            my_img_num = -1
            #mlp = 0
            for line in f:
                #print 'MLP:' + str(mlp)
                #mlp = mlp + 1
                line_seperated = line.split()
                if len(line_seperated) == 0:
                    continue
                elif line_seperated[0] == 'Event':
                    pass
                elif line_seperated[0] == 'Quality':
                    pass
                elif line_seperated[0] == '#---------------------------------------------':
                    break        
                elif line_seperated[0] == 'Image':
                    myRebs.Image.append(Image())
                    my_img_num = my_img_num + 1
                    print(my_img_num)
                    locs = line_seperated[2].strip('()').split(',')
                    myRebs.Image[my_img_num].x_location = float(locs[0])
                    myRebs.Image[my_img_num].z_location = float(locs[1])
                    replicate_ind=0
                elif line_seperated[0] == 'Replicate':
                    path_load = os.path.join(directory_load,line_seperated[-1])
                    if replicate_ind == 0:
                        #First replicate - just output to matrix
                        try:
                            newrd = load_rebspng(path_load)
                        except:
                            print(("Error in Load_Summary on",path_load))
                            print(("Image Number:",my_img_num+1))
                            print("I am going to continue, but note that you will have one less replicate")
                            myRebs.Image[my_img_num].Good=False
                            continue
                            
                        myRebs.Image[my_img_num].Replicate_Data = newrd
                        myRebs.Image[my_img_num].Replicate_Name = [line_seperated[-1]]
                        
                        #extract the replicate time

                        try:
                            myRebs.Image[my_img_num].Replicate_Time = line_seperated[2].split('(')[1].strip('(s):')
                        except:
                            myRebs.Image[my_img_num].Replicate_Time = line_seperated[1].split("(")[1].strip(")")
                        #Set replicate_ind to 1 indicating that we have more replicates
                        replicate_ind = 1
                    else: 
                        oldrd = myRebs.Image[my_img_num].Replicate_Data
                        try:
                            newrd = load_rebspng(path_load)
                        except:
                            print(("Error in Load_Summary on",path_load))
                            print(("Image Number:",my_img_num))
                            print("I am going to continue, but note that you will have one less replicate")
                            myRebs.Image[my_img_num].Good=False
                            continue
                        myRebs.Image[my_img_num].Replicate_Data = np.dstack((oldrd,newrd)) 
                        myRebs.Image[my_img_num].Replicate_Name.append(line_seperated[-1])
                else:
                    pass        
        myimsize = myRebs.Image[0].Replicate_Data.shape
        #import pdb
        #pdb.set_trace()
        myRebs.Summary.Spectra_Quality=np.zeros((my_img_num,int(myimsize[0])))
        return myRebs
                
    else:
        raise IOError(12,'NoFileFound')
        return
        #'File not Found','noFile',os.path.join(directory_load,'*SpotInfo.txt'))         
                
def load_spotinfo(directory_load,QC=False,QC_Dir = '',QC_Name='colldata.txt'):         

    import glob
    import os
    import numpy as np
    from pandas import Timedelta
    from re import sub
    import ntpath
    
   
    allfile2load = glob.glob(os.path.join(directory_load,'*SpotInfo.txt'))

    if len(allfile2load) == 1:
        file2load = allfile2load[0]
        myRebs = Collection()
        myRebs.Image = []    
        myRebs.Summary.Background_Data=load_bkr(directory_load)
    
        myRebs.Summary.Imaging_Start=convert_rebspath_to_datetime(directory_load)
        
        myRebs.Summary.Save_Name = myRebs.Summary.Imaging_Start.strftime('%Y%m%d_%H%M%S')

            
        with open(file2load,'r') as f:
            
            #currentStatus = 'data'
            stl = f.readline
            print(f)
            #Read first six lines
            stl = f.readline().strip().split('--')
            f.readline() #Blank Line in file
            f.readline() #This is session ID (throwing out for now)
            cidl = f.readline().strip().split('=') #This is Collection ID 
            f.readline() #This is spot ID (throwing out for now
            f.readline() #Blank Line
            
            if stl[0] == '' and cidl[0] == '':
                #IN this case, the spotinfo exists, but was not written to (e.g. power failure). 
                myBackupRebs = load_rebsdirectory(directory_load)
                #raise IOError(11,'REBS Shutdown') 
                return myBackupRebs

            my_img_num = -1
            #print os.path.basename(file2load)
            myRebs.Collector.Data_File = os.path.basename(os.path.dirname(file2load))
            
            for line in f:
                
                line_seperated = line.split()
                if len(line_seperated) == 0:
                    continue
                elif line_seperated[0] == 'Targeting':
                    # For Every Targeting Data, Append a new image
                    myRebs.Image.append(Image())
                    my_img_num = my_img_num + 1
                    replicate_ind = 0
                    
                    myRebs.Image[my_img_num].x_location = my_img_num*2
                    
                    # Set up the Path Name to Use                
                    path_load = convert_rebspath_to_computerpath(line,directory_load)
                    myRebs.Image[my_img_num].Target_Data = load_rebspng(path_load)
                    
                elif line_seperated[0] == 'Bleaching':
                    path_load = convert_rebspath_to_computerpath(line,directory_load)
                    myRebs.Image[my_img_num].Bleach_Data = load_rebspng(path_load)
                    myRebs.Image[my_img_num].Bleach_Name = ntpath.basename(path_load)
                
                elif line_seperated[0] == 'Identification':
                    path_load = convert_rebspath_to_computerpath(line,directory_load)
                    if replicate_ind == 0:
                        #First replicate - just output to matrix
                        myRebs.Image[my_img_num].Replicate_Data = load_rebspng(path_load) 
                        myRebs.Image[my_img_num].Replicate_Name = [ntpath.basename(path_load)]
                        replicate_ind = 1
                    else:
                        
                        oldrd = myRebs.Image[my_img_num].Replicate_Data
                        myRebs.Image[my_img_num].Replicate_Data = np.dstack((oldrd,load_rebspng(path_load))) 
                        myRebs.Image[my_img_num].Replicate_Name.append(ntpath.basename(path_load))
                        #Concatenate these two matrices together
                elif "_".join(line_seperated[4:7]) == 'Requested_identification_time':
                    myRebs.Summary.Identification_Time = float(sub('[()]', '', line_seperated[7]))
                    myRebs.Summary.Imaging_End = myRebs.Summary.Imaging_Start + Timedelta(myRebs.Summary.Identification_Time,'s')
                elif "_".join(line_seperated[3:6]) == 'Spot_Identification_Ended.':
                    #end_time = line.split('--')[0]
                    pass
                        
                
                if line.strip() == '-- SUMMARY --':
                    break
                    
                    
            for line in f:
                #Loop through Summary
                if line.strip() == '-- Totals --':
                    break
                    
            for line in f:
                #Loop through Summary     
                pass
        
        if len(myRebs.Image) == 0:
            #Currently an IO Error. 
            #Someday we may change this to a 'warning'
            raise IOError(11,'NoImagesFound') 
            

        myimsize = myRebs.Image[0].Replicate_Data.shape   
        myRebs.Summary.Spectra_Quality=np.zeros([myimsize[0],my_img_num+1])            
        return myRebs
                
    else:
        raise IOError(12,'NoFileFound')
 
           
def load_rebsdirectory(input_directory):
    import glob
    import os
    import ntpath
    import numpy as np

    
    myRebs = Collection()
    myRebs.Image = []    
    myRebs.Summary.Background_Data=load_bkr(input_directory)
    myRebs.Summary.Imaging_Start=convert_rebspath_to_datetime(input_directory)
    #Add to note that there is an error. 
    myRebs.Summary.Error = True
    
    print('Using Simple REBS loader....caution is indicated. May break other code')
    myRebs.Collector.Data_File = os.path.basename(input_directory)
    my_img_num = -1
            
    search_path = os.path.join(input_directory,'REBS RN*.png')        
    
    all_files = glob.glob(search_path)
    
    for myfile in all_files:
        
        if 'bkr' in os.path.basename(myfile):
            continue
        elif 'bkt' in os.path.basename(myfile):
            continue
        elif 'bkrs' in os.path.basename(myfile):
            continue
        elif 'bkts' in os.path.basename(myfile):
            continue
        elif '_t' in os.path.basename(myfile):        
            #We have a new target. 
            myRebs.Image.append(Image())
            my_img_num = my_img_num + 1
            replicate_ind = 0
            #print(myfile)
        elif '_b' in os.path.basename(myfile):
            #print(myfile)
            myRebs.Image[my_img_num].Bleach_Data = load_rebspng(myfile)
            myRebs.Image[my_img_num].Bleach_Name = ntpath.basename(myfile)   
        elif '_r' in os.path.basename(myfile):
            #print(myfile)
            if replicate_ind == 0:
                #First replicate - just output to matrix
                myRebs.Image[my_img_num].Replicate_Data = load_rebspng(myfile) 
                myRebs.Image[my_img_num].Replicate_Name = [ntpath.basename(myfile)]
                replicate_ind = 1
            else:
                
                oldrd = myRebs.Image[my_img_num].Replicate_Data
                myRebs.Image[my_img_num].Replicate_Data = np.dstack((oldrd,load_rebspng(myfile))) 
                myRebs.Image[my_img_num].Replicate_Name.append(ntpath.basename(myfile))
        
    return myRebs 

def load_bkr(mydir='',clean=True):
    #    Given a directory, finds the background file, and loads it
    #    if no directory, uses current directory
    #    To manually load background file, just use load_rebspng    
    #
    import glob
    import numpy as np
    import os
    import ntpath
    #Search for a file in the path that is a png with *bkr* in the name
    mysearchpath = os.path.join(mydir,'*_bkr.png')
    bkrfilename = glob.glob(mysearchpath)
    print(bkrfilename)
    #Check if we found anything
    if not bkrfilename:
        #If not, print error, return NaN
        print('No BKR file found')
        return np.NaN
    else:
        #Otherwise, check if we only found one file
        if len(bkrfilename) == 1:
            #If only one file, add the path to the filename
            bkrfile2load = bkrfilename[0]
            #Load the file
            bkr = load_rebspng(bkrfile2load)
            return bkr
        else:
            #Otherwise, there are mutiple files
            print('Mutiple files Found, be more specific')
            return np.NaN  

def load_rebspng(myfile):
    #    Loads the raw REBS png as a file and returns the data
    #    Needs to be a full directory path and filename
    import matplotlib.image as mpimg
    import numpy as np
    try:
        myimg_rebs = mpimg.imread(myfile)
        return myimg_rebs
    except IOError as e:
        print(myfile)
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
        raise e
    except:
        print("Non-IO error")
        raise
        

def cl_output_sep(f_data,f_meta,nrtd,collection_num = 0):
    """ exports seperated data from the collection file. This means that individual replicates are retained
    replicate number is included in the meta file. (So be very careful to analyze correctly)
    inputs:
        f_data: this is a file handle pointing to the data file
        nrtd: this is a collection class containing the data from one collectionimport numpy as np
        f_meta: this is a meta file containing the metadata from the collection.
        collection_num: this is the number of collection. If you are analyzing a lot of collections, this could be nice to have
            as a simple way to seperate out the collections. defaults to zero
            
    outputs: 
        there are no outputs
    usage:
        XXXXXX
    """
    import numpy as np
    
    num_rows,num_wns,num_replicates = nrtd.Image[0].Replicate_Data.shape
    
    for i,Image in enumerate(nrtd.Image):

        for j in range(0,num_rows):
    
            for k in range(0,num_replicates):
                #Get the values to be written
                t = nrtd.Summary.Imaging_Start
                x = Image.x_location
                z = Image.z_location
                y = j
                rep = k
                cn = collection_num
                p1 = 0.0
                p2 = 0.0
                p3 = 0.0
                p4 = 0.0
                p5 = 0.0
                
                if len(Image.Fluorescence_Sum) > 0: 
                    Fluo = Image.Fluorescence_Sum[j,k]
                    Flt = Image.Total_Sum[j,k]
                else:
                    Fluo = 0.0
                    Flt = 0.0
                
                mydata = nrtd.Image[i].Replicate_Data[j,:,k]  
                
                #Write Metadata
                #f_meta.write('DateTime,x,y,z,r,cn,fl,flt,p1,p2,p3,p4,p5\n')
                f_meta.write(str(t))
                f_meta.write(',%03d,%03d,%03d,%03d,%03d,%1.4f,%1.4f,%03d,%03d,%03d,%03d,%03d' % (x,y,z,rep,cn,Fluo,Flt,p1,p2,p3,p4,p5))
                f_meta.write('\n')  
                #write the rest + newline
                datastr = '%1.4f' + ',%1.4f'*1023 
                
                #Interpolate NaN values
                try:
                    nans,x=nan_helper(mydata)
                    mydata[nans] = np.interp(x(nans),x(~nans),mydata[~nans])
                    f_data.write(datastr % tuple(mydata.tolist()))
                    f_data.write('\n') 
                except:
                    f_data.write(datastr % tuple(mydata.tolist()))
                    f_data.write('\n')    
                                                                              
#rt.dc_output_sep(f_data,dc,f_meta,t,dx,dz,collection_num=0,Fluo=fl,Flt=ft)                                                                                                                                                                                                                                       
def dc_output_sep(f_data,dc,f_meta,t,dx,dz,collection_num=0,Fsum=[0],Fmax=[0],Rmax=[0],Rqc=[0],dump_nans=False):
    import numpy as np
    #Outputs the collection
    #This is called 'sep' becuase it outputs location data
    # seperately from 'meta' data
    #Primarily used for (for example) getting data into hyperspec format. 
    # Note vor rebstools v21 and up: there was a major change in the computation and order of some of these
    #values. For example, the x_position is no longer determined via strtX, but rather via dx.
    #To maintain continuity, we also add a value for 'replicate number' - for this scenario the replicate number will always be 'zero'
    
    
    (n_rows,n_columns,n_images) = dc.shape 
    
    for i in range(0,n_images):
        for j in range(0,n_rows):
            
            #t = nrtd.Summary.Imaging_Start
            x = dx[i]# this is the image location
            y = j
            z = dz[i]
            #z: this is the image height
            rep = 0 #Set to zero because we are analyzing the 'dir cube'
            cn = collection_num
            
            #These are placeholders
            p3 = 0.0
            p4 = 0.0
            p5 = 0.0

            Fl = Fsum[j,i] if len(Fsum)> 1 else 0.0
            Fx = Fmax[j,i]  if len(Fmax) > 1 else 0.0
            Rx = Rmax[j,i]  if len(Rmax) > 1 else 0.0            
            qc = Rqc[j,i] if len(Rqc) > 1 else 0.0                                
            #Write the data
            datastr = '%1.4f' + ',%1.4f'*1023 
            #Interpolate NaN values
            mydata = dc[j,:,i]
            
            try:
                nans,ind=nan_helpfcn(mydata)
                mydata[nans] = np.interp(ind(nans),ind(~nans),mydata[~nans])
                #note: This will throw an exception if you have all nan's
                f_data.write(datastr % tuple(mydata.tolist()))
                f_data.write('\n') 
                dump_meta = True #Everything worked, we need to output the metadata
                #Write the date
            except:
                #Generally this meanss that there was a problem interpolating NaN Values. 
                #IN this case, the dump_nan boolean will help us decide if we want to dump our NaN values or not 
                if dump_nans == True:
                    print("Error in nan_helper at",i,j, "trying normal data write")
                    f_data.write(datastr % tuple(mydata.tolist()))
                    f_data.write('\n') 
                    dump_meta = True #We still need to output the appropriate metadata file
                else:
                    print("Error in nan_helper at",i,j,"skipping")
                    dump_meta = False #Since we skipped dumping the datafile, we do not want to dump the metadata for the corresponding data point. 
                    
            if dump_meta == True:
                f_meta.write(str(t))
                #import pdb
                #pdb.set_trace()
                try:
                    f_meta.write(',%03d,%03d,%03d,%03d,%03d,%1.5f,%1.5f,%1.5f,%03d,%03d,%03d,%03d' % (x,y,z,rep,cn,Fl,Fx,Rx,qc,p3,p4,p5))
                except:
                    import pdb
                    pdb.set_trace()
                    print("Problem with meta file at",i,j,"writing -999 for all parameters")
                    f_meta.write(',%03d,%03d,%03d,%03d,%03d,%1.5f,%1.5f,%1.5f,%03d,%03d,%03d,%03d' % (-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999))
                f_meta.write('\n')   
                      
                      
def dc_output_sep_fl(f_data,dc,f_fluo,fc,f_meta,t,dx,dz,collection_num=0,Fsum=[0],Fmax=[0],Rmax=[0],Rqc=[0],dump_nans=False):
    import numpy as np
    #Outputs the collection
    #This is called 'sep' becuase it outputs location data
    # seperately from 'meta' data
    #Primarily used for (for example) getting data into hyperspec format. 
    # Note vor rebstools v21 and up: there was a major change in the computation and order of some of these
    #values. For example, the x_position is no longer determined via strtX, but rather via dx.
    #To maintain continuity, we also add a value for 'replicate number' - for this scenario the replicate number will always be 'zero'
    
    
    (n_rows,n_columns,n_images) = dc.shape 
    
    for i in range(0,n_images):
        for j in range(0,n_rows):
            
            #t = nrtd.Summary.Imaging_Start
            x = dx[i]# this is the image location
            y = j
            z = dz[i]
            #z: this is the image height
            rep = 0 #Set to zero because we are analyzing the 'dir cube'
            cn = collection_num
            
            #These are placeholders
            p2 = 0.0
            p3 = 0.0
            p4 = 0.0
            p5 = 0.0

            Fl = Fsum[j,i] if len(Fsum)> 1 else 0.0
            Fx = Fmax[j,i]  if len(Fmax) > 1 else 0.0
            Rx = Rmax[j,i]  if len(Rmax) > 1 else 0.0            
            qc = Rqc[j,i] if len(Rqc) > 1 else 0.0                                
                                            
            #Write the data
            datastr = '%1.4f' + ',%1.4f'*1023 
            #Interpolate NaN values
            mydata = dc[j,:,i]
            mydata_fl = fc[j,:,i]
            
            try:
                nans,ind=nan_helpfcn(mydata)
                
                mydata[nans] = np.interp(ind(nans),ind(~nans),mydata[~nans])
                #note: This will throw an exception if you have all nan's
                f_data.write(datastr % tuple(mydata.tolist()))
                f_data.write('\n') 
                dump_meta = True #Everything worked, we need to output the metadata
                #Write the date
            except:
                #Generally this meanss that there was a problem interpolating NaN Values. 
                #IN this case, the dump_nan boolean will help us decide if we want to dump our NaN values or not 
                if dump_nans == True:
                    print("Error in nan_helpfcn at",i,j, "trying normal data write")
                    f_data.write(datastr % tuple(mydata.tolist()))
                    f_data.write('\n') 
                    dump_meta = True #We still need to output the appropriate metadata file
                else:
                    print("Error in nan_helpfcn at",i,j,"skipping")
                    dump_meta = False #Since we skipped dumping the datafile, we do not want to dump the metadata for the corresponding data point. 
                    
            f_fluo.write(datastr % tuple(mydata_fl.tolist()))
            f_fluo.write('\n') 
            
            if dump_meta == True:
                f_meta.write(str(t))
                #import pdb
                #pdb.set_trace()
                try:
                    f_meta.write(',%03d,%03d,%03d,%03d,%03d,%1.5f,%1.5f,%1.5f,%03d,%03d,%03d,%03d' % (x,y,z,rep,cn,Fl,Fx,Rx,qc,p3,p4,p5))
                except:
                    import pdb
                    pdb.set_trace()
                    print("Problem with meta file at",i,j,"writing -999 for all parameters")
                    f_meta.write(',%03d,%03d,%03d,%03d,%03d,%1.5f,%1.5f,%1.5f,%03d,%03d,%03d,%03d' % (-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999))
                f_meta.write('\n')   
                      
                                      
def output_Rdata_single(dc,dx,dz,t,path_to_output='',x_values=None,Output_Name=''):
    ''' The purpose of this code is to output a single collection in hyperspec format
    the point is if you just need to do something quickly
    inputs:
        dc,dx,dz,t : these are the outputs by process_collection
        path_to_output (optional): this is the output location. Default: home directory
        x_values: what are the bin wavenumber values. default: Rn14 values
        output_name: what do you want the output data file to be
    outputs:
        none
    '''
    import os
    
    if x_values is None:
        x_values = get_rebs_calibration(rn=14)
    
    #Dumps a single dircube to an R file (appropriately named)
    fname_data = t.strftime('%Y%m%d_%H%M%S') + '.rdat'
    fname_meta = t.strftime('%Y%m%d_%H%M%S') + '.rmta'            
    path_data = os.path.join(path_to_output,fname_data)
    path_meta = os.path.join(path_to_output,fname_meta)
    #
    with open(path_data,'w') as f_data, open(path_meta,'w') as f_meta:
        datastr = '%1.4f' + ',%1.4f'*1023 + '\n' 
        f_data.write(datastr % tuple(x_values))
        f_meta.write('DateTime,x,y,p1,p2\n')
        dc_output_sep(f_data,dc,f_meta,t,dx,dz)   
        
                                                                                                                    
################################################################################
################################################################################
################################################################################
#
#
#   Section 3: Processing Data
#
#
#
#
#
    
def clean_collection(nrtd):
    from copy import deepcopy
    import numpy as np
    #This handles a case where you had a Target image taken
    #But are missing a replicate data
    nurtd = deepcopy(nrtd)
    nurtd.Image = [Image for Image in nrtd.Image if np.size(Image.Replicate_Data)>1]
    return nurtd 

def remove_saturation(nrtd,st=10,saturation_limit=0.99):
    print("No longer used...saturation is removed (if desired) after QC Spectrum")
    print("Please use detect_saturation() and qc_spectrum() to remove saturated spectra")
    print("I am running detect_saturation instead....")
    nrtd = detect_saturation(nrtd,st,saturation_limit)
    return nrtd
   
def detect_saturation(nrtd,st=10,saturation_limit=0.99):
    """ this detects
    IMPORTANT NOTE: This relies on the maximum value being 1 and minimum zero. 
    If you use another loader or have other values (e.g. you have  a shorter bleach spectrum
    Then make sure you run this analysis prior to running "use_bleach". 
    Note: 11/2018: Note: Previous iterations of this code (remove_saturation()) removed saturation
                        Current version does not remove saturation, it just sets spectra_quality to 1.  
                        Future iterations might show the number and location of saturations (so one could select to use non-saturated spectra). .  
    Notes:
        -This code removes saturations in the bleach spectra as well as in replicate spectra
        - For counting num_saturation - this code *ONLY* currently counts the number of saturations 
        -This code works for n replicates
    Inputs:
        nrtd: this is a collection class. It should have been already loaded
        st: this is the saturation tolerance....i.e. the number of points 
            that can reach saturation before removing that spectra. 
            Reccomend at least higher than 1 so that a few saturation peaks
            won't result in removal of the entire spectra"
    Usage:
        myCollection = remove_saturation(myCollection,st=5)
    This assumes that the data have been normalized
    """
    import numpy as np

    saturation_tolerance=st
    
    num_saturation = 0
    satmat = np.empty((0,5))

        
    for i,Image in enumerate(nrtd.Image):
        len_reps = Image.Replicate_Data.shape[2]
        if len(Image.Bleach_Data) > 0:
            #If we have bleach data
            bleach_data = Image.Bleach_Data                 
            bd = bleach_data[:,100::] >= saturation_limit
            sv = np.nansum(bd,axis=1)
            badrows = np.where(sv>saturation_tolerance)
            nrtd.Summary.Spectra_Quality[badrows,i] = 1 
        for j in range(0,len_reps):
            rv=j
            bd = Image.Replicate_Data[:,100::,j] >= saturation_limit
            sv = np.nansum(bd,axis=1)
            badrows = np.where(sv>saturation_tolerance)
            
            # added 5/2017 - count saturation
            num_saturation = num_saturation + sum(sv>saturation_tolerance)
            if len(badrows[0]>0):
                try:
                    for k,mybadrow in enumerate(badrows[0]):
                        rowvs = np.array([-999,i,mybadrow,rv,sv[mybadrow]])
                        satmat = np.vstack((satmat,rowvs))
                except:
                    import pdb
                    print("Error in detect_saturation")
                    pdb.set_trace()
                
            # added 5/2017 - remove all replicates
            #Image.Replicate_Data[badrows,:,:] = np.NaN
            #Set the image quality to 1 for 'saturated'
            nrtd.Summary.Spectra_Quality[badrows,i] = 1
            #Image.Replicate_Data[badrows,:,j] = np.NaN
            
    nrtd.Summary.NSaturation = num_saturation
    nrtd.Summary.Saturation_Matrix=satmat
    
    return nrtd
    
def clean_badlocs(nrtd,rn=0):
    """ All the different Rebs instruments have problem pixels. 
    Maybe something is on the CCD (and should be cleaned)? Some pixels 
    are 'hot'. In any case, we use this to remove the pixels that clearly 
    are bad. We go throuhgh and manually determine the bad locations. 
    Notes:
        -This code analyzes the bleached data!
        -This code works for n replicates
        - This code deletes the first 50 pixels
    Inputs: 
        nrtd: This is a collection class
        rn: this is the ars serial number, which tells you which set of 
            bad locations to use
    Outputs: 
        nrtd: this is a collection class
    Usage:
        >>> myCollection = clean_badlocs(myCollection,rn=13)
    """
    #Cleans known bad pixels
    #From arrays
    import sys
    import numpy as np
    if rn == 14:
        #Originally the size here was 38
        #Modified to 36- may need to check date 
        allbadlocs = np.zeros((36,1024),dtype=bool)
        allbadlocs[:,0:51] = True #This is a modification. Remove all points below ~150
        allbadlocs[0,list(range(813,814))]  = True
        allbadlocs[2,list(range(974,981))]  = True
        allbadlocs[3,list(range(974,981))]  = True
        allbadlocs[7,list(range(968,976))]  = True
        allbadlocs[8,list(range(970,976))]  = True
        allbadlocs[10,list(range(943,952))]  = True
        allbadlocs[11,list(range(234,240)) + list(range(943,952))]  = True
        allbadlocs[12,813]  = True
        allbadlocs[16,list(range(970,980))] = True
        allbadlocs[17,list(range(966,980))] = True
        allbadlocs[22,list(range(779,790))] = True
        allbadlocs[23,list(range(779,790))] = True
        allbadlocs[24,list(range(542,549)) + list(range(780,790)) ] = True
        allbadlocs[25,list(range(542,550)) + list(range(780,788)) ] = True
        allbadlocs[26,list(range(542,551))] = True
        allbadlocs[27,list(range(542,552))] = True
        allbadlocs[28,list(range(542,552))] = True        
        allbadlocs[34,847] = True
    elif rn == 13:
        allbadlocs = np.zeros((44,1024),dtype=bool)
        allbadlocs[:,0:51] = True #This is a modification. Remove all points below ~150 cm -1
        allbadlocs[0,list(range(506,523))]  = True  
        allbadlocs[1,list(range(506,525))+list(range(750,751))]  = True    
        allbadlocs[2,list(range(506,530))]  = True    
        allbadlocs[3,list(range(506,525))]  = True
        allbadlocs[4,list(range(424,436))+list(range(510,520))]  = True
        allbadlocs[5,list(range(251,252)) + list(range(424,436))+list(range(510,520))]  = True
        allbadlocs[6,list(range(424,436))+list(range(510,520))]  = True
        allbadlocs[7,list(range(424,436))+list(range(510,520)) + list(range(251,252))]  = True
        allbadlocs[8,list(range(122,123)) + list(range(424,436))+ list(range(511,517))]  = True
        allbadlocs[9,list(range(251,252)) + list(range(511,517))]  = True
        allbadlocs[10,list(range(511,517))]  = True
        allbadlocs[11,list(range(511,517))]  = True
        allbadlocs[12,list(range(511,517))]  = True
        allbadlocs[13,list(range(481,490))+ list(range(511,517))]  = True  
        allbadlocs[14,list(range(481,490))+ list(range(511,517))]  = True  
        allbadlocs[15,list(range(481,490))+ list(range(511,517))]  = True
        allbadlocs[16,list(range(119,122)) + list(range(481,490)) + list(range(511,517))]  = True
        allbadlocs[17,list(range(119,122))+ list(range(481,490))+ list(range(511,517))]  = True
        allbadlocs[18,list(range(119,122)) + list(range(481,490))+ list(range(511,517))]  = True
        allbadlocs[19,list(range(119,122)) + list(range(481,490))+ list(range(511,517))]  = True    
        allbadlocs[20,list(range(480,492))]  = True 
        allbadlocs[21,list(range(474,490))]  = True 
        allbadlocs[22,list(range(474,489))]  = True 
        allbadlocs[23,list(range(474,488))]  = True
        allbadlocs[24,list(range(474,488))]  = True
        allbadlocs[25,list(range(350,351)) + list(range(474,488))]  = True #Commented this out - we think this might be real - this addition of 350-353 was 'new'- possibly a resonance raman?
        #allbadlocs[25,list(range(474,488))]  = True
        allbadlocs[28,list(range(135,136))]  = True
        allbadlocs[31,list(range(350,351))]  = True #Commented this out - we think this might be a real particle - this has been here for a long time
        allbadlocs[36,list(range(146,147))]  = True
        allbadlocs[35,list(range(146,147))]  = True 
        allbadlocs[37,list(range(350,351))]  = True #Commented this out - we think this might be a real raman peak - this has been here for a long time
        allbadlocs[42,list(range(350,351))]  = True #Commented this out - we think this might be a real raman peak - this has been here for a long time
        allbadlocs[43,list(range(924,925))]  = True
    elif rn == 11:
        allbadlocs = np.zeros((37,1024),dtype=bool)
        allbadlocs[7,list(range(237,238)) ] = True
        allbadlocs[8,list(range(235,239)) ] = True
        allbadlocs[9,list(range(235,239)) ] = True
        allbadlocs[10,list(range(235,240)) ] = True
        allbadlocs[11,list(range(236,239)) ] = True
        allbadlocs[13,list(range(477,482)) ] = True
        allbadlocs[14,list(range(476,483)) ] = True
        allbadlocs[15,list(range(474,485)) ] = True
        allbadlocs[16,list(range(474,486)) ] = True
        allbadlocs[17,list(range(474,486)) ] = True
        allbadlocs[18,list(range(475,486)) ] = True
        allbadlocs[19,list(range(482,487)) ] = True
        allbadlocs[20,list(range(482,487)) ] = True
        allbadlocs[21,list(range(483,486)) + list(range(663,667)) ] = True        
        allbadlocs[22,list(range(662,668)) ] = True
        allbadlocs[23,list(range(662,668)) ] = True
        allbadlocs[24,list(range(662,668)) ] = True
        allbadlocs[25,list(range(662,668)) ] = True
        allbadlocs[26,list(range(662,668)) ] = True
        allbadlocs[27,list(range(662,668)) ] = True        
        allbadlocs[28,list(range(663,667)) ] = True
      
    else: 
        print('no RN specified - Could not clean bad locations (clean_badlocs)')
        return nrtd
    
    for i,Image in enumerate(nrtd.Image):
        #print i
        if len(Image.Bleach_Data) > 0:
            #If we have bleach data, remove the bad bleach locations
            nrtd.Image[i].Bleach_Data[allbadlocs] = np.NaN
            try:
                nrtd.Image[i].Bleach_Data = np.apply_along_axis(lininterp_nan,1,nrtd.Image[i].Bleach_Data)
            except Exception as e:
                #Error, likely row removed due to saturation
                print(('Clean_badlocs: Error with interp ',i,'b'))
                print(e)
                nrtd.Image[i].Bleach_Data[:,0] = 0
                nrtd.Image[i].Bleach_Data[np.where(nrtd.Image[i].Bleach_Data[:,1023]==np.nan),1023] = 0
                try:
                    nrtd.Image[i].Bleach_Data = np.apply_along_axis(lininterp_nan,1,nrtd.Image[i].Bleach_Data) 
                except:
                    print(("Still oculdn't fix error, (blc) removing all data",i))
                    print(sys.exc_info()[0])
                    nrtd.Image[i].Bleach_Data = np.zeros(nrtd.Image[i].Bleach_Data.shape)
              
        num_replicates = Image.Replicate_Data.shape[2]
        for j in range(0,num_replicates):
            #print j
            rdata = nrtd.Image[i].Replicate_Data[:,:,j]
            rdata[allbadlocs] = np.NaN
            try:
                nrtd.Image[i].Replicate_Data[:,:,j] = np.apply_along_axis(lininterp_nan,1,rdata)
            except Exception as e:
                print(("Clean_badlocs: Error with interp",i,j))
                print(e)
                rdata[:,0] = 0
                rdata[np.where(rdata[:,1023]==np.nan),1023] = 0
                try:
                    nrtd.Image[i].Replicate_Data[:,:,j] = np.apply_along_axis(lininterp_nan,1,rdata) 
                except ValueError as ve:
                    print(("Still oculdn't fix error, (rep) removing all data",i))
                    print(sys.exc_info()[0])
                    print(ve)
                    nrtd.Image[i].Replicate_Data[:,:,j] = np.zeros(nrtd.Image[i].Replicate_Data[:,:,j].shape)              
    return nrtd

        
def use_bleach(nrtd,bbm=1):
    """ This turns a 'bleach' into a 'replicate'. In the old code, bleach and replicates
    were treated seperately through the whole processing period, and then brought together
    at the very end. This didn't make any sense and complicated the analysis process. 
    So this code adds 'bleach_data' onto 'replicate data, essentially turning the bleach
    into replicate 0.  
    ALL OF THE CODES BELOW THIS ONE DO NOT ANALYZE BLEACH DATA!!
    inputs:
        nrtd: this is a myCollection
        bbm: bleach bkr multiplier. If the bleach is shorter than the replicates, what do you have to multiply the bleach by os that
            it has the same 'equivalent' time period as the replicates. Default: 1 (No mulitplier) 
            For example, if the replicate time is 10 seconds, and the bleach time is 5 seconds,
            bbm should be 2
    """
    import numpy as np
    for i,Image in enumerate(nrtd.Image):
        nrtd.Image[i].Replicate_Data = np.dstack((Image.Bleach_Data*bbm,nrtd.Image[i].Replicate_Data))
    return nrtd    
            
def compute_bkr_collection(myCollection,percentile=10,make_images=False,image_name=''):
    """ Computes a synthetic background value for a given collection, based on
    the lowest values at each point for each image. it treats row (long axis of laser line)
    and wavenumber seperately.
    Notes: 
        -Does NOT consider bleach! If you want to include the 'bleach' in this analysis, you should run the 'use_bleach' code
    inputs: 
        myCollection: This is your collection data. Should be Collection class
        percentile: this is the lower percentile you wish to treat as the 'background'. 
            default:10%
        make_images: this tells this code to dump images of the BKR. 
            default: False
        image_name = this is prepended onto the image filenames if make_image
            is set to true.
            default: ''
    outputs:
        bkr_values: ths is a nxm matrix where n is the number of pixels along the
            laser long axis and m is the number of bins in the wavenumber dimension. 
            This value correspoinds to the "synthetic" background values. 
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from copy import deepcopy
    #GEt a dircube. Currently only use the 'replicates'
    #
	#	There was an issue here but it is fixed now.
	#

    dc00,dx,dz,t,fl,ft,fm = collection_process(myCollection,method='avg')
    num_rows,num_wns,num_images = dc00.shape
    pctl = np.nanpercentile(dc00,percentile,axis=2)   
    nudc_pctl = deepcopy(dc00)
    
    for j in range(0,num_images):
        #Make all values here equal to NaN
        myv = nudc_pctl[:,:,j]
        myv[np.where(myv > pctl)] = np.NaN
           
    bkr_values = np.nanmean(nudc_pctl,axis=2)

    #Should we output figures?
    if make_images==False:
        return bkr_values
    else:
        for myn in range(0,num_rows):
        
            #Name to use to save the data        
            savename_fign = image_name + '_' + str(myn)         
            
            mydcv = dc00[myn,:,:]
            
            #####
            ##
            ## THis shows a list of sorted values at various locations
            ##
            #####
            #THis could be used to test different values
            plt.subplot(2,1,2)
            plt.plot(np.diff(np.sort(mydcv[111,:]))) #420...this is* in the realm of montmorillonite. 
            plt.plot(np.diff(np.sort(mydcv[272,:])))#INorganicSpike
            plt.plot(np.diff(np.sort(mydcv[367,:])))#D/G
            plt.plot(np.diff(np.sort(mydcv[445,:])))#D/G
            plt.plot(np.diff(np.sort(mydcv[909,:])))#CH
            plt.plot(np.diff(np.sort(mydcv[600,:])))
            plt.plot(np.diff(np.sort(mydcv[700,:])))
            plt.plot(np.diff(np.sort(mydcv[1000,:])))  
            plt.legend(('420','1000','D','G','CH','Test1','Test2','end'),loc=9,ncol=4,prop={'size':12})                   
            plt.ylim([0,0.002])
            plt.subplot(2,1,1)
            plt.plot(np.sort(mydcv[111,:])) #420...this is* in the realm of montmorillonite. 
            plt.plot(np.sort(mydcv[272,:]))#INorganicSpike
            plt.plot(np.sort(mydcv[367,:]))#D/G
            plt.plot(np.sort(mydcv[445,:]))#D/G
            plt.plot(np.sort(mydcv[909,:]))#CH
            plt.plot(np.sort(mydcv[600,:]))
            plt.plot(np.sort(mydcv[700,:]))
            plt.plot(np.sort(mydcv[1000,:]))  
            #plt.legend(('420','1000','D','G','CH','Test1','Test2','end'),loc=9,ncol=2)  
            plt.savefig(savename_fign + '_sorted.png',transparent=True)
            plt.close()     
        
        #Plot raw values (gray)
        for j in range(0,num_images):
            plt.plot(dc00[myn,:,j],color=[0.6,0.6,0.6])

            #Plot 10th percentile data (blue)        
            plt.plot(pctl[myn,:],color='magenta')
            plt.plot(bkr_values[myn,:],color='cyan')
            savename_fign = image_name + '_' + str(j) 
            savedata = savename_fign + 'allcomp.jpg'
            plt.savefig(savedata)  
            plt.close()
        return bkr_values

        
def collection_subtract_bkr(nrtd,bkr_data,bleach_bkr_multiplier=None,rep_bkr_multiplier=None):
    """ Subtracts the background value from all of the replicates in all images 
    of a collection. 
    notes:
        -Does NOT consider bleach! If you want to include the 'bleach' in this analysis, 
            you should run the 'use_bleach' code or write your own
    inputs:
        nrtd: this is your collection class
        bkr_data: this is an nxm array containing the background data. 
        bleach_bkr_multiplier: this is an old input that is no longer used. I keep it to avoid causing problems with the data
        rep_bkr_multiplier: this is an old input that is no longer used. I keep it to avoid causing problems with the data        
    outputs:
        nrtd: background data 
    """
    bad_file = False
    
    for i,Image in enumerate(nrtd.Image):
        #THIS IS WRONG!!!! BUT THEN WHAT HAPPENS TO THE LAST REPLICATE
        num_rows,num_wns,num_replicates = Image.Replicate_Data.shape
        
        for k in range(0,num_replicates):
            try:
                nrtd.Image[i].Replicate_Data[:,:,k] = nrtd.Image[i].Replicate_Data[:,:,k]-bkr_data                
            except:
                bad_file = True
                print(('Failure on Replicate Subtract',j))
                continue
                
    if bad_file == False:
        nrtd.Background_Subtracted = True             
    return nrtd 
    
                                                               
def collection_subtract_bkr_alt(nrtd,bkr_data,bleach_bkr_multiplier=None,rep_bkr_multiplier=None):
    """ figucts the background value from all of the replicates in all images 
    of a collection. 
    notes:
        -Does NOT consider bleach! If you want to include the 'bleach' in this analysis, 
            you should run the 'use_bleach' code or write your own
    inputs:
        nrtd: this is your collection class
        bkr_data: this is an nxm array containing the background data. 
        bleach_bkr_multiplier: this is an old input that is no longer used. I keep it to avoid causing problems with the data
        rep_bkr_multiplier: this is an old input that is no longer used. I keep it to avoid causing problems with the data        
    outputs:
        nrtd: background data 
    """
    bad_file = False
    
    for i,Image in enumerate(nrtd.Image):
        #THIS IS WRONG!!!! BUT THEN WHAT HAPPENS TO THE LAST REPLICATE
        num_rows,num_wns,num_replicates = Image.Replicate_Data.shape
        
        for k in range(0,num_replicates):
            try:
                nrtd.Image[i].Replicate_Data[:,:,k] = nrtd.Image[i].Replicate_Data[:,:,k]*65565-bkr_data*65565               
            except:
                bad_file = True
                print(('Failure on Replicate Subtract',j))
                continue
                
    if bad_file == False:
        nrtd.Background_Subtracted = True             
    return nrtd                                                                                                                                                                                  

def collection_divide_bkr(nrtd,bkr_data,bleach_bkr_multiplier=None,rep_bkr_multiplier=None,bkg_floor=300):
    """ Divides the background value from all of the replicates in all images 
    of a collection. 
    notes:
        -Does NOT consider bleach! If you want to include the 'bleach' in this analysis, 
            you should run the 'use_bleach' code or write your own
    inputs:
        nrtd: this is your collection class
        bkr_data: this is an nxm array containing the background data. 
        bleach_bkr_multiplier: this is an old input that is no longer used. I keep it to avoid causing problems with the data
        rep_bkr_multiplier: this is an old input that is no longer used. I keep it to avoid causing problems with the data        
    outputs:
        nrtd: background data 
    """
    import numpy as np
    bad_file = False
    
    for i,Image in enumerate(nrtd.Image):
        #THIS IS WRONG!!!! BUT THEN WHAT HAPPENS TO THE LAST REPLICATE
        num_rows,num_wns,num_replicates = Image.Replicate_Data.shape
        
        for k in range(0,num_replicates):
            try:
                upperdiff = nrtd.Image[i].Replicate_Data[:,:,k]*65535 -bkg_floor + 1
                lowerdiff = bkr_data*65535-bkg_floor + 1
                nrtd.Image[i].Replicate_Data[:,:,k] = np.true_divide(upperdiff,lowerdiff)-1               
            except:
                bad_file = True
                print(('Failure on Replicate Subtract',j))
                continue
                
    if bad_file == False:
        nrtd.Background_Subtracted = True             
    return nrtd         

def remove_cosmic(nrtd,plot=False):
    """ Remove cosmic rays from each image by comparing the selected replicate
    with all other replicates. The more replicates you use, the lower your thresholds 
    can be, and the lower the chance of treating good data as spikes, or permitting bad data to pass. 
    notes:
        -Does NOT consider bleach! If you want to include the 'bleach' in this analysis, 
            you should run the 'use_bleach' code or write your own
    inputs:
        nrtd: this is a collection class
        plot: should you make plots of the cosmic ray removal
    outputs: 
        nrtd: this is a collection class, with the cosmic rays removed.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from copy import deepcopy

    
    ncosmic = 0
    
    for i,Image in enumerate(nrtd.Image):
        #print 'i=',i

        nrows,nwns,nreps = Image.Replicate_Data.shape
	
        for j in range(0,nrows): 
            #print 'j=',j  
            
            oldrows = deepcopy(Image.Replicate_Data[j,:,:])
            

            
            for k in range(0,nreps):
                #if (i == 3) and (j == 20):
                    #print k
                    #import pdb
                    #pdb.set_trace()
                #print i,j,k
                myRow = Image.Replicate_Data[j,:,k]
                myComparison = np.vstack((Image.Replicate_Data[j,:,:k].transpose(),Image.Replicate_Data[j,:,k+1:].transpose()))
                #Check if all values are nans....if so, skip and continue
                if np.sum(~np.isnan(myRow)) == 0:
                    print("Could not handle cosmic Rays....")
                    continue

                crl,sdb,sdc = remove_cosmic_core(myRow,myComparison)

                #But I think I did this because sometimes the spike happened in
                #Way that there were issues on either side. Thus you really needed
                #to remove the values on either side of this spike. 
                all_crl = np.hstack((crl,np.add(crl,1),np.add(crl,-1)))

                Image.Replicate_Data[j,all_crl,k] = np.nan
           
                try:
                    nrtd.Image[i].Replicate_Data[j,:,k] = lininterp_nan(Image.Replicate_Data[j,:,k])

                except:
                    print('Error finding cosmic rays')
            
                if len(all_crl) > 0:
                    ncosmic = ncosmic + 1
                    
                if (len(all_crl) > 0) & plot:

                    #print "Plot:"
                    #print all_bad_inds
                    cgy = "#a0a0a0"
                    co = "#f4a442"
                    cgn = "#2f9143"

                    
                    
                    nuRow = oldrows[:,k]
                    nuComparison = np.vstack((oldrows[:,:k].transpose(),oldrows[:,k+1:].transpose()))
                    
                    myshape = nuComparison.shape

                    if len(myshape) == 1:
                        spec_nice = nuComparison[None,:]
                    else:
                        spec_nice = nuComparison
                    nuComp = np.median(spec_nice,axis=0)   
                    
                    x = np.arange(1024)
                    f, ax = plt.subplots(3, sharex=True)   
                       
                    print(x[crl].shape,nrtd.Image[i].Replicate_Data[j,crl,k].shape)
                    #Plot the basic spectra
                    ax[0].plot(x,oldrows[:,k],color=co)
                    ax[0].plot(x,nrtd.Image[i].Replicate_Data[j,:,k],color=cgy)
                    ax[0].plot(x[crl],oldrows[crl,k],'o', markerfacecolor="None",markeredgecolor='k')
                    ax[0].set_ylim([0,np.nanmax(nrtd.Image[i].Replicate_Data[j,100::,k])+0.01])

                    #remove_cosmic_core calcs
                    ax[1].plot(x,sdb,color=cgn)
                    ax[1].plot(x,sdc,color=cgy)
                    ax[1].set_ylim([-0.002,0.002])
                    
                    dr = sdb-sdc
                    z_score = calculate_z_score(dr)               
                    ax[2].plot(x,z_score,color=cgn)
                    ax[2].plot([x[0],x[-1]],[-3.5,-3.5],'k')
                    ax[2].set_ylim([-15,15])


                    img_num = i
                    row_num = j
                    rep_num = k
                    
                    ax[0].set_title(['Spike Detected: Image:' + str(img_num) + 'Row:' + str(row_num)])
                    coll_main = nrtd.Collector.Data_File.split(' ')[2]
    
                    savestr = 'spk' + coll_main + '_img' + str(img_num) + '_row' + str(row_num) + '_rep' + str(rep_num) +  '.png'
                    plt.savefig(savestr,dpi=100)  
                    plt.close('all')
                
    nrtd.Summary.Cosmic_Removed = True
    nrtd.Summary.nCosmic = ncosmic
        
    return(nrtd)   
    
def add_binwn(nrtd,x_value):
    nrtd.Summary.binwn = x_value
    return(nrtd)
 



def detect_charring(nrtd,limits=[-9999,-9999],limits_as_wavenumber = True, bin_wavenumber = None, make_plot=True,file_save='',vmax=0.1,vmin=-0.1,threshold_l=0,count_burning=True):
    '''
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    import pdb
    import os
    from copy import deepcopy
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
            limit_upper = find_nearest(limits[1],bin_wavenumber)
            limit_lower = find_nearest(limits[0],bin_wavenumber)
            #print(limit_lower,limit_upper)
        except:
            print('Error (Extract_Banddiff): could not find bin index')
            print('You likely need to supply wavenumber values')
            return None                    
    else:
        limit_lower = limits[0]
        limit_upper = limits[1] + 1 #note we add one
    
    #############################
    #
    #   Determine Band Image Plot
    #                                                                                                                                  
    num_rows,num_wns,num_replicates = nrtd.Image[0].Replicate_Data.shape
    num_images = len(nrtd.Image)
    
    percdiffcube = np.zeros((num_rows,num_images))
    meancube = np.zeros((num_rows,num_images))
    
    for i,Image in enumerate(nrtd.Image):
        
        meanvs = np.nanmean(Image.Replicate_Data[:,limit_lower:limit_upper,:],axis=1)
        rep1vs = meanvs[:,0]#This is the first replicate value
        repmxv = np.nanmax(meanvs[:,1::],axis=1) #This is the maximum value in any subsequent replicate
        allmean = np.nanmean(meanvs,axis=1)
        meancube[:,i] = allmean
        percdiffcube[:,i] = np.true_divide(repmxv-rep1vs,rep1vs)
        
    threshold_l = 0.8
    mean_l = 0.0005
    threshold_m = 0.4
    mean_m = 0.002
    threshold_h = 0.2
    mean_h = 0.005     
    burninglocs_h = np.where((((percdiffcube > threshold_h) & (meancube>mean_h)))) 
    burninglocs_m = np.where((((percdiffcube > threshold_m) & (meancube > mean_m) & (meancube <= mean_h))))
    burninglocs_l = np.where((((percdiffcube > threshold_l) & (meancube > mean_l) & (meancube <= mean_m))))
    
    burninglocs = np.where((((percdiffcube > threshold_h) & (meancube>mean_h))| ( (percdiffcube > threshold_m) & (meancube > mean_m) & (meancube <= mean_h)  ) | ( (percdiffcube > threshold_l) & (meancube > mean_l) & (meancube <= mean_m)  ) )) 
    nburning = len(burninglocs[0])
    output_band_image = deepcopy(percdiffcube)
    output_band_image[meancube<mean_l] = 0 
     
    
    #Set the Quality Stat to 2 for 'burning'
    nrtd.Summary.Spectra_Quality[burninglocs] = 2

    if make_plot:          
    
        fig = plt.figure()   
        ax = fig.add_subplot(111)   
    
        x,y = np.meshgrid(list(range(num_images)),list(range(num_rows)))

        if vmax<0:
            vmax=np.nanmax(output_band_image)
            vmin=-1*vmax
            
        xlocs = x[burninglocs] + 0.5
        ylocs = y[burninglocs] + 0.5
            
        imgplot= ax.pcolormesh(x,y,output_band_image,vmin=vmin,linewidth=0,cmap='bwr',vmax=vmax) 
        cbh = plt.colorbar(imgplot,orientation='vertical') 
        ax.scatter(xlocs,ylocs,facecolors='none',edgecolors='k')
        ax.set_title(nrtd.Summary.Save_Name + ' Rep2-Bleach' )                                
        ax.set_xlabel('Spectrum Number')
        ax.set_ylabel('Vertical Distance (micron)')
        ax.set_xlim([0,np.nanmax(x)])
        ax.set_ylim([0,np.nanmax(y)])
        cbh.set_label('Mean Raman Intensity Change 1300-1650 cm-1 (a.u.)')
        savename= file_save + '_dimg.png'
        plt.savefig(savename,transparent=True)
        
        xbv = x[burninglocs]
        ybv = y[burninglocs]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        
        for i in range(nburning):
            xloc =  xbv[i]
            yloc = ybv[i]
            #print(i)

            ax.plot(bin_wavenumber,nrtd.Image[xloc].Replicate_Data[yloc,:,:])
            ax.legend(['Replicate 1','Replicate 2','Replicate 3'])
            ax.set_xlabel('Raman Shift (cm-1)')
            ax.set_ylabel('Raman Intensity (A.U.)')
            ax.set_xlim([300,3200])
            #pdb.set_trace()
            mymean =     meancube[burninglocs[0][i],burninglocs[1][i]] 
            myperc = percdiffcube[burninglocs[0][i],burninglocs[1][i]] 
            ax.set_title("Mean:" + "{:.5f}".format(mymean) + " |Perc:" + "{:.5f}".format(myperc))
            savename = file_save + "_" + str(xloc) + "_" + str(yloc) + '.png'
            plt.savefig(savename,transparent=True)
            #savename = "fig_burn/burn_" + "{:.4f}".format(mymean) + "_" + "{:.4f}".format(myperc) + "_" + os.path.basename(file_save) + ".png"
            #plt.savefig(savename,transparent=True)
            plt.cla()
            

        #if count_burning=True:
        #    nrows,ncols = diffcube.shape
        #    for row in nrows:
        #        for col in ncols:
        #            if diffcube[i,j] > thresholddiff
                    
    
    nrtd.Summary.nBurning = nburning
    return nrtd,output_band_image      

def extract_banddiff(nrtd,limits=[-9999,-9999],limits_as_wavenumber = True, bin_wavenumber = None, make_plot=True,file_save='',vmax=0.5,vmin=-0.5,threshold_l=0,count_burning=True):
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
            limit_upper = find_nearest(limits[1],bin_wavenumber)
            limit_lower = find_nearest(limits[0],bin_wavenumber)
            #print(limit_lower,limit_upper)
        except:
            print('Error (Extract_Banddiff): could not find bin index')
            print('You likely need to supply wavenumber values')
            return None                    
    else:
        limit_lower = limits[0]
        limit_upper = limits[1] + 1 #note we add one
    
    #############################
    #
    #   Determine Band Image Plot
    #                                                                                                                                  
    num_rows,num_wns,num_replicates = nrtd.Image[0].Replicate_Data.shape
    num_images = len(nrtd.Image)
    
    diffcube = np.zeros((num_rows,num_images))
    meancube = np.zeros((num_rows,num_images))
    
    for i,Image in enumerate(nrtd.Image):
        
        meanvs = np.nanmean(Image.Replicate_Data[:,limit_lower:limit_upper,:],axis=1)
        allmean = np.nanmean(meanvs,axis=1)
        
        meancube[:,i] = allmean
        diffcube[:,i] = np.subtract(meanvs[:,-1],meanvs[:,0])
        
        #This checks to see if Raman values are above a minimum threshold. 
        #diffcube[allmean<=threshold,i] = np.NaN
        
     
    #This is the first method: just check what the magnitude of the difference is
    #output_band_image = diffcube
    #thresholddiff = 0.001
    #nburning = sum(diffcube[~np.isnan(diffcube)]>thresholddiff)
    #burninglocs = np.where(diffcube > thresholddiff)
    
    thresholddiff = 0.1
    percdiffcube = np.divide(diffcube,meancube)
    nburning = sum(percdiffcube[~np.isnan(percdiffcube)]>thresholddiff)
    burninglocs = np.where(percdiffcube > thresholddiff)
    output_band_image = percdiffcube
    
    #Set the Quality Stat to 2 for 'burning'
    nrtd.Summary.Spectra_Quality[burninglocs] = 2

    if make_plot:          
    
        fig = plt.figure()   
        ax = fig.add_subplot(111)   
    
        x,y = np.meshgrid(list(range(num_images)),list(range(num_rows)))

        if vmax<0:
            vmax=np.nanmax(output_band_image)
            vmin=-1*vmax
            
        xlocs = x[burninglocs] + 0.5
        ylocs = y[burninglocs] + 0.5
            
        imgplot= ax.pcolormesh(x,y,output_band_image,vmin=vmin,linewidth=0,cmap='bwr',vmax=vmax) 
        cbh = plt.colorbar(imgplot,orientation='vertical') 
        ax.scatter(xlocs,ylocs,facecolors='none',edgecolors='k')
        ax.set_title(nrtd.Summary.Save_Name + ' Rep2-Bleach' )                                
        ax.set_xlabel('Spectrum Number')
        ax.set_ylabel('Vertical Distance (micron)')
        ax.set_xlim([0,np.nanmax(x)])
        ax.set_ylim([0,np.nanmax(y)])
        cbh.set_label('Mean Raman Intensity Change 1300-1650 cm-1 (a.u.)')
        savename= file_save + '_dimg.png'
        plt.savefig(savename,transparent=True)
        
        xbv = x[burninglocs]
        ybv = y[burninglocs]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        
        for i in range(nburning):
            xloc =  xbv[i]
            yloc = ybv[i]
            #print(i)

            ax.plot(bin_wavenumber,nrtd.Image[xloc].Replicate_Data[yloc,:,:])
            ax.legend(['Replicate 1','Replicate 2','Replicate 3'])
            ax.set_xlabel('Raman Shift (cm-1)')
            ax.set_ylabel('Raman Intensity (A.U.)')
            ax.set_xlim([300,3200])
            savename = file_save + "_" + str(xloc) + "_" + str(yloc) + '.png'
            plt.savefig(savename,transparent=True)
            plt.cla()
            

        #if count_burning=True:
        #    nrows,ncols = diffcube.shape
        #    for row in nrows:
        #        for col in ncols:
        #            if diffcube[i,j] > thresholddiff
                    
    
    nrtd.Summary.nBurning = nburning
    return nrtd,output_band_image      
                            
def remove_fluorescence(nrtd,p=0.01,lmb=1e6,clip_ind=100):
    """ Removes fluorescence using asymmetric least squares fit (Eilers et al 2002)
    Uses remove_cosmic_core to do the calculation. NOw works with infinite replicates.
    Notes:
        -Does NOT consider bleach! If you want to include the 'bleach' in this analysis, 
            you should run the 'use_bleach' code or write your own
    inputs:
        nrtd: this is a Collection class that contains the data
        p: controls the asymmetry (how much do you weight the downward excursions vs the upward peaks)
        lmb: controls the smoothness (how quickly should you go 'up' into peaks)
            for lambda, a value of 1e6 is decent for Raman spectra and should not
            remove too much of the D/G peaks. A value of around 10 will remove all the 
            d/g peaks and only leave very sharp peaks. 
    outputs:
        nrtd: this is a collection class that contains the fluorescence removed data
    """
    
    import numpy as np
    
    for i,Image in enumerate(nrtd.Image):  
        #Image.Replicate_Data.shape
        nrows,nwns,nreps = Image.Replicate_Data.shape
        nrtd.Image[i].Clip_Ind = clip_ind
        nrtd.Image[i].Fluorescence_Data = np.zeros((nrows,nwns,nreps))
        for j in range(0,nrows):   
            for k in range(0,nreps):
                fluo_rep = background_als_core_nu(nrtd.Image[i].Replicate_Data[j,clip_ind::,k],handle_end=True,p=p,lmb=lmb)
                nrtd.Image[i].Replicate_Data[j,clip_ind::,k]= nrtd.Image[i].Replicate_Data[j,clip_ind::,k]-fluo_rep
                nrtd.Image[i].Fluorescence_Data[j,clip_ind::,k] = fluo_rep
                            
    nrtd.Summary.Fluorescence_Removed = True
                
    return(nrtd)

def qc_spectrum(nrtd,ph1=0,ph2 = 0):
    """ Goes through all spectra - if they fail a QC check - then they get set to zero
        Note that fluorescence data are not set to zero - so be careful when you run it
    Notes:
        -Does NOT consider bleach! If you want to include the 'bleach' in this analysis, 
            you should run the 'use_bleach' code or write your own
    inputs:
        nrtd: this is a Collection class that contains the data
        ph1 : placeholder
        ph2 : placeholder

    outputs:
        nrtd: this is a collection class that contains the fluorescence removed data
    """
    
    for i,Image in enumerate(nrtd.Image):  
        #Image.Replicate_Data.shape
        nrows,nwns,nreps = Image.Replicate_Data.shape
        for j in range(0,nrows):
            if nrtd.Summary.Spectra_Quality[j,i] != 0 :  
                nrtd.Image[i].Replicate_Data[j,:,:]= 0

                            
    nrtd.QC_applied = True
                
    return(nrtd)

def qc_spectrum_na(nrtd,ph1=0,ph2 = 0):
    """ Goes through all spectra - if they fail a QC check - then they get set to NaN
        Note that fluorescence data are not set to zero - so be careful when you run it
    Notes:
        -Does NOT consider bleach! If you want to include the 'bleach' in this analysis, 
            you should run the 'use_bleach' code or write your own
    inputs:
        nrtd: this is a Collection class that contains the data
        ph1 : placeholder
        ph2 : placeholder

    outputs:
        nrtd: this is a collection class that contains the fluorescence removed data
    """
    
    for i,Image in enumerate(nrtd.Image):  
        #Image.Replicate_Data.shape
        nrows,nwns,nreps = Image.Replicate_Data.shape
        for j in range(0,nrows):
            if nrtd.Summary.Spectra_Quality[j,i] != 0 :  
                nrtd.Image[i].Replicate_Data[j,:,:]= np.NaN

                            
    nrtd.QC_applied = True
                
    return(nrtd)

def integrate_spectra(nrtd,wmin=None,wmax=None):
    """ Integrates Raman Spectra
    Notes:
        -Does NOT consider bleach! If you want to include the 'bleach' in this analysis, 
            you should run the 'use_bleach' code or write your own
    inputs:
        nrtd: this is a Collection class that contains the data 
        wmax: this is the wavenumber to end the integration. Must be in reciprocal centimeters
                in order to use this function, you must also have set nrtd.Summary.binwn
        wmin: this is the wavenumber to begin the integration. Must be one value in reciprocal centimeters
                In order to use this feature, you must also have set nrtd.Summary.binwn
    outputs:
        nrtd: this is a collection class that contains the data, with Fluorescence_Sum, Raman_Sum, and Raman_Max all included. 
    dependencies:
            find_nearest
            numpy
    test notes: 
        Spot tested on 2/21/2017. differentintegrationtesting.py
    """

    #Determines the limits for the Summation    
    myinds = nrtd.Summary.binwn

    if wmax:#If the upper limit is specified
        indmax = find_nearest(myinds,wmax)+1
    else:
        indmax = len(myinds)
           
    if wmin:#If the lower limit is specified
        indmin = find_nearest(myinds,wmin)
    else:
        indmin =  nrtd.Image[0].Clip_Ind
              
    import numpy as np
    
    for i,Image in enumerate(nrtd.Image):  
        nrows,nwns,nreps = Image.Replicate_Data.shape
        nrtd.Image[i].Fluorescence_Sum = np.zeros((nrows,nreps))
        nrtd.Image[i].Raman_Sum = np.zeros((nrows,nreps))
        nrtd.Image[i].Raman_Max = np.zeros((nrows,nreps))
        nrtd.Image[i].Fluorescence_Max = np.zeros((nrows,nreps))
        for j in range(0,nrows):   
            for k in range(0,nreps):
                nrtd.Image[i].Fluorescence_Sum[j,k] = np.nansum(nrtd.Image[i].Fluorescence_Data[j,indmin:indmax,k])
                nrtd.Image[i].Fluorescence_Max[j,k] = np.nanmax(nrtd.Image[i].Fluorescence_Data[j,indmin:indmax,k])
                nrtd.Image[i].Raman_Sum[j,k] = np.nansum(nrtd.Image[i].Replicate_Data[j,indmin:indmax,k])
                nrtd.Image[i].Raman_Max[j,k] = np.nanmax(nrtd.Image[i].Replicate_Data[j,indmin:indmax,k])
                
    return(nrtd)
        
	
def collection_process(nrtd,method='avg',proc_fluo=False,qc=True):
    """ makes a 'dircube, which is a 3-d dataset corresponding to
    (laser_dimension, wavenumber dimension, imaging_x_position dimension)
        laser_dimension: this is the number of usable pixels in the laser line.
            each corresponds roughly to 1 micron
        wavenumber_dimension: this corresponds to wavenumber. 1024 pixels
        imaging x position: this is the x location of the image. Corresponds to physical space. 
            Note that an x position may have multiple z locations If there are vertical scans
            I have opted to keep the dircube 3d for ease of use and so that the code 
            functions with previously written methods. 
    Notes:
        -This code analyzes the bleached data!
        -This code works for n replicates
    Inputs: 
        nrtd: this is a collection class containing the data from the collection
        method: this is the method to use. Options
            'sum': take the sum of all of the points. Legacy method.
            'avg': take the average of all the points
        proc_fluo: Processes the fluorescence. Don't need to do for determination of the BKR
    """
    import numpy as np

    image_time = nrtd.Summary.Imaging_Start
                                                                
    num_rows,num_wns,num_replicates = nrtd.Image[0].Replicate_Data.shape    
    num_images = len(nrtd.Image)
    
    final_dircube = np.zeros((num_rows,num_wns,num_images))
    final_fmax = np.zeros((num_rows,num_images))
    final_fsum = np.zeros((num_rows,num_images)) 
    final_rmax = np.zeros((num_rows,num_images))     
    output_x = np.zeros(num_images)
    output_z = np.zeros(num_images)
    
    for i,Image in enumerate(nrtd.Image):

        output_x[i] = Image.x_location
        output_z[i] = Image.z_location

        output_mat = Image.Replicate_Data  
        if method == 'avg'  :      
            final_dircube[:,:,i] = np.nanmean(output_mat,axis=2)
        else:
            final_dircube[:,:,i] = np.nansum(output_mat,axis=2)

        if proc_fluo == True:
            try:
                final_fsum[:,i] = np.nanmean(Image.Fluorescence_Sum,axis=1)  
                final_fmax[:,i] = np.nanmean(Image.Fluorescence_Max,axis=1)     
                final_rmax[:,i] = np.nanmean(Image.Raman_Max,axis=1)  
            except Exception as e:
                imgerrtxt = 'Error on image ' + str(i)
                print(imgerrtxt)
                print((str(e)))
                print('Problem averaging SumFluo, SumR, or Rmax')
                print('Possibly you have not calculated these values')
                print('I am going to continue....and set SumFluo,Sumr, and Rmax to zero')
                final_fsum[:,i] = 0
                final_fmax[:,i] = 0 
                final_rmax[:,i] = 0 
        else:
            final_fsum[:,i] = 0
            final_fmax[:,i] = 0 
            final_rmax[:,i] = 0 
            
    return (final_dircube,output_x,output_z,image_time,final_fsum,final_fmax,final_rmax)
	
def collection_process_fl(nrtd,method='avg'):
    """ makes a 'dircube, which is a 3-d dataset corresponding to
    (laser_dimension, wavenumber dimeion, imaging_x_position dimension)
        laser_dimension: this is the number of usable pixels in the laser line.
            each corresponds roughly to 1 micron
        wavenumber_dimension: this corresponds to wavenumber. 1024 pixels
        imaging x position: this is the x location of the image. Corresponds to physical space. 
            Note that an x position may have multiple z locations IF there are vertical scans
            I have opted to keep the dircube 3d for ease of use and so that the code 
            functions with previously written methods. 
    Notes:
        -This code analyzes the bleached data!
        -This :q
code works for n replicates
    Inputs: 
        nrtd: this is a collection class containing the data from the collection
        method: this is the method to use. Options
            'sum': take the sum of all of the points. Legacy method.
            'avg': take the average of all the points
        proc_fluo: Processes the fluorescence. Don't need to do for determination of the BKR
    """
    import numpy as np

    image_time = nrtd.Summary.Imaging_Start
                                                                
    num_rows,num_wns,num_replicates = nrtd.Image[0].Fluorescence_Data.shape    
    num_images = len(nrtd.Image)
    
    final_dircube = np.zeros((num_rows,num_wns,num_images))
    final_fluo = np.zeros((num_rows,num_images))
    final_avg = np.zeros((num_rows,num_images)) 
    final_max = np.zeros((num_rows,num_images))     
    output_x = np.zeros(num_images)
    output_z = np.zeros(num_images)
    
    for i,Image in enumerate(nrtd.Image):

        output_x[i] = Image.x_location
        output_z[i] = Image.z_location

        output_mat = Image.Fluorescence_Data
        if method == 'avg'  :      
            final_dircube[:,:,i] = np.nanmean(output_mat,axis=2)
        else:
            final_dircube[:,:,i] = np.nansum(output_mat,axis=2)

            
    return (final_dircube,output_x,output_z,image_time,final_fluo,final_avg,final_max)
#these are two 'core' codes that are called by other programss             

def remove_cosmic_core(*args,**kwargs):
    """ removes cosmic rays. The first argument is the line you want to check for cosmic rays
    The second argument is any other lines you need to use to compare (you need at least one)
    inputs:
        first argument: this should be a 1d array of the spectrum you are interested in
            all other argumnents: should be 1d arrays of comparison spectra (OR) 2-d arrays
            of spectra stacked vertically. 
    outputs:
        bad_locations: this gives you the locations of the cosmic rays. Note that it also removes two points on either side 
            of the bad locations
        secdrv_base: this is the second derivatie that is compared
        secdrv_comp: this is the second derivate that is compared to
    notes:
        this is optimized for the REBS. If you want to use it for another purpose, look carefully
        at the last two lines in this program. 
    useage:
        >>> a = np.random.rand(300)/200.0
        >>> b = np.random.rand(300)/200.0
        >>> b[100] = 100
        >>> b[200] = 500
        >>> bad = rt.remove_cosmic_core(b,a)
        >>> plt.plot(a)
        >>> plt.plot(b)
        >>> plt.plot(bad,b[bad],'o')
        >>> plt.figure()
        >>> plt.plot(rt.better_deriv(xv,a,n=2))
        >>> plt.plot(rt.better_deriv(xv,b,n=2))
        > [6,7,8,9,10]
    """
    import numpy as np
    from scipy.signal import savgol_filter
    
    
    ####
    #
    #   Parameters used to test

    #This is the critical ratio of second derivatives
    #I am pretty sure we could remove this
    negthrsh_rat =  3 
    #Tis is the Z score of the difference of second derivative values.
    #This is the key one. Reccomend -4 to -6
    #Essentially, this is the standard deviation value, shifted down (or up) by the mean value
    #So basically, we compare the measured second derivative in the analyzed column with the 
    #Second derivative in the 'comparison' column and calculate a standard deviation of those values. 
    #This would be the fifth standard deviation down. 
    #An alternate method would be to look at the interquartile range and do q1-3*iqr. 
    
    z_thresh = -5 
                    
    
    savgol_N = 5 #Number of points for savitzky-golay
    savgol_O = 2 #Order of savitzky-golay filter
    
    spec_base = args[0]
    #print(len(args))
    if len(args) == 2:
        allcompspec = args[1]
        myshape = allcompspec.shape
        #print "length = 2: computing spec_comp"
        #print myshape
        if len(myshape) == 1:
            spec_nice = allcompspec[None,:]
        else:
            spec_nice = allcompspec
        spec_comp = np.median(spec_nice,axis=0)
    else: 
        spec_all= args[1]
        for i in range(2,len(args)):
            spec_all = np.vstack((spec_all,args[i]))
        spec_comp = np.median(spec_all,axis=0)
		#print spec_all
		
		
    #print spec_base.shape
    #print('Comparison spectra')
    #print spec_comp.shape
    xv = np.arange(len(spec_base))
    secdrv_base = savgol_filter(spec_base,savgol_N ,savgol_O ,deriv=2)
    secdrv_comp = savgol_filter(spec_comp,savgol_N ,savgol_O ,deriv=2) 

    deriv_diff = secdrv_base-secdrv_comp
    z_diff = calculate_z_score(deriv_diff)
    
    #This is the alternate method that we could use. 
    #p25,p75 = np.percentile(dr,[25,75])
    #iqrdiff = p25 - 1.5*(p75-p25)
    #bigiqrdiff = p25 - 3*(p75-p25)
    #ax[3].plot(x,dr,color=cgn)
    #ax[3].plot([x[0],x[-1]],[iqrdiff,iqrdiff],'k')
    #ax[3].plot([x[0],x[-1]],[bigiqrdiff,bigiqrdiff],co)
    
    deriv_rat = np.true_divide(np.abs(secdrv_base),np.abs(secdrv_comp))

    locs = np.flatnonzero((z_diff<z_thresh)  & (deriv_rat > negthrsh_rat) & (xv > 80) & (xv<1020))
    #locs = np.flatnonzero((z_diff<z_thresh) & (deriv_diff<negthrsh) & (deriv_rat > negthrsh_rat) & (xv > 80) & (xv<1020))
    return locs,secdrv_base,secdrv_comp
                 
                                                   
def background_als_core(y,p=0.01,lmb=1e6,max_iter=10,handle_end=False):
    """This implements the asymmetric least squares smoothing algorithm from 
    Eilers 2002. Eilers' code used Cholesky decomposition to solve the 
    system of equations. We here use the spsolve to solve the system of equations.
    I have tried to make it as fast as I can without resorting to undue machinations.
    Time/Speed requirements: For a 3000 point matrix, it's around 40 ms for 10 iterations
    IE, ~ 4ms per iteration.
    Notes:
        I have set use_umfpack=True in the spsolve call. I do not use it on 
        my system but you may use it and speed things up. 
    Inputs:
        y: 1d numpy array that will have the background removed
        p: controls the asymmetry (how much do you weight the downward excursions vs the upward peaks)
        lmb: controls the smoothness (how quickly should you go 'up' into peaks)
        max_iter: how many iterations to use
        handle_end: a semi-kludgy way to handle the end points....just 
            copy the last point x100. Set to true for this.
    Outputs:
        z: this is your background. It has the same dimensions as y
    Usage:
        >>> x = np.arange(0,1000)
        >>> baseline = np.sin(x*pi/1000)
	"""
    import numpy as np
    from scipy import sparse 
    from scipy.sparse import linalg as spl
    
    if handle_end:
        nuy = np.zeros((1,100))
        nuy[:] = y[-1]
        y = np.append(y,nuy)
    
    m = len(y) #get the length of the array
    w=np.ones(m) #make the initial weight array
    D = sparse.dia_matrix(([np.ones(m),-2*np.ones(m),np.ones(m)],[0,-1,-2]),shape=[m,m-2]) #this is used for the smoother
    DD = lmb*D*D.H
    
    for i in range(max_iter):
        W = sparse.spdiags(w,0,m,m) #Sparse matrix of weights
        z = spl.spsolve(W + DD,np.multiply(w,y),use_umfpack=True) #Solve Ax = b
        w0 = w#get initial W value
        w = p*(y>z) + (1-p)*(y < z) #new weight
        if np.nansum(abs(w-w0)) <= 0: #if no change from last iteration, break
            break
    if handle_end:
        z = z[0:-100]
    
    return z 
    
def background_als_core_nu(y,p=0.01,lmb=1e6,max_iter=10,handle_end=False):
    """This implements the asymmetric least squares smoothing algorithm from 
    Eilers 2002. Eilers' code used Cholesky decomposition to solve the 
    system of equations. We here use the spsolve to solve the system of equations.
    I have tried to make it as fast as I can without resorting to undue machinations.
    Time/Speed requirements: For a 3000 point matrix, it's around 40 ms for 10 iterations
    IE, ~ 4ms per iteration.
    Notes:
        I have set use_umfpack=True in the spsolve call. I do not use it on 
        my system but you may use it and speed things up. 
    Inputs:
        y: 1d numpy array that will have the background removed
        p: controls the asymmetry (how much do you weight the downward excursions vs the upward peaks)
        lmb: controls the smoothness (how quickly should you go 'up' into peaks)
        max_iter: how many iterations to use
        handle_end: a semi-kludgy way to handle the end points....just 
            copy the last point x100. Set to true for this.
    Outputs:
        z: this is your background. It has the same dimensions as y
    Usage:
        >>> x = np.arange(0,1000)
        >>> baseline = np.sin(x*pi/1000)
	"""
    import numpy as np
    from scipy import sparse 
    from scipy.sparse import linalg as spl
    
    hel = 100
    if handle_end:
        nuy = np.zeros((1,hel))
        nuy[:] = y[-1]
        y = np.append(y,nuy)
        nuy = np.zeros((1,hel))
        nuy[:] = y[0]
        y = np.append(nuy,y)
        
        
    
    m = len(y) #get the length of the array
    w=np.ones(m) #make the initial weight array
    D = sparse.dia_matrix(([np.ones(m),-2*np.ones(m),np.ones(m)],[0,-1,-2]),shape=[m,m-2]) #this is used for the smoother
    DD = lmb*D*D.H
    
    for i in range(max_iter):
        W = sparse.spdiags(w,0,m,m) #Sparse matrix of weights
        z = spl.spsolve(W + DD,np.multiply(w,y),use_umfpack=True) #Solve Ax = b
        w0 = w#get initial W value
        w = p*(y>z) + (1-p)*(y < z) #new weight
        if np.nansum(abs(w-w0)) <= 0: #if no change from last iteration, break
            break
    if handle_end:
        z = z[hel:-hel]
    
    return z 

################################################################################
################################################################################
################################################################################
#
#
#   Section 4: Plotting Data
#
#
#
#
#


#####
#
#   plot_allspectra is code to generate a stack of Raman spectra, roughly equivalent
#       to one image from the REBS. It is a quick way to look at the data in a methoed that allows
#       you to approximate the locations of individual peaks
#      

def plot_format(fs=16):
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
    
    
    plt.rc('axes',linewidth=defLW)
    plt.rc('axes',labelsize=fSize)
    plt.rc('axes',labelweight=fWght)
    
    #plt.rc('axes',edgecolor=[0.1,0.1,0.1])#,color='black')
    plt.rc('lines',linewidth = defLW)
    
def plot_allspectra(myimg,file_save='',x_values =[0],title='',rn=None,close=False,min_ind=0):
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    values = list(range(myimg.shape[0]))
    ax = plt.axes()
    ax.set_prop_cycle('color',[plt.cm.brg(i) for i in np.linspace(0, 1, 12)])

    if len(x_values)>1:
        x_value = x_values
    else:
        #DEfault: Use RN13 calibration
        print("No Raman Shift Values provided....using default")
        x_value = get_rebs_calibration(rn=13)
    maxl = 0.4
    sdv = 0.01
    for i in values:

        specdiff = np.multiply(sdv,i)
        myspec = myimg[i,min_ind::]
        plt.plot(x_value[min_ind::],myspec+specdiff) 
        if np.nanmax(myspec+specdiff) > maxl:
            maxl = np.nanmax(myspec+specdiff)
      
    if title:
        plt.title(title )
    plt.xlabel('Raman Shift ($cm^{-1}$)')
    plt.ylabel('Intensity(arb. units)')
    
    plt.axis([200,3200,0,maxl])
    if file_save:
        plt.savefig(file_save)
    else:
        plt.show()
        
    if close:
        plt.close()


def output_rebs_images(myCollection,output_directory,bin_wavenumber=[],note=''):
    import os
    import matplotlib.pyplot as plt
    #Short program to dump figures from a collection 
    basename_image = myCollection.Summary.Save_Name
    if len(bin_wavenumber) ==0 :

        #DEfault: Use RN13 calibration
        bin_wavenumber = get_rebs_calibration(rn=13)
                                 
    for i,Image in enumerate(myCollection.Image):
        
        tsi = ''
        if Image.Background == True:
            tsi = tsi + 'B'
        if Image.Cleaned == True:
            tsi = tsi + 'C'
        if Image.Good == True:
            tsi = tsi + 'G'
        
        title_string = str(myCollection.Summary.Imaging_Start) + ' Img: ' + str(i) + ' (' + tsi + ') ' + note
        #Base save name                        
        savename_img =  basename_image + '_' + format(i,'03d') 
        
        num_rows,num_wns,num_replicates = Image.Replicate_Data.shape
        
        for j in range(0,num_replicates):
            replicate_cleaned = Image.Replicate_Data[:,:,j]
            savename_as = os.path.join(output_directory,savename_img + '_as_r' + format(j,'03d')  + '.png')
            #savename_gi = os.path.join(output_directory,savename_img + '_gi_r' + format(j,'03d')  + '.png')            
            plot_allspectra(replicate_cleaned,file_save = savename_as,title=title_string + ' Rep ' + str(j),x_values=bin_wavenumber,min_ind=50) 
            #plot_genimg(replicate_cleaned,file_save=savename_gi)
            plt.cla() 

def output_rebs_images_dc(mdc,output_directory,bin_wavenumber=[],note='',basename_image=''):
    import os
    import matplotlib.pyplot as plt
    
    if len(bin_wavenumber) == 0 :
        #DEfault: Use RN13 calibration
        bin_wavenumber = get_rebs_calibration(rn=13)

    plt.figure(figsize=(8.5,11))
    
    ncols,nwns,nimgs = mdc.shape
        
    for i in range(0,nimgs):
        savename_img =  basename_image + '_' + format(i,'03d') 
        savename_as = os.path.join(output_directory,savename_img + '_as.png') 
        title_string = basename_image + ' ' + str(i) 
        plot_allspectra(mdc[:,:,i],file_save = savename_as,title=title_string,min_ind=50)
        plt.cla()
        
def nicePalette():
    """ Returns a nice palette of colors. 
    outputs:
            nicepalette (list): this is a list of hex colors
    usage:
        myCols = nicepalette()
        plt.plot(x,y,mycols[2])
    """
    nicepalette = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    return nicepalette
################################################################################
################################################################################
################################################################################
#
#
#   Section 5: Tools
#
#

#SEction 1: These work on the default code wth RN 10,11, 13, 14
#The goal of these codes are to convert the paths to verious properties. 
def get_rn(filename):
    """ Gets the rebs serial number from the file path
    NOTE: THIS IS SPECIFIC TO THE DEFAULT CODE THAT COMES WITH RN 10,11,13,14
    inputs:
        -filename(string): this is the name of a REBS filename
    outputs:
        - rn (int): this is the serial number of the rebs.
    """
    import os
    filename = os.path.basename(filename)
    try:
        outnum = filename[8:10]
        outputval = int(outnum)
    except ValueError:
        
        print('Bad REBS Filename' + filename)
        outputval = 0
    return outputval
    
def convert_rebspath_to_computerpath(spot_info_line,computer_directory):
    import ntpath
    import os
    
    line_seperated_dash = spot_info_line.split('-') #SPlit line by dash
    strp = line_seperated_dash[1].rstrip() #Strip newline and whitespace
    bsn = ntpath.basename(strp) #Strip off path (just get file name)
    pthld = os.path.join(computer_directory,bsn) #Join to computer directory
    return pthld
    
def convert_rebspath_to_datetime(rebspath):
    import ntpath
    from pandas import to_datetime
    bsn = ntpath.basename(rebspath)
    file_name_date = bsn.split(' ')[-1].split('_')[1:3]
    dt = to_datetime(file_name_date[0] + ' ' + file_name_date[1])
    return dt
    
#Section 2: Rebs Codes
#
#   General Purpose Rebs codes
#
    
def get_rebs_calibration(cal_file='',cal_type='fit',rn=13,return_wl=False,pixel_count=1024,strsep=','):
    '''
    This loads the laser wavelength and spectrometer wavelength calibration files
    and returns either the bin wavenumber associated with each pixel or the wavelength.  
    
    '''

    import numpy as np
    av = np.arange(0,pixel_count) #This is the bin number of the CCD
    fit_coeffs=[]
    laser_wavelength=[]
    header=[]
    print('a')
    try:
        if cal_type == 'fit':
            #The format of a fit file - this is generated by ddoughty
            #USes "#" at the front for any information
            #Then the first 'data' line is the fits
            #The second 'data' line is the laser wavelength'
            print('b')
            
            fcstr = ''
            wlstr = ''
            with open(cal_file,'r') as f:
                for line in f:
                    #print(line)
                    if line[0]=='#':
                        header.append(line.strip())
                        #print('1','fc',fcstr,'wl',wlstr)
                    elif (fcstr=='') & (wlstr==''):
                        fcstr = line.strip().split(':')[1]
                        #print('2','fc',fcstr,'wl',wlstr)
                    elif (fcstr!='') & (wlstr == ''):
                        wlstr = line.strip().split(':')[1]
                        #print('3','fc',fcstr,'wl',wlstr)
            fit_coeffs = np.fromstring(fcstr,sep=strsep)
            laser_wavelength = float(wlstr)         

    
        elif cal_type == 'values':
            print('c')
            #Format of 'values' file
            #Uses "#" at the begininning for any needed information
            # Then the first 'data' line is the bin wavenumber
            calstr=''
            
            with open(cal_file,'r') as f:
                for line in f:
                    #print(line)
                    if line[0]=='#':
                        header.append(line.strip())
                        #print('1','fc',fcstr,'wl',wlstr)
                    elif calstr=='':
                        calstr = line.strip()
            myCalibration = np.fromstring(calstr,dtype=float,sep=strsep)

            
        else:
            raise NameError("Incorrect calibration type selected")
        
    except:
        print("Exception encountered when loading cal file...instead defaulting to pre-saved numbers")
        cal_type='fit'
    
    #############'
    ##
    ##     Defaults
    ##              
    ##      These are default calibration informatin sets
    ##      note: this will get you roughly within 10-15 cm-1 if you pick any of these
    ##      But calibration is relaly instrument specific - so be careful. 
    ##
    #import pdb
    #pdb.set_trace()
    
    if laser_wavelength !=[]:
        pass
    elif laser_wavelength==[]:
        #20160825        
        print("Caution: I am using the default calibration values in rs_tools")
        print("These could be wrong by over 10cm-1")
        myCalibration= [-30.8,-26.6,-22.5,-18.3,-14.2,-10.0,-5.9,-1.7,2.4,6.5,10.7,14.8,18.9,23.0,27.1,31.3,35.4,39.5,43.6,47.7,51.8,55.9,60.0,64.1,68.2,72.3,76.4,80.5,84.6,88.6,92.7,96.8,100.9,104.9,109.0,113.1,117.1,121.2,125.3,129.3,133.4,137.4,141.5,145.5,149.6,153.6,157.6,161.7,165.7,169.7,173.8,177.8,181.8,185.8,189.9,193.9,197.9,201.9,205.9,209.9,213.9,217.9,221.9,225.9,229.9,233.9,237.9,241.9,245.9,249.9,253.9,257.8,261.8,265.8,269.8,273.7,277.7,281.7,285.6,289.6,293.5,297.5,301.4,305.4,309.3,313.3,317.2,321.2,325.1,329.0,333.0,336.9,340.8,344.8,348.7,352.6,356.5,360.4,364.4,368.3,372.2,376.1,380.0,383.9,387.8,391.7,395.6,399.5,403.4,407.3,411.2,415.0,418.9,422.8,426.7,430.6,434.4,438.3,442.2,446.0,449.9,453.8,457.6,461.5,465.3,469.2,473.0,476.9,480.7,484.6,488.4,492.2,496.1,499.9,503.7,507.6,511.4,515.2,519.0,522.9,526.7,530.5,534.3,538.1,541.9,545.7,549.5,553.3,557.1,560.9,564.7,568.5,572.3,576.1,579.9,583.7,587.5,591.2,595.0,598.8,602.6,606.3,610.1,613.9,617.6,621.4,625.1,628.9,632.7,636.4,640.2,643.9,647.7,651.4,655.1,658.9,662.6,666.3,670.1,673.8,677.5,681.3,685.0,688.7,692.4,696.1,699.9,703.6,707.3,711.0,714.7,718.4,722.1,725.8,729.5,733.2,736.9,740.6,744.3,748.0,751.6,755.3,759.0,762.7,766.4,770.0,773.7,777.4,781.0,784.7,788.4,792.0,795.7,799.3,803.0,806.6,810.3,813.9,817.6,821.2,824.9,828.5,832.1,835.8,839.4,843.0,846.7,850.3,853.9,857.5,861.2,864.8,868.4,872.0,875.6,879.2,882.8,886.4,890.1,893.7,897.3,900.8,904.4,908.0,911.6,915.2,918.8,922.4,926.0,929.5,933.1,936.7,940.3,943.8,947.4,951.0,954.5,958.1,961.7,965.2,968.8,972.3,975.9,979.4,983.0,986.5,990.1,993.6,997.2,1000.7,1004.2,1007.8,1011.3,1014.8,1018.4,1021.9,1025.4,1028.9,1032.5,1036.0,1039.5,1043.0,1046.5,1050.0,1053.5,1057.1,1060.6,1064.1,1067.6,1071.1,1074.6,1078.0,1081.5,1085.0,1088.5,1092.0,1095.5,1099.0,1102.4,1105.9,1109.4,1112.9,1116.3,1119.8,1123.3,1126.7,1130.2,1133.7,1137.1,1140.6,1144.0,1147.5,1151.0,1154.4,1157.8,1161.3,1164.7,1168.2,1171.6,1175.1,1178.5,1181.9,1185.4,1188.8,1192.2,1195.6,1199.1,1202.5,1205.9,1209.3,1212.7,1216.2,1219.6,1223.0,1226.4,1229.8,1233.2,1236.6,1240.0,1243.4,1246.8,1250.2,1253.6,1257.0,1260.4,1263.8,1267.1,1270.5,1273.9,1277.3,1280.7,1284.0,1287.4,1290.8,1294.1,1297.5,1300.9,1304.2,1307.6,1311.0,1314.3,1317.7,1321.0,1324.4,1327.7,1331.1,1334.4,1337.8,1341.1,1344.5,1347.8,1351.1,1354.5,1357.8,1361.1,1364.5,1367.8,1371.1,1374.4,1377.8,1381.1,1384.4,1387.7,1391.0,1394.3,1397.7,1401.0,1404.3,1407.6,1410.9,1414.2,1417.5,1420.8,1424.1,1427.4,1430.7,1434.0,1437.2,1440.5,1443.8,1447.1,1450.4,1453.7,1456.9,1460.2,1463.5,1466.7,1470.0,1473.3,1476.6,1479.8,1483.1,1486.3,1489.6,1492.9,1496.1,1499.4,1502.6,1505.9,1509.1,1512.4,1515.6,1518.8,1522.1,1525.3,1528.6,1531.8,1535.0,1538.3,1541.5,1544.7,1547.9,1551.2,1554.4,1557.6,1560.8,1564.0,1567.2,1570.5,1573.7,1576.9,1580.1,1583.3,1586.5,1589.7,1592.9,1596.1,1599.3,1602.5,1605.7,1608.9,1612.1,1615.2,1618.4,1621.6,1624.8,1628.0,1631.2,1634.3,1637.5,1640.7,1643.9,1647.0,1650.2,1653.4,1656.5,1659.7,1662.8,1666.0,1669.2,1672.3,1675.5,1678.6,1681.8,1684.9,1688.1,1691.2,1694.4,1697.5,1700.6,1703.8,1706.9,1710.1,1713.2,1716.3,1719.4,1722.6,1725.7,1728.8,1731.9,1735.1,1738.2,1741.3,1744.4,1747.5,1750.6,1753.7,1756.9,1760.0,1763.1,1766.2,1769.3,1772.4,1775.5,1778.6,1781.7,1784.8,1787.8,1790.9,1794.0,1797.1,1800.2,1803.3,1806.4,1809.4,1812.5,1815.6,1818.7,1821.7,1824.8,1827.9,1830.9,1834.0,1837.1,1840.1,1843.2,1846.2,1849.3,1852.4,1855.4,1858.5,1861.5,1864.6,1867.6,1870.7,1873.7,1876.7,1879.8,1882.8,1885.9,1888.9,1891.9,1895.0,1898.0,1901.0,1904.0,1907.1,1910.1,1913.1,1916.1,1919.1,1922.2,1925.2,1928.2,1931.2,1934.2,1937.2,1940.2,1943.2,1946.2,1949.2,1952.2,1955.2,1958.2,1961.2,1964.2,1967.2,1970.2,1973.2,1976.2,1979.2,1982.1,1985.1,1988.1,1991.1,1994.1,1997.0,2000.0,2003.0,2006.0,2008.9,2011.9,2014.9,2017.8,2020.8,2023.7,2026.7,2029.7,2032.6,2035.6,2038.5,2041.5,2044.4,2047.4,2050.3,2053.3,2056.2,2059.1,2062.1,2065.0,2068.0,2070.9,2073.8,2076.8,2079.7,2082.6,2085.5,2088.5,2091.4,2094.3,2097.2,2100.1,2103.1,2106.0,2108.9,2111.8,2114.7,2117.6,2120.5,2123.4,2126.3,2129.2,2132.1,2135.0,2137.9,2140.8,2143.7,2146.6,2149.5,2152.4,2155.3,2158.2,2161.1,2164.0,2166.8,2169.7,2172.6,2175.5,2178.3,2181.2,2184.1,2187.0,2189.8,2192.7,2195.6,2198.4,2201.3,2204.2,2207.0,2209.9,2212.7,2215.6,2218.4,2221.3,2224.1,2227.0,2229.8,2232.7,2235.5,2238.4,2241.2,2244.1,2246.9,2249.7,2252.6,2255.4,2258.2,2261.1,2263.9,2266.7,2269.6,2272.4,2275.2,2278.0,2280.8,2283.7,2286.5,2289.3,2292.1,2294.9,2297.7,2300.5,2303.3,2306.2,2309.0,2311.8,2314.6,2317.4,2320.2,2323.0,2325.8,2328.5,2331.3,2334.1,2336.9,2339.7,2342.5,2345.3,2348.1,2350.8,2353.6,2356.4,2359.2,2362.0,2364.7,2367.5,2370.3,2373.1,2375.8,2378.6,2381.4,2384.1,2386.9,2389.6,2392.4,2395.2,2397.9,2400.7,2403.4,2406.2,2408.9,2411.7,2414.4,2417.2,2419.9,2422.7,2425.4,2428.1,2430.9,2433.6,2436.4,2439.1,2441.8,2444.6,2447.3,2450.0,2452.7,2455.5,2458.2,2460.9,2463.6,2466.4,2469.1,2471.8,2474.5,2477.2,2479.9,2482.6,2485.4,2488.1,2490.8,2493.5,2496.2,2498.9,2501.6,2504.3,2507.0,2509.7,2512.4,2515.1,2517.8,2520.4,2523.1,2525.8,2528.5,2531.2,2533.9,2536.6,2539.2,2541.9,2544.6,2547.3,2550.0,2552.6,2555.3,2558.0,2560.6,2563.3,2566.0,2568.6,2571.3,2574.0,2576.6,2579.3,2581.9,2584.6,2587.3,2589.9,2592.6,2595.2,2597.9,2600.5,2603.2,2605.8,2608.4,2611.1,2613.7,2616.4,2619.0,2621.6,2624.3,2626.9,2629.5,2632.2,2634.8,2637.4,2640.1,2642.7,2645.3,2647.9,2650.6,2653.2,2655.8,2658.4,2661.0,2663.6,2666.3,2668.9,2671.5,2674.1,2676.7,2679.3,2681.9,2684.5,2687.1,2689.7,2692.3,2694.9,2697.5,2700.1,2702.7,2705.3,2707.9,2710.5,2713.0,2715.6,2718.2,2720.8,2723.4,2726.0,2728.5,2731.1,2733.7,2736.3,2738.9,2741.4,2744.0,2746.6,2749.1,2751.7,2754.3,2756.8,2759.4,2762.0,2764.5,2767.1,2769.6,2772.2,2774.8,2777.3,2779.9,2782.4,2785.0,2787.5,2790.1,2792.6,2795.2,2797.7,2800.2,2802.8,2805.3,2807.9,2810.4,2812.9,2815.5,2818.0,2820.5,2823.1,2825.6,2828.1,2830.6,2833.2,2835.7,2838.2,2840.7,2843.2,2845.8,2848.3,2850.8,2853.3,2855.8,2858.3,2860.8,2863.4,2865.9,2868.4,2870.9,2873.4,2875.9,2878.4,2880.9,2883.4,2885.9,2888.4,2890.9,2893.4,2895.8,2898.3,2900.8,2903.3,2905.8,2908.3,2910.8,2913.3,2915.7,2918.2,2920.7,2923.2,2925.6,2928.1,2930.6,2933.1,2935.5,2938.0,2940.5,2942.9,2945.4,2947.9,2950.3,2952.8,2955.3,2957.7,2960.2,2962.6,2965.1,2967.5,2970.0,2972.4,2974.9,2977.3,2979.8,2982.2,2984.7,2987.1,2989.6,2992.0,2994.4,2996.9,2999.3,3001.8,3004.2,3006.6,3009.1,3011.5,3013.9,3016.3,3018.8,3021.2,3023.6,3026.0,3028.5,3030.9,3033.3,3035.7,3038.1,3040.6,3043.0,3045.4,3047.8,3050.2,3052.6,3055.0,3057.4,3059.8,3062.2,3064.7,3067.1,3069.5,3071.9,3074.3,3076.6,3079.0,3081.4,3083.8,3086.2,3088.6,3091.0,3093.4,3095.8,3098.2,3100.5,3102.9,3105.3,3107.7,3110.1,3112.5,3114.8,3117.2,3119.6,3122.0,3124.3,3126.7,3129.1,3131.4,3133.8,3136.2,3138.5,3140.9,3143.3,3145.6,3148.0,3150.3,3152.7,3155.1,3157.4,3159.8,3162.1,3164.5,3166.8,3169.2,3171.5,3173.9,3176.2,3178.5]
        cal_type='values'
    else:
        ValueError("Something went wrong with the laser wavelength assignment")
#    else:
#        raise ValueError("Laser wavelength was empty (implying no cal file provided), and no RN (or incorrect RN) was specified for default cal values")
       
            
    if cal_type=='fit':       
        cv_wl = np.polyval(fit_coeffs,av)#07/08 raman shift values (nm)

        bin_wn = convert_to_wn_raman(cv_wl,laser_wavelength)#original raman shift values cm^-1
        
        if return_wl == True:
            return_value = cv_wl
        else:
            return_value = bin_wn    
        
    elif cal_type == 'values':
        return_value = myCalibration
    else:
        NameError("did not properly specify cal_type. Either use 'fit' or 'values'")
        
    return return_value
    

#Section 2: Computational Codes
#These do math. 
    
def nan_helpfcn(myarray):
    """
	Helper function to return the locations of Nan values as a boolean array, plus a function to return the index of the array.  
    Code inspired by: http://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    Input:
        - myarray, 1d numpy array with possible NaNs, e.g. np.array([1,2,NaN,4,NaN,6])
    Output:
        - nans, logical indices of NaNs, e.g. for the example above [False,False,True,False,True,False]
        - indf,  this gives us the indices of the matrix - shifted. this is a lambda function. e.g. indes(nans) where nans was [False,False,True,False,True,False] gives [2,4]
				This is functionally equivalent to np.array(range(len(myarray)))
    Example:
		>>> myarray = np.array=[1,2,np.NaN,4,np.NaN,6])
		>>> nanlocs,indf=nan_helpfcn(myarray)
		>>>nanlocs
			[False,False,True,False,True,False]
		>>> indf[nanlocs]
			[2,4]
		>>> indf[~nanlocs]
			[0,1,3,5]

    """
    import numpy as np
    return np.isnan(myarray), lambda z: z.nonzero()[0]  

def lininterp_nan(myarray):
    """ This takes in a numpy 1-dimensional array, and linearly interpolates
    across any NAN values.
	Code inspired by: http://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    Input:
        - myarray, 1d numpy array with possible NaNs
    Output
        - myarray, 1d numpy array with NANs linearly interpolated
    Example:
        >>> nan_removed_array = lininterp_nan(array_with_nans)
    """
    import numpy as np
    nanlocs,indf =nan_helpfcn(myarray)
    myarray[nanlocs]= np.interp(indf(nanlocs), indf(~nanlocs), myarray[~nanlocs])
    return(myarray)

def imageinterp(X,Y,data,res=0.1):
    """ This program takes in a 2d image in X and Y, and interpolates to give
    you an image at a much finer resolution
    Input:
        - X: x values generated from meshgrid(xrange,yrange)
        - Y: y values generated from meshgrid(xrange,yrange)
        - Data: cube of data values
        - res: what resolution do you want this to be
    Output:
        Xn: new x values 
        Yn: cube of new y values
        data1: cube of new interpolated values
    Example:
        >>> x,y = meshgrid(np.arange(0,10),np.arange(0,10))
        >>> new_x,new_y,new_scalar = imageinterp(x,y,S)
    """
    from scipy.interpolate import interp2d
    import numpy as np
    # scipy interp. cubic
    f = interp2d(X, Y, data, kind='cubic')
    xnew = np.arange(np.nanmin(X),np.nanmax(X)+1, res)
    ynew = np.arange(np.nanmin(Y),np.nanmax(Y)+1, res)
    data1 = f(xnew,ynew)
    Xn, Yn = np.meshgrid(xnew, ynew)
    return Xn,Yn,data1

def better_deriv(x,y,n,flt=False):
    """ This program takes the derivative of order n of 1d array Y having 
    x values (x). It allows you to filter the array before taking the derivative. 
    Input:
        x: X values (1d array)
        y: Y values (1d array)
        n: order of the derivative 
        flt: If true, we will filter before taking the derivative. 
    Output:
        ddy: this is the derivative of Y at points X.
    """
    from scipy import interpolate
    if flt==True:
        from scipy.signal import savgol_filter
        y = savgol_filter(y,7,3) #Use smoothing that we understand (it is also faster)
    spl = interpolate.splrep(x,y,k=3,s=0) # no smoothing, 3rd order spline
    ddy = interpolate.splev(x,spl,der=n) # use those knots to get second derivative 
    return ddy
    
    
def convert_to_wl_raman(raman_shift,laser_wavelength):
    import numpy as np
    #converts raman shift values (in reciprocal centimeters) 
    # to an absolute wavelength
    
     #Convert laser wavelength to recp. cm
    laser_invcm = np.divide(10000000,laser_wavelength)
    return np.divide(10000000,np.subtract(laser_invcm,raman_shift))

    
    
def convert_to_wn_raman(raman_shift,laser_wavelength):
    import numpy as np

    #Converts raman shift values (in wavelength)
    #To inverse centimeters. Note that wavelength is absolute
    #Raman shift is 'relative' magnitude
    wl_invcm = np.divide(10000000,raman_shift) #Convert bin wavelengths to inverse centimeters
    laser_invcm = np.divide(10000000,laser_wavelength) #Convert laser wavength to inverse centimeters
    raman_wn = laser_invcm-wl_invcm #Subtract form laser wavelength    
    return raman_wn
         
def find_nearest(array,value):
    '''This code outputs the index of array closest to 'value'
    From: http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    inputs: 
        array: this is a 1-d np array
        value: this is the value to search for
    outputs: 
        this is the location closest to value. 
    
    example:
    
    a = np.arange(0,100,5)
    loc = find_nearest(a,7)
    print loc
    print a[loc]
    
    returns: 
    >> 1
    >> 5
    '''
    import numpy as np
    idx = (np.abs(array-value)).argmin()
    return idx                

def get_total_intensity(nrtd):

    #Currently only works with 1 replicate/bleach
    #
    #   Changing Threshold
    #
    import numpy as np
    nrows = nrtd.Image[0].Bleach_Data.shape[0]
    
    for i,Image in enumerate(nrtd.Image):
        nrtd.Image[i].Total_Sum = np.zeros(nrows)
        for j in range(0,nrows):              
             inival = 50       
             nrtd.Image[i].Total_Sum[j] = np.nansum(nrtd.Image[i].Bleach_Data[j,inival::]) + np.nansum(nrtd.Image[i].Replicate_Data[j,inival::,0])                                         
    return(nrtd) 
    
def get_bin_edges(bincenters_invcm):
    #GET BIN EDGES (approximate):
    #There is likely a better way to do this
    #Currently a place holder until we get more info.
    import numpy as np
    md = np.divide(np.diff(bincenters_invcm),2)
    be_ctr = bincenters_invcm[0:-1] + md
    be_ctr1 = np.insert(be_ctr,0,bincenters_invcm[0]-md[0])
    binedges_invcm = np.append(be_ctr1,bincenters_invcm[-1] + md[-1])
    return binedges_invcm
    
def calculate_z_score(values):
    ''' A z-score is the number of standard deviations a value is from the mean of all values
    This code takes in an array of values (numpy 1d array) and reports the
    z-score for each value.
    inputs:
        values: this is a numpy 1d array of values
    outputs:
        z-score: this is a numpy 1d array of the z score for each of 'values'
    usage:
        >>>a = np.random.normal(10,5,5) #Pull five values from a normal distribution centered on 10 with a standard deviation of 5
        >>>a #Note that in this case the mean is 11.4 and the std is 5.37
            array([ 16.84785836,  14.57856086,   1.93135413,   9.06061827,  14.51387853]) 
        >>>z = calculate_z_score(a)
            array([ 1.01553907,  0.59356697, -1.75816013, -0.4324853 ,  0.58153939])  
    '''
    import numpy as np
    mean = np.nanmean(values)
    stddr = np.nanstd(values)
    z_score = np.divide(np.subtract(values,mean),stddr)
    return z_score
    
def normalize(values,method='max'):
    import numpy as np
    if method == 'diff':
        maxv = np.nanmax(values)
        minv = np.nanmin(values)
        return np.true_divide(values-minv,maxv-minv)
    else:
        return np.true_divide(values,np.nanmax(values))   
        
################################################################################
################################################################################
################################################################################
