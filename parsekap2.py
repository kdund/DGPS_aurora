import numpy as np
import csv as csv
import string as string
from datetime import datetime
import scipy.stats as sps
import glob as glob
import bisect


def parsekap(fname):
    t0 = datetime(1970,1,1)
    
    
    n=0
    f = open(fname,'r')
    lines = f.readlines()
    data=[]
    for i,line in enumerate(lines):
        if "Date" in line:
            ret = np.zeros(7)
            dateline = string.split(line,',')[1]
            exline = string.split(lines[i+1],',')
            eyline = string.split(lines[i+2],',')
            ezline = string.split(lines[i+3],',')
    
            #dt=datetime.strptime(dateline,"%Y-%m-%d:%H:%M:%S")
            dt=datetime.strptime(dateline," %Y-%m-%d %H:%M:%S\r\n")
            dt = (dt-t0).total_seconds()
    
            ret[0]=dt
            ret[1]=exline[2]
            ret[2]=exline[3]
            ret[3]=eyline[2]
            ret[4]=eyline[3]
            ret[5]=ezline[2]
            ret[6]=ezline[3]
            data.append(ret)
    
    data=np.array(data)
    return data

def collateData(datavector):
#function to take N measuring stations and collate the info from each. 
#to return: time,station1,station2,station..N
#with stationi = xerr,xrms,yerr,yrms,zerr,zrms
#important: only choose times where the time difference is less than 30m
#important2: only choose data we have from _all_ stations. 

    dt = 30.*60. #time jitter acceptance

    ndata = len(datavector)
    timesvector = []
    for i,data in enumerate(datavector):
        timesvector.append(data[:,0])

    commontimes =[]
    commonindex =[]
    t0 = timesvector[0]

    for t in t0:
        common=True
        indices =[]
        for times in timesvector:
            index=np.where(abs(times-t)<dt)[0]
            if (len(index)==0):
                common=False
            else: 
                indices.append(index[0])

        if common: 
            commonindex.append(indices)


    commonindex = np.array(commonindex)

    times = np.zeros(len(commonindex))
    for i,data in enumerate(datavector): 
        times += data[commonindex[:,i],0]
    times /= (1.*len(datavector))

    data = times.T

    for i,d in enumerate(datavector): 
        data = np.column_stack([data,d[commonindex[:,i],1:]])

    return data




def collateFiles(filenames):
    datavector=[]
    for filename in filenames: 
        datavector.append(parsekap(filename))
    return collateData(datavector)

def getStationDates(stations=["kap"],folder="."):
#return dates when all stations has data
    
    data = []
    stationdates=[]
    
    mindate = np.zeros(len(stations))
    maxdate = np.zeros(len(stations))
    
    for i,station in enumerate(stations):
        data.append(glob.glob(folder+"/*."+station))
        n = len(data[i])
        dates = np.zeros(n)
        for j in range(n):
            dates[j]=int((data[i][j].replace("."+station,"")).replace(folder+"/",""))
        stationdates.append(dates)
        mindate[i] = np.amin(dates)
        maxdate[i] = np.amax(dates)
    
    mindate = np.amin(mindate)
    maxdate = np.amax(maxdate)
    alldates =  np.arange(mindate,maxdate+1)
    
    countnumber = np.zeros(len(stations))

    commondates = []
    
    for date in alldates: 
        presents = np.zeros(len(stations))
        for i,station in enumerate(stations): 
            if date in stationdates[i]:
                presents[i]=1
    
        if presents.sum()==len(stations):
            commondates.append(date)

    commondates = np.array(commondates)
    return commondates

    

def timestackDGPS(stations = ["kap"],folder='.'):
#function to take a list of dgps stations, and collect data for all dates with all three stations
    commondates = getStationDates(stations,folder)

    datedata =[]
    for date in commondates:
        names = []
        for station in stations: 
            names.append(folder+"/%i."%date+station)
        datedata.append(collateFiles(names))


    return np.vstack(datedata)



def readEISCAT(fdname):
    firstline = True
    t0 = datetime(1970,1,1)
    ddata =[]
    with open(fdname,'rb') as csvfile:
        reader = csv.reader(csvfile,delimiter=' ')
        csvfile.readline() #skip 1st line
        csvfile.readline() #skip 2st line
        csvfile.readline() #skip 3st line
        #for row in reader: 
        for line in csvfile:
            row = line.split()
            date = row[0]
            time = row[1]
            time = date+":"+time
            dt=datetime.strptime(time,"%Y-%m-%d:%H:%M:%S")
            dt = (dt-t0).total_seconds()
    
    
            data=[dt,float(row[3]),float(row[4]),float(row[5]),float(row[6]),]
            ddata.append(data)

    ddata=np.array(ddata)
    return ddata

def shapeECAT(ecatdata,dgpsdata):
#finds the ecat that corresponds to each dgps datapoint
#return: (target,data), where target is the 1h average of 
    ndgps = len(dgpsdata)
    necat = len(ecatdata)
    dteiscat = ecatdata[1,0]-ecatdata[0,0] #delta t for ecat
    dieiscat = int(3600./dteiscat) #number of integers to shift per h
    dieiscat2 = int(dieiscat/2) #number of integers to go up/down 

    meancosz = np.zeros(ndgps)
    meantec  = np.zeros(ndgps)


    teiscat = ecatdata[:,0]
    tdgps = dgpsdata[:,0]

    eiscatindex = np.zeros(len(tdgps))

    for i,t in enumerate(tdgps):
        eiscatindex[i]= bisect.bisect(teiscat,t)


    for i in range(ndgps):
        jmin = eiscatindex[i]-dieiscat2
        jmax = eiscatindex[i]+dieiscat2
        jmin = max(0,jmin)
        jmax = min(necat,jmax) #avoid running out of eiscat
        
        meancosz[i] = np.average(ecatdata[jmin:jmax,1])
        tecs=(ecatdata[jmin:jmax,4])
        #tecs=(ecatdata[jmin:jmax,3]/ecatdata[jmin:jmax,2])
        meantec[i] =  np.average(tecs[np.nonzero(tecs)])

    data = np.column_stack([dgpsdata,meancosz])

    return (meantec,data)


def readData(dgpsstations,dgpsfolder,eiscatname):
    dgpsdata = timestackDGPS(dgpsstations,dgpsfolder)
    eiscatdata = readEISCAT(eiscatname)

    tec,data=shapeECAT(eiscatdata,dgpsdata)
    time = data[:,0]
    data = data[:,1:]

    #remove NANs- periods with no EISCAT data: 
    
    indices = ~np.isnan(tec)
    tec = tec[indices]
    time = time[indices]
    data = data[indices]
    return tec,time,data

def chunkData(arrs,alphas):
    #assume all arrs equal
    n = len(arrs[0])
    indices = np.arange(n)
    np.random.shuffle(indices) 
    ret=[]
    for arr in arrs:
        reta =[]
        for i in range(len(alphas)+1):
            imin=0.
            imax=n
            if i!=0:
                imin = int(n*alphas[i-1])
            if i<len(alphas):
                imax = int(n*alphas[i])
            reta.append(arr[indices[imin:imax]])
        ret.append(reta)
    return ret



    



if __name__=="__main__":
    #load data: 
    stations = ["bju","kap","got","hol","hob"]
    folder ="FTPDL"
    eiscatname = "EISCAT201205to201605.txt"

    split =[0.5,0.75]
    if True: 

        tec,time,data = readData(stations,folder,eiscatname)
        np.save("tec.npy",tec)
        np.save("time.npy",time)
        np.save("data.npy",data)


    tec = np.load("tec.npy")
    time = np.load("time.npy")
    data = np.load("data.npy")

    if True: 
        for i in range(data.shape[1]):
            mean = np.mean(data[:,i])
            sdev = np.std(data[:,i])
            data[:,i]-=mean
            data[:,i]/=sdev

    if False:
        data[:,-1]=0.

    if False: 
        data[:,:-1]=0.

    print tec.shape
    print time.shape
    print data.shape
    teccs,timecs,datacs = chunkData([tec,time,data],split)
    (tec_train,tec_test,tec_validate)=teccs
    (time_train,time_test,time_validate)=timecs
    (data_train,data_test,data_validate)=datacs

    #linar fit: 
    from sklearn.svm import SVR, LinearSVC
    from sklearn.linear_model import LogisticRegression,LinearRegression
    from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
    from sklearn import gaussian_process

    #fitter = SVR(kernel='rbf',C=1e2)
    #fitter = AdaBoostRegressor(n_estimators=50)
    #fitter = gaussian_process.GaussianProcess()
    fitter = LinearRegression()





    tec_validate_fit = fitter.fit(data_train,tec_train).predict(data_validate)

    print fitter.get_params(deep=True)
    #coefs = fitter.coef_
    #print abs(coefs[0:6]).sum()
    #print abs(coefs[6:12]).sum()
    #print abs(coefs[12:18]).sum()
    #print coefs[-1]

    #MSE: 
    mse = np.mean((tec_validate_fit-tec_validate)**2)
    print "smse",np.sqrt(mse)
    #print fitter.coef_


    #plot 
    import matplotlib.pyplot as plt

    fig,axs = plt.subplots(2,2)

    ax0=axs.flatten()[0]
    ax1=axs.flatten()[1]
    ax2=axs.flatten()[2]
    ax3=axs.flatten()[3]

    ax0.scatter(data_train[:,-1],tec_train,alpha=0.5)

    ax1.scatter(data_validate[:,-1],tec_validate_fit/tec_validate)
    ax2.scatter(tec_validate,tec_validate_fit,alpha=0.5)
    #ax2.plot([0,30],[0,30])
    ax2.set_aspect('equal')
    ax3.hist(tec_validate_fit/tec_validate)

    plt.savefig("fittest.pdf")











