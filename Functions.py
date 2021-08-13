V210604
import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from collections import OrderedDict as odict
import time

#Get the name of events as a dictionary, the event names are sorted. 
def GetEventPaths(Path):

    PathsDict = {}
    FolderNames = os.listdir(Path)

    for FolderName in FolderNames:
        Ind = FolderName.find('MODEL')
        if Ind > -1:
            EventName = FolderName[Ind+8:]
            PathsDict[EventName] = os.path.join(Path, FolderName)
    return dict(sorted(PathsDict.items()))
    
def ReadData(Path, EventName, FileNo, ObjNo=None):
    
    PathsDict = GetEventPaths(Path)
    SubPaths = PathsDict[EventName]
    
    FileNames = os.listdir(SubPaths)
    FileNames.sort()
    
    Data = read_snana_fits(os.path.join(SubPaths,FileNames[FileNo*2]),os.path.join(SubPaths,FileNames[FileNo*2+1]))
    
    if ObjNo == None:
        return Data
    else:
        return Data[ObjNo]


# FileNo and ObjNo given in 0, 1, 2, 3... to N-1. 
def PlotLightCurve(Path, EventName , Band , SeedFile=None, SeedObj=None, Prop='SIM_MAGOBS' ,FileNo=None, ObjNo=None, ls=None, marker=None, sharex=True, sharey='none', xlim=None, ylim=None): 
    """
    asfasfasfdasf.
    """
    
    if len(Band) == 1:
        Band = Band+' '
                
    if Band not in ['u ', 'g ', 'r ', 'i ', 'z ', 'Y ']:
        print('The band input does not exist.')
        return
    
    if Prop not in ['SIM_MAGOBS', 'FLUXCAL']:
        print("The input for Prop should be 'SIM_MAGOBS' or 'FLUXCAL'")
        return

    PathsDict = GetEventPaths(Path)
    
    if EventName == '?':
        return list(PathsDict.keys())        

    SubPaths = PathsDict[EventName]
        
    FileNames = os.listdir(SubPaths)
    FileNames.sort()
    
    Mask = ['HEAD' in FileName for FileName in FileNames]
    TotFileNo = sum(Mask)
    
    if FileNo is None:
        if SeedObj is not None:
            np.random.seed(SeedFile)
        FileNo = np.random.randint(0,TotFileNo-1,5)
    else:
        if any([II>TotFileNo for II in FileNo]):
            print('The FileNo excceed the ')     
            
    if ObjNo is None:
        if SeedObj is not None:
            np.random.seed(SeedObj)
        ObjNoRatio = np.random.rand(5)
        FigNo = len(FileNo) * len(ObjNoRatio)
    else:
        FigNo = len(FileNo) * len(ObjNo)
        
    RowNo, ColNo = RowColNo(FigNo)
        
    fig, axs = plt.subplots( RowNo, ColNo, figsize = (6*ColNo, 4*RowNo,), sharex=sharex, sharey=sharey)
    fig.subplots_adjust(hspace=0, wspace=0)
    
    for II, FNo in enumerate(FileNo):        
        Data = read_snana_fits(os.path.join(SubPaths,FileNames[FNo*2]),os.path.join(SubPaths,FileNames[FNo*2+1]))
        
        if ObjNo is None:
            ObjNo = ObjNoRatio*len(Data)
            ObjNo = ObjNo.astype(int)

        for JJ, ONo in enumerate(ObjNo):
            Mask = Data[ONo]['BAND'] == Band
#             Mask*= Data[ONo]['MJD'] > 53095
            
            axs[II, JJ].plot(Data[ONo]['MJD'][Mask], Data[ONo][Prop][Mask], ls = ls, marker = marker)
            axs[II, JJ].text(0.99, 0.99, 'FileNo {}, ObjNo{}'.format(FNo, ONo), ha='right', va='top', transform=axs[II, JJ].transAxes)
            
    if xlim != None:
        sharex = True
        axs[0,0].set_xlim(xlim)
            
    if ylim != None:
        sharey = True
        axs[0,0].set_ylim(ylim)        
        
    if Prop == 'SIM_MAGOBS':
        
        if sharey == True:
            axs[0,0].invert_yaxis()
        else:
            for ax in axs.flatten():
                ax.invert_yaxis()    
            
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False )

    plt.xlabel('MJD', fontsize=15 )
    plt.ylabel(Prop, fontsize=15)


#Count the sizes of the HEAD files and PHOT files respectively.
def GetFileSize(Path, output=0, plot=1):
    
    SizesH = []
    SizesP = []
    
    PathsDict = GetEventPaths(Path)
    
    for Event in PathsDict:

        SubPath = PathsDict[Event]
        FileNames = os.listdir(SubPath)

        SizeH = 0
        SizeP = 0

        for FileName in FileNames:

            if FileName.find('HEAD') > -1:

                FilePath = os.path.join(SubPath, FileName)
                SizeH = SizeH + os.path.getsize(FilePath)

            elif FileName.find('PHOT') > -1:
                FilePath = os.path.join(SubPath, FileName)
                SizeP = SizeP + os.path.getsize(FilePath)

        SizesH.append(SizeH)
        SizesP.append(SizeP)
            
    SizeHMB = np.array(SizesH) / 1024**2
    SizePMB = np.array(SizesP) / 1024**2
    
    if plot == 1:
    
        plt.figure(figsize=(15,5))

        plt.bar(PathsDict.keys(), SizeHMB)
        plt.xticks(rotation = 60) 
        plt.xlabel('Event Names')
        plt.ylabel('Sizes of Head Files (MB)')
        plt.title('The total size is {} MB'.format(SizeHMB.sum()))

        plt.figure(figsize=(15,5))

        plt.bar(PathsDict.keys(), SizePMB)
        plt.xticks(rotation = 60) 
        plt.xlabel('Event Names')
        plt.ylabel('Sizes of Phot Files (MB)')
        plt.title('The total size is {} MB'.format(SizePMB.sum()))
        
    if output == 1:
        return SizeHMB, SizePMB
    
#Count the observation numbers of each object contained in the given path
def GetObsNum(Path, output=0, plot=1):
    
    PathsDict = GetEventPaths(Path)
    Bands = ['u ', 'g ', 'r ', 'i ', 'z ', 'Y ']
    
    ObjectNums = []
    
    ObsNums =  [ [] for _ in range(len(Bands)) ]
    ObsNumStd = [ [] for _ in range(len(Bands)) ]
    
    for Event in PathsDict:
        
        print('Counting {:<25}'.format(Event+':'), end='')
        start = time.time()

        SubPath = PathsDict[Event]
        FileNames = os.listdir(SubPath)

        ObjectNum = 0
        ObjObsNums = [ [] for _ in range(len(Bands)) ]

        for FileName in FileNames:

            Ind = FileName.find('HEAD')

            if Ind > -1:                

                print('|', end='')

                FileNamePHOT = FileName[:Ind] + 'PHOT.FITS.gz'

                HeadFilePath = os.path.join(SubPath, FileName)
                PhotFilePath = os.path.join(SubPath, FileNamePHOT)        
                Data = read_snana_fits(HeadFilePath, PhotFilePath)

                FileObjNum = len(Data)
                ObjectNum = ObjectNum + FileObjNum        

                for Obj in Data:
                    
                    for II, Band in enumerate(Bands):
                        
                        Mask = Obj['BAND'] == Band
                        ObjObsNums[II].append(Mask.sum())

        ObjectNums.append(ObjectNum)
        
        ObjObsNumsMean = np.mean(ObjObsNums, 1)
        ObjObsNumsStd = np.std(ObjObsNums, 1, ddof=1)        
        
        for II in range(len(Bands)):

            ObsNums[II].append(ObjObsNumsMean[II])
            ObsNumStd[II].append(ObjObsNumsStd[II])
        
#         [ [ObsNums[II].append(ObjObsNumsMean[II]), [ObsNumStd[II].append(ObjObsNumsStd[II])] ] for II in range(len(Bands)) ]
        
        end = time.time()

        print('\t time spent: {0:6.3f} s'.format(end-start))
        
    if plot == 1:

        plt.figure(figsize=(15,5))

        plt.bar(PathsDict.keys(), ObjectNums)
        plt.xticks(rotation = 60) 
        plt.xlabel('Event Names')
        plt.ylabel('Number of Objects')

        fig, axs = plt.subplots(6, 1, figsize=(15,15), sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0)

        for II, ax in enumerate(axs.flat):

            ax.bar(PathsDict.keys(), ObsNums[II])
            ax.errorbar(PathsDict.keys(), ObsNums[II], ObsNumStd[II], fmt='o', color = 'r')
            ax.text(0.99, 0.99, 'Band '+Bands[II], transform=ax.transAxes, ha='right', va='top')
            
            ax.tick_params(axis='x', rotation=60)
            
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False )

#         plt.xlabel('Event Names', fontsize=15 )
        plt.ylabel('Number of Observations', fontsize=15)
#         plt.title('Statistics for Observation Numbers', fontsize = 20, loc='center')
        
        if output == 1:
            
            return PathsDict.keys(), ObsNums, ObsNumStd
        
        
        
# Find out the gaps between observations
def GetObsGaps(Path, Band, Events=None, output=0, plot=1):

    PathsDict = GetEventPaths(Path)
    EventNames = list(PathsDict.keys())
    
    GapMean = []
    GapStd = []
    
            
    if len(Band) == 1:
        Band = Band+' '
        
    if Band not in ['u ', 'g ', 'r ', 'i ', 'z ', 'Y ']:
        print('The band input does not exist.')
        return
    
    if Events == None:
        Events = list(PathsDict.keys())
    
    for Event in Events:

        print('Counting {:<25}'.format(Event+':'), end='')
        start = time.time()

        SubPath = PathsDict[Event]
        FileNames = os.listdir(SubPath)

        #Start counting one type of events.

        Gaps = []

        for FileName in FileNames:

            Ind = FileName.find('HEAD')

            if Ind > -1:

                print('|', end='')

                FileNamePHOT = FileName[:Ind] + 'PHOT.FITS.gz'

                HeadFilePath = os.path.join(SubPath, FileName)
                PhotFilePath = os.path.join(SubPath, FileNamePHOT)        
#                 Data = np.array(read_snana_fits(HeadFilePath, PhotFilePath))
                Data = read_snana_fits(HeadFilePath, PhotFilePath)
#                 Data = np.array(Data)
        
                for Obj in Data:
                    Mask = Obj['BAND']==Band
                    Gaps.append(Obj['MJD'][Mask][1:] - Obj['MJD'][Mask][:-1])    

        GapsArray = np.array(Gaps)

        GapMean.append(GapsArray.mean(axis=0))
        GapStd.append(GapsArray.std(axis=0, ddof=1))

        end = time.time()

        print('\t time spent: {0:6.3f} s'.format(end-start))

    GapMean = [np.round(II, 2) for II in GapMean]
    GapStd = [np.round(II, 2) for II in GapStd]

    FigNo = len(Events)

    RowNo, ColNo = RowColNo(FigNo)
      
    fig, axs = plt.subplots( RowNo, ColNo, figsize = (20, 4*RowNo,), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0, wspace=0)

    axflat = axs.flatten()
    
    for II in range(FigNo):
        
        X = np.arange(len(GapMean[II]))
        axflat[II].plot(X, GapMean[II])
        axflat[II].text(0.05, 0.95, Events[II], fontsize=10, ha='left', va='top', transform=axflat[II].transAxes)

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False )

    plt.xlabel("Number of observation", fontsize=15)
    plt.ylabel("Observation gaps of band {}(MJD)".format(Band), fontsize=15)


      



    
    
# Find out the gaps between observations, convert Data to Array at the beginning. 
def GetObsGaps2(Path, Band, Events=None, output=0, plot=1):

    PathsDict = GetEventPaths(Path)
    
    GapMean = []
    GapStd = []
    
            
    if len(Band) == 1:
        Band = Band+' '
        
    if Band not in ['u ', 'g ', 'r ', 'i ', 'z ', 'Y ']:
        print('The band input does not exist.')
        return
    else:
        Band = np.bytes_(Band)
        
    if Events == None:
        Events = list(PathsDict.keys())
    
    for Event in Events:

        print('Counting {:<25}'.format(Event+':'), end='')
        start = time.time()

        SubPath = PathsDict[Event]
        FileNames = os.listdir(SubPath)

        #Start counting one type of events.

        Gaps = []

        for FileName in FileNames:

            Ind = FileName.find('HEAD')

            if Ind > -1:

                print('|', end='')

                FileNamePHOT = FileName[:Ind] + 'PHOT.FITS.gz'

                HeadFilePath = os.path.join(SubPath, FileName)
                PhotFilePath = os.path.join(SubPath, FileNamePHOT)        

                Data = read_snana_fits(HeadFilePath, PhotFilePath)
                Data = np.array([np.array(tabl) for tabl in Data])
                
                ObjNoInFile = len(Data)
                
                Mask = Data['BAND'] == Band
                DataBandMJD = Data['MJD'][Mask].reshape(ObjNoInFile, -1)

                Gaps.append(DataBandMJD[:, 1:] - DataBandMJD[:, :-1])

        GapsArray = np.array(Gaps).reshape(len(Gaps)*len(Gaps[0]), -1)

        GapMean.append(GapsArray.mean(axis=0))
        GapStd.append(GapsArray.std(axis=0, ddof=1))

        end = time.time()

        print('\t time spent: {0:6.3f} s'.format(end-start))

    GapMean = [np.round(II, 2) for II in GapMean]
    GapStd = [np.round(ii, 2) for II in GapStd]
        
    FigNo = len(Events)

    RowNo, ColNo = RowColNo(FigNo)
      
    fig, axs = plt.subplots( RowNo, ColNo, figsize = (20, 4*RowNo,), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0, wspace=0)

    axflat = axs.flatten()
    
    for II in range(FigNo):
        
        X = np.arange(len(GapMean[II]))
        axflat[II].plot(X, GapMean[II])
        axflat[II].text(0.05, 0.95, Events[II], fontsize=10, ha='left', va='top', transform=axflat[II].transAxes)

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False )
    
    
#Find the range in time of the objects.
def GetTimeRange(Path, Band, Events=None, Prop='MJD', output=0, plot=1):
    
    print('bbb')

    PathsDict = GetEventPaths(Path)
    EventNames = list(PathsDict.keys())
    
    StartMean = []
    StartStd = []
    EndMean = []
    EndStd = []    
            
    if len(Band) == 1:
        Band = Band+' '
        
    if Band not in ['u ', 'g ', 'r ', 'i ', 'z ', 'Y ']:
        print('The band input does not exist.')
        return
    
    if Events == None:
        Events = list(PathsDict.keys())
    
    for Event in Events:

        print('Counting {:<25}'.format(Event+':'), end='')
        start = time.time()

        SubPath = PathsDict[Event]
        FileNames = os.listdir(SubPath)

        #Start counting one type of events.

        Gaps = []

        Starts = []
        Ends = []

        for FileName in FileNames:

            Ind = FileName.find('HEAD')

            if Ind > -1:

                print('|', end='')

                FileNamePHOT = FileName[:Ind] + 'PHOT.FITS.gz'

                HeadFilePath = os.path.join(SubPath, FileName)
                PhotFilePath = os.path.join(SubPath, FileNamePHOT)        
#                 Data = np.array(read_snana_fits(HeadFilePath, PhotFilePath))
                Data = read_snana_fits(HeadFilePath, PhotFilePath)
#                 Data = np.array(Data)
        
                for Obj in Data:
                    Mask = Obj['BAND']==Band

                    Starts.append(Obj['MJD'][Mask][0])
                    Ends.append(Obj['MJD'][Mask][-1])

        GapsArray = np.array(Gaps)
        
        StartMean.append(np.mean(Starts))
        StartStd.append(np.std(Starts, ddof=1))

        EndMean.append(np.mean(Ends))
        EndStd.append(np.std(Ends, ddof=1))

        end = time.time()
        print('\t time spent: {0:6.3f} s'.format(end-start))

    if plot == 1:

        plt.figure(figsize=(15,5))

        plt.plot(Events, StartMean)
        plt.errorbar(Events, StartMean, StartStd)
        
        plt.plot(Events, EndMean)
        plt.errorbar(Events, EndMean, EndStd)

        plt.xticks(rotation = 60) 
        plt.xlabel('Event Names')
        plt.ylabel('Start and End Time')

    # GapMean = [np.round(II, 2) for II in GapMean]
    # GapStd = [np.round(II, 2) for II in GapStd]

    # FigNo = len(Events)

    # RowNo, ColNo = RowColNo(FigNo)
      
    # fig, axs = plt.subplots( RowNo, ColNo, figsize = (20, 4*RowNo,), sharex=True, sharey=True)
    # fig.subplots_adjust(hspace=0, wspace=0)

    # axflat = axs.flatten()
    
    # for II in range(FigNo):

    #     axflat[II].plot(X, GapMean[II])
    #     axflat[II].text(0.05, 0.95, Events[II], fontsize=10, ha='left', va='top', transform=axflat[II].transAxes)

    # fig.add_subplot(111, frame_on=False)
    # plt.tick_params(labelcolor="none", bottom=False, left=False )

    # plt.xlabel("Number of observation", fontsize=15)
    # plt.ylabel("Observation gaps of band {}(MJD)".format(Band), fontsize=15)
    
    if output == 1:
        return StartMean, StartStd, EndMean, EndStd
    
    
#plot the Presto Diagram of a selected event.    
def PlotPrestoDiagram(Path, EventName, Band1, Band2, dT1, dT2, Thr=30, SamplingInterval=5):

    if dT1 < 0:
        print('dT1 should be larger than 0!')
        return

    fig, ax = plt.subplots(1,1, figsize=[10,5])
    ax.set_xlabel("$\Delta {0}$".format(Band1), fontsize=10)
    ax.set_ylabel('${0}-{1}$'.format(Band1,Band2), fontsize=10)

    dT1P = int(dT1 / SamplingInterval) #The difference of time in points of data. 
    dT2P = int(dT2 / SamplingInterval)

    Band1 = BandInNo(Band1) #Convert band letters to numbers, 'u'=1, 'g'=2, and so on
    Band2 = BandInNo(Band2)

    DataNum = np.load(os.path.join(Path, EventName+'_Num.npy'))

    for Obj in DataNum:

        Mask = Obj[Band1] < Thr
        Ind = np.where(Mask)[0]
        if len(Ind) == 0:
            continue
        Mask1 = np.zeros_like(Mask)
#         print(Obj[Band1])
#         print(Ind)
    
        Mask1[Ind[0]: Ind[-1] - dT1P] = True

        Mask = Obj[Band2] < Thr
        Ind = np.where(Mask)[0]
#         print(Mask)
        if len(Ind) == 0:
            continue        
        Mask2 = np.zeros_like(Mask)
        Mask2[Ind[0]: Ind[-1]] = True
        
        Mask1Trans2 = np.hstack( (Mask1[-dT2P:], Mask1[:-dT2P]) )
        Mask1Trans2 = Mask1Trans2 & Mask2
        Mask1 = np.hstack( (Mask1Trans2[dT2P:], Mask1Trans2[:dT2P]) )

        Mask1Trans1 = np.hstack( (Mask1[-dT1P:], Mask1[:-dT1P]) )

        dMag = Obj[Band1][Mask1] - Obj[Band1][Mask1Trans1]
        Color = Obj[Band1][Mask1] - Obj[Band2][Mask1Trans2]

        ax.plot(dMag, Color, '.', color='b')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def BandInNo(Band):
    BandDict = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'Y': 5}
    return BandDict[Band]    
    
    
# Return the row and column numbers of a subplot according to the number of the plots.
def RowColNo(FigNo):
    
    if FigNo<7:
        ColNo = 2
    elif FigNo<13:
        ColNo = 3
    elif FigNo<21:
        ColNo = 4
    else:
        ColNo = 5
    RowNo = int(np.ceil(FigNo/ColNo))
    return RowNo, ColNo
    
#Funtion from https://sncosmo.readthedocs.io/en/v2.1.x/api/sncosmo.read_snana_fits.html
def read_snana_fits(head_file, phot_file, snids=None, n=None):

    # Should we memmap? Only if we're going to read only a part of the file
    memmap = (snids is not None or n is not None)

    # Get metadata for all the SNe
    head_data = fits.getdata(head_file, 1, view=np.ndarray)
    phot_data = fits.getdata(phot_file, 1, view=np.ndarray, memmap=memmap)

    # Strip trailing whitespace characters from SNID.
    if 'SNID' in head_data.dtype.names:
        try:
            head_data['SNID'][:] = np.char.strip(head_data['SNID'])
        except TypeError:
            pass

    # Check which indicies to return.
    if snids is None and n is None:
        idx = range(len(head_data))
    elif n is None:
        if 'SNID' not in head_data.dtype.names:
            raise RuntimeError('Specific snids requested, but head file does'
                               ' not contain SNID column')
        idx = []
        for snid in snids:
            i = np.flatnonzero(head_data['SNID'] == snid)
            if len(i) != 1:
                raise RuntimeError('Unique snid requested, but there are '
                                   '{0:d} matching entries'.format(len(i)))
            idx.append(i[0])
    elif snids is None:
        idx = range(n)
    else:
        raise ValueError("cannot specify both 'snids' and 'n' arguments")

    # Loop over SNe in HEAD file
    sne = []
    for i in idx:
        meta = odict(zip(head_data.dtype.names, head_data[i]))

        j0 = head_data['PTROBS_MIN'][i] - 1
        j1 = head_data['PTROBS_MAX'][i]
        data = phot_data[j0:j1]
        if 'FLT' in data.dtype.names:
            data['FLT'][:] = np.char.strip(data['FLT'])
        sne.append(Table(data, meta=meta, copy=False))

    return sne