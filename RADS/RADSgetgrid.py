import numpy as np
from time import strptime
from calendar import timegm
import os

from badpass import RejectPass


class RADSgrid():

    def __init__(self, path: str, pathtype: str="Abs", epoch: int=1640995200, skip: int=12, autoreject: bool= False):
        if pathtype != "Abs":
            dirname = os.path.dirname(__file__)
            path = os.path.join(dirname, path)
        if not path.endswith("/"):
            path += "/"
        self.path = path
        self.epoch = epoch #recommended to be unix time of jan-1-%year 0:00:00, this gives enough precision
        self.skip = skip
        self.autoreject = autoreject

        self.dateline = 6
        self.month = [None,"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"] #might not be actual abbreviations used


    def getday(self, day: int) -> list:
            """
            input: day relative to epoch
            output: [array, ..., array] for arrays in day selected
            """
            time = (day-1)*86400

            output = []

            #----- find all datafiles in path
            
            for filename in os.listdir(self.path):
                #print(filename)
                if filename.endswith(".asc"):
                    #-----extract date and time
                    with open(self.path+filename) as f:
                        for index,line in enumerate(iter(f)):
                            if index+1 == self.dateline:
                                filedate = line.split()[4][1:].split("-")
                                filetime = line.split()[5][:-1]


                    #-----get date and time in unix time
                    try:
                        filedate[1] = self.month.index(filedate[1])
                    except:
                        Exception("MonthAbbreviation not found")
                    filedate = "-".join(str(x) for x in filedate)

                    timestamp = timegm(strptime(filedate+" "+filetime, '%d-%m-%Y %H:%M:%S'))-self.epoch #"convert" to a unix epoch of %year for better time precision in float64 as float128 is not available in Windows

                    if time<timestamp<time+86400:

                        #-----timestamp conversion to absolute time relative to set epoch
                        data = np.loadtxt(self.path+filename, skiprows = self.skip, ndmin=2)
                        if self.autoreject:
                            if RejectPass(data[:,3], ui=True, filename=filename):
                                continue

                        if data.ndim < 2:
                            data[0] += timestamp
                        else:
                            data[:,0] += timestamp
                        
                        output.append(data)
            return output
    
    def multiday(self, startday: int, n: int=1) -> list:
        """
        input: day relative to epoch, number of days
        output: [array, ..., array] for arrays in days selected
        """
        output = []
        for day in range(startday, startday+n):
            output.extend(self.getday(day))
        return output

    #def getpass(self, Pass: int, Cycle: int=1):



if __name__ == "__main__": 
    #epoch2018 = 1514764800
    #epoch2022 = 1640995200
    #grid = RADSgrid("example","Rel",epoch2018)

    #print(len(grid.multiday(157,7)))

    dir_CRYOSAT2 = "C:/Users/User/Desktop/radsdata/"
    #dir_JASON3 = "C:/Users/User/OneDrive - Delft University of Technology/RADS_data/JASON-3"
    #dir_SARAL = "C:/Users/User/OneDrive - Delft University of Technology/RADS_data/SARAL"
    #dir_SNTNL3A = "C:/Users/User/OneDrive - Delft University of Technology/RADS_data/SNTNL-3A"

    # Initiation of the RADSgrid class for all satellites
    CRYOSAT2 = RADSgrid(dir_CRYOSAT2, autoreject=True)
    #JASON3 = RADSgrid(dir_JASON3)
    #SARAL = RADSgrid(dir_SARAL)
    #SNTNL3A = RADSgrid(dir_SNTNL3A)

    startday=15
    n=7

     # Raw data imported from the RADSgrid class
    raw_data_CRYOSAT2 = CRYOSAT2.multiday(startday, n)
    #raw_data_JASON3 = JASON3.multiday(startday, n)
    #raw_data_SARAL = SARAL.multiday(startday, n)
    #raw_data_SNTNL3A = SNTNL3A.multiday(startday, n)
    """
    raw_data_join_CRYOSAT2 = np.concatenate(raw_data_CRYOSAT2, axis=0)
    raw_data_join_JASON3 = np.concatenate(raw_data_JASON3, axis=0)
    raw_data_join_SARAL = np.concatenate(raw_data_SARAL, axis=0)
    raw_data_join_SNTNL3A = np.concatenate(raw_data_SNTNL3A, axis=0)

    np.savetxt('RADS/raw_dat_CRYOSAT2.txt', raw_data_join_CRYOSAT2)
    np.savetxt('RADS/raw_dat_JASON3.txt', raw_data_join_JASON3)
    np.savetxt('RADS/raw_dat_SARAL.txt', raw_data_join_SARAL)
    np.savetxt('RADS/raw_dat_SNTNL3A.txt', raw_data_join_SNTNL3A)
    """