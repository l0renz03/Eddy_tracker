import numpy as np
import scipy as sp
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from global_land_mask import globe

import time
import os

from RADSgetgrid import RADSgrid

from paths import *

def t(): return np.round((time.time()-t_0),2)


def data_initialization(ui:bool=False) -> RADSgrid:

    if ui:
        print('data_initialization initiated...')

    # Directories of relevant RADS satelittes for 2022
    # dir_CRYOSAT2 = "C:/Users/User/OneDrive - Delft University of Technology/RADS_data/CRYOSAT2"
    # dir_JASON3 = "C:/Users/User/OneDrive - Delft University of Technology/RADS_data/JASON-3"
    # dir_SARAL = "C:/Users/User/OneDrive - Delft University of Technology/RADS_data/SARAL"
    # dir_SNTNL3A = "C:/Users/User/OneDrive - Delft University of Technology/RADS_data/SNTNL-3A"

    # Initiation of the RADSgrid class for all satellites
    autoreject = True
    CRYOSAT2 = RADSgrid(dir_CRYOSAT2, autoreject=autoreject)
    JASON3 = RADSgrid(dir_JASON3, autoreject=autoreject)
    SARAL = RADSgrid(dir_SARAL, autoreject=autoreject)
    SNTNL3A = RADSgrid(dir_SNTNL3A, autoreject=autoreject)

    if ui:
        print(f'data_initiation Done! [t={t()} s]')

    return CRYOSAT2, JASON3, SARAL, SNTNL3A


def getRADSrawAllSat(startday:int, n:int=1, memory:bool=True, dir_raw_memory:str="RADS/getRADSrawAllSat_memory/", ui:bool=False, data_filter:bool=True) -> np.array:
    """return a Numpy array with raw data from the four satellites starting at startday and for a span of n days

    columns in output array: TIME / LAT / LON / SLA

    memory=True to check if the calculation has already been done and loading the data directly
    dir_raw_memory: location of stored raw data for the memory check

    data_filter: filter out erroneous data (+/- 3 std)

    optional: ui=True to print progress of function
    """

    if ui:
        print(f'getRADSrawAllSat initiated with startday={startday} and n={n} [t={t()} s]')
        print(f'Checking existing memory for data: [t={t()} s]')

    # checking if function has been run before with same parameters
    if memory:
        for file in os.listdir(dir_raw_memory):
            if file == f'getRADSrawAllSat_{startday}_{n}.npy':
                raw_data_AllSat = np.load(dir_raw_memory+file)
                if ui:
                    print(f'Data found: getRADSrawAllData Done! [t={t()} s]')
                return raw_data_AllSat

    if ui:
        print(f'Data NOT found! [t={t()} s]')
        print(f'Importing raw data: [t={t()} s]')

    CRYOSAT2, JASON3, SARAL, SNTNL3A = data_initialization(ui)

    # Raw data imported from the RADSgrid class
    raw_data_CRYOSAT2 = CRYOSAT2.multiday(startday, n)
    if ui: print(f' - CRYOSAT2: Done! [t={t()} s]')
    raw_data_JASON3 = JASON3.multiday(startday, n)
    if ui: print(f' - JASON3: Done! [t={t()} s]')
    raw_data_SARAL = SARAL.multiday(startday, n)
    if ui: print(f' - SARAL: Done! [t={t()} s]')
    raw_data_SNTNL3A = SNTNL3A.multiday(startday, n)
    if ui: print(f' - SNTNL3A: Done! [t={t()} s]')

    print(np.shape(raw_data_CRYOSAT2), np.shape(raw_data_JASON3), np.shape(raw_data_SARAL), np.shape(raw_data_SNTNL3A))

    if ui:
        print(f'Importing raw data: Done! [t={t()} s]')
        print(f'Joining raw data: [t={t()} s]')

    # Raw data joined into one Numpy array
    if np.shape(raw_data_CRYOSAT2)[0] != 0:
        raw_data_join_CRYOSAT2 = np.concatenate(raw_data_CRYOSAT2, axis=0)
    else: raw_data_join_CRYOSAT2 = np.zeros((0, 4))
    if ui: print(f' - CRYOSAT2: Done! [t={t()} s]')
    if np.shape(raw_data_JASON3)[0] != 0:
        raw_data_join_JASON3 = np.concatenate(raw_data_JASON3, axis=0)
    else: raw_data_join_JASON3 = np.zeros((0, 4))
    if ui: print(f' - JASON3: Done! [t={t()} s]')
    if np.shape(raw_data_SARAL)[0] != 0:
        raw_data_join_SARAL = np.concatenate(raw_data_SARAL, axis=0)
    else: raw_data_join_SARAL = np.zeros((0, 4))
    if ui: print(f' - SARAL: Done! [t={t()} s]')
    if np.shape(raw_data_SNTNL3A)[0] != 0:
        raw_data_join_SNTNL3A = np.concatenate(raw_data_SNTNL3A, axis=0)
    else: raw_data_join_SNTNL3A = np.zeros((0, 4))
    if ui: print(f' - SNTNL3A: Done! [t={t()} s]')

    if ui:
        print(f'Joining raw data: Done! [t={t()} s]')
        print(f'Joining all satellites into one array: [t={t()} s]')

    # Join all satellites into one Numpy array
    raw_data_join_AllSat = np.concatenate([
        raw_data_join_CRYOSAT2,
        raw_data_join_JASON3,
        raw_data_join_SARAL,
        raw_data_join_SNTNL3A
    ])

    if ui:
        print(f'Done! [t={t()} s]')
        print(f'Filtering out NaN values: [t={t()} s]')

    # filtering out NaN values
    raw_data_AllSat = raw_data_join_AllSat[~np.isnan(raw_data_join_AllSat).any(axis=1)]

    # filtering out erroneous data
    if data_filter:
        if ui: print(f'Filtering data beyond 3 stds: [t={t()} s]')
        raw_data_AllSat_sla = raw_data_AllSat[:, 3]
        sla_mean = raw_data_AllSat_sla.mean()
        sla_std = raw_data_AllSat_sla.std()
        raw_data_AllSat = raw_data_AllSat[(raw_data_AllSat_sla>=sla_mean-3*sla_std) & (raw_data_AllSat_sla<=sla_mean+3*sla_std)]
        if ui: print(f'Filtering: Done! (mean = {sla_mean}, std = {sla_std}) [t={t()} s]')

    if memory:
        if ui: print(f'Saving data in memory for future use: [t={t()} s]')
        np.save(f'{dir_raw_memory}getRADSrawAllSat_{startday}_{n}.npy', raw_data_AllSat)
        if ui: print(f'Done! [t={t()} s]')

    if ui:
        print(f'getRADSrawAllData Done! [t={t()} s]')
    

    return raw_data_AllSat


def getRADSgridAllSat(startday:int, n:int=1, grid_resolution:float=0.1, memory:bool=True, extents:tuple=(-85, 25, 0, 70), interpolation_method:str='linear', dir_memory:str="RADS/getRADSgridAllSat_memory/", dir_raw_memory:str="RADS/getRADSrawAllSat_memory/", filter_on_land:bool=True, smoothen:bool=True, smoothen_sigma:float=1., ui:bool=False) -> np.array:
    """return Numpy arrays with grid data from the four satellites starting at startday and for a span of n days

    grid_resolution: resolution of the output grid in degrees
    extents: extents of the output grid (lon_0, lon_1, lat_0, lat_1)
    interpolation_method: method of interpolation for the scipy.interpolate griddata function

    memory=True to check if the calculation has already been done and loading the data directly
    dir_memory: location of stored data for the memory check
    dir_raw_memory: location of stored raw data for the memory check

    filter_on_land: change values on land to NaN
    smoothen: use a scipy.ndimage.gaussian_filter to smoothen the data
    smoothen_sigma: SD input for gaussian_filter

    optional: ui=True to print progress of function
    """

    # extracting coordinates from extents
    lon_0, lon_1, lat_0, lat_1 = extents

    if ui:
        print(f'getRADSgridAllSat initiated with startday={startday} and n={n}, extents=({lon_0}, {lon_1}, {lat_0}, {lat_1}) [t={t()} s]')
        print(f'Checking existing memory for data: [t={t()} s]')

    # checking if function has been run before with same parameters
    if memory:
        for file in os.listdir(dir_memory):
            print(file)
            if file == f'getRADSgridAllSat_{startday}_{n}__{lon_0}_{lon_1}_{lat_0}_{lat_1}.npz':
                AllData_grid = np.load(dir_memory+file)
                sla_grid = AllData_grid['arr_0']
                longitude_grid = AllData_grid['arr_1']
                latitude_grid = AllData_grid['arr_2']
                if ui:
                    print(f'Data found: getRADSrawAllData Done! [t={t()} s]')
                return sla_grid, longitude_grid, latitude_grid

    if ui:
        print(f'Data NOT found! [t={t()} s]')
        print(f'Running getRADSrawAllSat: [t={t()} s]')

    # running getRADSrawAllSat to get raw data for all satellites
    raw_data_AllSat = getRADSrawAllSat(startday, n, memory, dir_raw_memory, ui)
    latitude_raw = raw_data_AllSat[:, 1]
    longitude_raw = raw_data_AllSat[:, 2]
    sla_raw = raw_data_AllSat[:, 3]

    if ui: print(f'Done! [t={t()} s]')

    # resolution of output grid (number of points)
    longitude_resolution = int((lon_1 - lon_0) / grid_resolution)
    latitude_resolution = int((lat_1 - lat_0) / grid_resolution)
    if ui: print(f'Calculating resolution of output grid: Done! [t={t()} s]')

    # list of points according to the resolution
    longitude_steps = np.linspace(lon_0, lon_1, longitude_resolution)
    latitude_steps = np.linspace(lat_0, lat_1, latitude_resolution)
    if ui: print(f'Creating list of point according to grid resolution: Done! [t={t()} s]')

    # creating meshgrid with specified resolution
    longitude_grid, latitude_grid = np.meshgrid(longitude_steps, latitude_steps, indexing='ij')
    if ui: print(f'Creating meshgrid with specified resolution: Done! [t={t()} s]')

    sla_grid = griddata((longitude_raw, latitude_raw), sla_raw, (longitude_grid, latitude_grid), method=interpolation_method)

    if memory:
        if ui: print(f'Saving data in memory for future use: [t={t()} s]')
        np.savez(f'{dir_memory}getRADSgridAllSat_{startday}_{n}__{lon_0}_{lon_1}_{lat_0}_{lat_1}.npz', sla_grid, longitude_grid, latitude_grid)
        if ui: print(f'Done! [t={t()} s]')

    if ui: print(f'applying gaussian filter for smoothing [t={t()} s]')

    # Smoothening image with gaussian_filter
    sla_grid[np.isnan(sla_grid)] = 0.
    if smoothen:
        sla_grid = sp.ndimage.gaussian_filter(sla_grid, sigma=smoothen_sigma, mode='constant')

    if ui: print(f'converting data points on land to NaN [t={t()} s]')

    # Converting data points on land to NaN
    if filter_on_land:
        overlay = globe.is_land(latitude_grid, longitude_grid)
        sla_grid[overlay==True] = np.nan

    if ui:
        print(f'getRADSgridAllSat Done! [t={t()} s]')

    return sla_grid, longitude_grid, latitude_grid


def getRADStimeAllSat3D(startday:int, n:int=1, grid_resolution:float=0.1, memory:bool=True, extents:tuple=(-85, 25, 0, 70), interpolation_method:str='linear', dir_memory:str="RADS/getRADStimeAllSat_memory/", dir_raw_memory:str="RADS/getRADSrawAllSat_memory/", ui:bool=False) -> np.array:
    """return Numpy arrays with grid data from the four satellites starting at startday and for a span of n days

    grid_resolution: resolution of the output grid in degrees
    extents: extents of the output grid (lon_0, lon_1, lat_0, lat_1)
    interpolation_method: method of interpolation for the scipy.interpolate griddata function

    memory=True to check if the calculation has already been done and loading the data directly
    dir_memory: location of stored data for the memory check
    dir_raw_memory: location of stored raw data for the memory check

    optional: ui=True to print progress of function
    """

    # extracting coordinates from extents
    lon_0, lon_1, lat_0, lat_1 = extents

    if ui:
        print(f'getRADStimeAllSat3D initiated with startday={startday} and n={n} [t={t()} s]')
        print(f'Checking existing memory for data: [t={t()} s]')

    # checking if function has been run before with same parameters
    if memory:
        for file in os.listdir(dir_memory):
            if file == f'getRADStimeAllSat_{startday}_{n}__{lon_0}_{lon_1}_{lat_0}_{lat_1}.npz':
                AllData_grid = np.load(dir_memory+file)
                sla_grid = AllData_grid['arr_0']
                longitude_grid = AllData_grid['arr_1']
                latitude_grid = AllData_grid['arr_2']
                time_grid = AllData_grid['arr_3']
                if ui:
                    print(f'Data found: getRADStimeAllSat3D Done! [t={t()} s]')
                return sla_grid, longitude_grid, latitude_grid, time_grid

    if ui:
        print(f'Data NOT found! [t={t()} s]')
        print(f'Running getRADSrawAllSat: [t={t()} s]')

    # running getRADSrawAllSat to get raw data for all satellites
    raw_data_AllSat = getRADSrawAllSat(startday, n, memory, dir_raw_memory, ui)
    time_raw = raw_data_AllSat[:, 0]/(2**19)
    time_raw = time_raw-np.min(time_raw)
    latitude_raw = raw_data_AllSat[:, 1]
    longitude_raw = raw_data_AllSat[:, 2]
    sla_raw = raw_data_AllSat[:, 3]

    # Adding data points with zero values if on_land
    values_on_land = np.load("RADS/land0points.npy")
    time_raw = np.concatenate([time_raw, values_on_land[:, 3]])
    latitude_raw = np.concatenate([latitude_raw, values_on_land[:, 1]])
    longitude_raw = np.concatenate([longitude_raw, values_on_land[:, 2]])
    sla_raw = np.concatenate([sla_raw, values_on_land[:, 0]])

    if ui: print(f'Done! [t={t()} s]')

    # resolution of output grid (number of points)
    longitude_resolution = int((lon_1 - lon_0) / grid_resolution)
    latitude_resolution = int((lat_1 - lat_0) / grid_resolution)
    if ui: print(f'Calculating resolution of output grid: Done! [t={t()} s]')

    # list of points according to the resolution
    time_steps = np.array([(np.max(time_raw)-np.min(time_raw))/2])
    #time_steps = np.linspace(np.min(time_raw), np.max(time_raw), 1)
    longitude_steps = np.linspace(lon_0, lon_1, longitude_resolution)
    latitude_steps = np.linspace(lat_0, lat_1, latitude_resolution)
    if ui: print(f'Creating list of point according to grid resolution: Done! [t={t()} s]')

    # creating meshgrid with specified resolution
    longitude_grid, latitude_grid, time_grid = np.meshgrid(longitude_steps, latitude_steps, time_steps, indexing='ij')
    if ui: print(f'Creating meshgrid with specified resolution: Done! [t={t()} s]')

    # interpolating sla onto new regular grid
    sla_grid = griddata((longitude_raw, latitude_raw, time_raw), sla_raw, (longitude_grid, latitude_grid, time_grid), method=interpolation_method)

    if memory:
        if ui: print(f'Saving data in memory for future use: [t={t()} s]')
        np.savez(f'{dir_memory}getRADStimeAllSat_{startday}_{n}__{lon_0}_{lon_1}_{lat_0}_{lat_1}.npz', sla_grid, longitude_grid, latitude_grid, time_grid)
        if ui: print(f'Done! [t={t()} s]')

    if ui:
        print(f'getRADSgridAllSat3D Done! [t={t()} s]')

    return sla_grid, longitude_grid, latitude_grid, time_grid


def getRADStimeAllSat(startday:int, n:int=1, grid_resolution:float=0.1, memory:bool=True, extents:tuple=(-85, 25, 0, 70), interpolation_method:str='linear', dir_memory:str="RADS/getRADStimeAllSat_memory/", dir_raw_memory:str="RADS/getRADSrawAllSat_memory/", filter_on_land:bool=True, smoothen:bool=True, smoothen_sigma:float=1., ui:bool=False) -> np.array:
    """return Numpy arrays with grid data from the four satellites starting at startday and for a span of n days

    grid_resolution: resolution of the output grid in degrees
    extents: extents of the output grid (lon_0, lon_1, lat_0, lat_1)
    interpolation_method: method of interpolation for the scipy.interpolate griddata function

    memory=True to check if the calculation has already been done and loading the data directly
    dir_memory: location of stored data for the memory check
    dir_raw_memory: location of stored raw data for the memory check

    filter_on_land: change values on land to NaN
    smoothen: use a scipy.ndimage.gaussian_filter to smoothen the data
    smoothen_sigma: SD input for gaussian_filter

    optional: ui=True to print progress of function
    """

    if ui: print('\033[95m'+f'getRADStimeAllSat initiated with startday={startday} and n={n} [t={t()} s]'+'\033[0m')

    # Calling getRADStimeAllSat for 3D arrays
    sla_grid_3D, longitude_grid_3D, latitude_grid_3D, _ = getRADStimeAllSat3D(startday, n, grid_resolution, memory, extents, interpolation_method, dir_memory, dir_raw_memory, ui)

    if ui: print(f'extracting middle layer from getRADStimeAllSat3D [t={t()} s]')

    # Extracting middle slice
    time_span = sla_grid_3D.shape[2]
    time_middle = int(np.floor(time_span / 2))
    sla_grid = sla_grid_3D[:, :, time_middle]
    longitude_grid = longitude_grid_3D[:, :, time_middle]
    latitude_grid = latitude_grid_3D[:, :, time_middle]

    if ui: print(f'applying gaussian filter for smoothing [t={t()} s]')

    # Smoothening image with gaussian_filter
    sla_grid[np.isnan(sla_grid)] = 0.
    if smoothen:
        sla_grid = sp.ndimage.gaussian_filter(sla_grid, sigma=smoothen_sigma, mode='constant')

    if ui: print(f'converting data points on land to NaN [t={t()} s]')

    # Converting data points on land to NaN
    if filter_on_land:
        overlay = globe.is_land(latitude_grid, longitude_grid)
        sla_grid[overlay==True] = np.nan

    if ui: print(f'getRADStimeAllSat Done! [t={t()} s]')

    return sla_grid, longitude_grid, latitude_grid


def RADSgrid2array(startday:int, n:int=1, grid_resolution:float=0.1, memory:bool=True, extents:tuple=(-85, 25, 0, 70), interpolation_method:str='linear', dir_memory:str="RADS/getRADSgridAllSat_memory/", dir_raw_memory:str="RADS/getRADSrawAllSat_memory/", ui:bool=False) -> np.array:
    """convert grid data from the four satellites starting at startday and for a span of n days into 1D arrays

    grid_resolution: resolution of the output grid in degrees
    extents: extents of the output grid (lon_0, lon_1, lat_0, lat_1)
    interpolation_method: method of interpolation for the scipy.interpolate griddata function

    memory=True to check if the calculation has already been done and loading the data directly
    dir_memory: location of stored data for the memory check
    dir_raw_memory: location of stored raw data for the memory check

    optional: ui=True to print progress of function
    """

    if ui:
        print(f'RADSgrid2array initiated with startday={startday} and n={n} [t={t()} s]')

    # calling getRADSgridAllSat
    sla_grid, longitude_grid, latitude_grid = getRADSgridAllSat(startday, n, grid_resolution, memory, extents, interpolation_method, dir_memory, dir_raw_memory, ui)

    # converting grids into 1D arrays
    sla_array = sla_grid.reshape((sla_grid.size,))
    longitude_array = longitude_grid.reshape((longitude_grid.size,))
    latitude_array = latitude_grid.reshape((latitude_grid.size,))

    if ui:
        print(f'RADSgrid2array Done! [t={t()} s]')

    return sla_array, longitude_array, latitude_array


def RADStime2array(startday:int, n:int=1, grid_resolution:float=0.1, memory:bool=True, extents:tuple=(-85, 25, 0, 70), interpolation_method:str='linear', dir_memory:str="RADS/getRADStimeAllSat_memory/", dir_raw_memory:str="RADS/getRADSrawAllSat_memory/", ui:bool=False) -> np.array:
    """convert grid data from the four satellites starting at startday and for a span of n days into 1D arrays

    grid_resolution: resolution of the output grid in degrees
    extents: extents of the output grid (lon_0, lon_1, lat_0, lat_1)
    interpolation_method: method of interpolation for the scipy.interpolate griddata function

    memory=True to check if the calculation has already been done and loading the data directly
    dir_memory: location of stored data for the memory check
    dir_raw_memory: location of stored raw data for the memory check

    optional: ui=True to print progress of function
    """

    if ui:
        print(f'RADSgrid2array initiated with startday={startday} and n={n} [t={t()} s]')

    # calling getRADSgridAllSat
    sla_grid, longitude_grid, latitude_grid, time_grid = getRADStimeAllSat(startday, n, grid_resolution, memory, extents, interpolation_method, dir_memory, dir_raw_memory, ui)

    # converting grids into 1D arrays
    sla_array = sla_grid.reshape((sla_grid.size,))
    longitude_array = longitude_grid.reshape((longitude_grid.size,))
    latitude_array = latitude_grid.reshape((latitude_grid.size,))
    time_array = time_grid.reshape((time_grid.size,))

    if ui:
        print(f'RADSgrid2array Done! [t={t()} s]')

    return sla_array, longitude_array, latitude_array, time_array


def plotRADSdataWorldMap(axis:mpl.axes, startday:int, n:int=1, grid_resolution:float=0.1, memory:bool=True, extents:tuple=(-85, 25, 0, 70), interpolation_method:str='linear', dir_memory:str="RADS/getRADSgridAllSat_memory/", dir_raw_memory:str="RADS/getRADSrawAllSat_memory/", filter_on_land:bool=True, smoothen:bool=True, smoothen_sigma:float=1., ui:bool=False):
    """plot data from the four satellites starting at startday and for a span of n days onto a world map

    axis: matplotlib.pyplot axis, output of plt.subplots()

    grid_resolution: resolution of the output grid in degrees
    extents: extents of the output grid (lon_0, lon_1, lat_0, lat_1)
    interpolation_method: method of interpolation for the scipy.interpolate griddata function

    memory=True to check if the calculation has already been done and loading the data directly
    dir_memory: location of stored data for the memory check
    dir_raw_memory: location of stored raw data for the memory check

    filter_on_land: change values on land to NaN
    smoothen: use a scipy.ndimage.gaussian_filter to smoothen the data
    smoothen_sigma: SD input for gaussian_filter

    optional: ui=True to print progress of function
    """
    
    if ui:
        print(f'plotRADSdataWorldMap initiated with startday={startday} and n={n} [t={t()} s]')

    # calling getRADSgridAllSat for inputs of plot
    sla_grid, longitude_grid, latitude_grid = getRADSgridAllSat(startday, n, grid_resolution, memory, extents, interpolation_method, dir_memory, dir_raw_memory, filter_on_land, smoothen, smoothen_sigma, ui)

    # unpacking coordinates
    lon_0, lon_1, lat_0, lat_1 = extents

    # initiating basemap
    m = Basemap(projection='merc', llcrnrlat=lat_0, urcrnrlat=lat_1, llcrnrlon=lon_0, urcrnrlon=lon_1, lat_ts=20, resolution='l', ax=axis)

    # normalization of the colormap, min=-1m max=+1m
    norm = mpl.colors.Normalize(-1, 1)

    # plotting data on axis
    axis.set_title(f'Interpolated Data ({interpolation_method})')
    map = m.pcolormesh(longitude_grid, latitude_grid,  sla_grid, latlon=True, cmap='seismic', norm=norm)
    # map features
    m.drawcoastlines()
    #m.fillcontinents()
    m.drawparallels(np.arange(10,90,20),labels=[1,1,0,1])
    m.drawmeridians(np.arange(-180,180,30),labels=[1,1,0,1])

    if ui:
        print(f'plotRADSdataWorldMap Done! [t={t()} s]')

    return map


def plotRADStimeWorldMap(axis:mpl.axes, startday:int, n:int=1, grid_resolution:float=0.1, memory:bool=True, extents:tuple=(-85, 25, 0, 70), interpolation_method:str='linear', dir_memory:str="RADS/getRADStimeAllSat_memory/", dir_raw_memory:str="RADS/getRADSrawAllSat_memory/", filter_on_land:bool=True, smoothen:bool=True, smoothen_sigma:float=1., ui:bool=False):
    """plot data from the four satellites starting at startday and for a span of n days onto a world map

    axis: matplotlib.pyplot axis, output of plt.subplots()

    grid_resolution: resolution of the output grid in degrees
    extents: extents of the output grid (lon_0, lon_1, lat_0, lat_1)
    interpolation_method: method of interpolation for the scipy.interpolate griddata function

    memory=True to check if the calculation has already been done and loading the data directly
    dir_memory: location of stored data for the memory check
    dir_raw_memory: location of stored raw data for the memory check

    filter_on_land: change values on land to NaN
    smoothen: use a scipy.ndimage.gaussian_filter to smoothen the data
    smoothen_sigma: SD input for gaussian_filter

    optional: ui=True to print progress of function
    """
    
    if ui:
        print(f'plotRADSdataWorldMap initiated with startday={startday} and n={n} [t={t()} s]')

    # calling getRADStimeAllSat for inputs of plot
    sla_grid, longitude_grid, latitude_grid = getRADStimeAllSat(startday, n, grid_resolution, memory, extents, interpolation_method, dir_memory, dir_raw_memory, filter_on_land, smoothen, smoothen_sigma, ui)

    # unpacking coordinates
    lon_0, lon_1, lat_0, lat_1 = extents

    # initiating basemap
    m = Basemap(projection='merc', llcrnrlat=lat_0, urcrnrlat=lat_1, llcrnrlon=lon_0, urcrnrlon=lon_1, lat_ts=20, resolution='l', ax=axis)

    # normalization of the colormap, min=-1m max=+1m
    norm = mpl.colors.Normalize(-1, 1)

    # plotting data on axis
    axis.set_title(f'Time Interpolated Data (startday: {startday}, n: {n})')
    map = m.pcolormesh(longitude_grid, latitude_grid, sla_grid, latlon=True, cmap='seismic', norm=norm)
    # map features
    m.drawcoastlines()
    #m.fillcontinents()
    m.drawparallels(np.arange(10,90,20),labels=[1,1,0,1])
    m.drawmeridians(np.arange(-180,180,30),labels=[1,1,0,1])

    if ui:
        print(f'plotRADSdataWorldMap Done! [t={t()} s]')

    return map


def plotRADSrawWorldMap(axis:mpl.axes, startday:int, n:int=1, memory:bool=True, extents:tuple=(-85, 25, 0, 70), dir_raw_memory:str="RADS/getRADSrawAllSat_memory/", ui:bool=False):
    """plot raw data from the four satellites starting at startday and for a span of n days onto a world map

    axis: matplotlib.pyplot axis, output of plt.subplots()

    grid_resolution: resolution of the output grid in degrees
    extents: extents of the output grid (lon_0, lon_1, lat_0, lat_1)
    interpolation_method: method of interpolation for the scipy.interpolate griddata function

    memory=True to check if the calculation has already been done and loading the data directly
    dir_raw_memory: location of stored data for the memory check

    optional: ui=True to print progress of function
    """
    
    if ui:
        print(f'plotRADSrawWorldMap initiated with startday={startday} and n={n} [t={t()} s]')

    # calling getRADSrawAllSat for inputs of plot
    raw_data_AllSat = getRADSrawAllSat(startday, n, memory, dir_raw_memory, ui)
    latitude_array = raw_data_AllSat[:, 1]
    longitude_array = raw_data_AllSat[:, 2]
    sla_array = raw_data_AllSat[:, 3]

    # unpacking coordinates
    lon_0, lon_1, lat_0, lat_1 = extents

    # initiating basemap
    m = Basemap(projection='merc', llcrnrlat=lat_0, urcrnrlat=lat_1, llcrnrlon=lon_0, urcrnrlon=lon_1, lat_ts=20, resolution='l', ax=axis)

    # normalization of the colormap, min=-1m max=+1m
    norm = mpl.colors.Normalize(-1, 1)

    # conversion of coordinates to basemap
    longitude_map, latitude_map = m(longitude_array, latitude_array)

    # plotting data on axis
    axis.set_title(f'Raw Data (Startday: {startday}, Days: {n})')
    axis.scatter(longitude_map, latitude_map, np.ones((sla_array.size,))/10, sla_array, cmap='seismic', norm=norm)
    # map features
    m.drawcoastlines()
    #m.fillcontinents()
    m.drawparallels(np.arange(10,90,20),labels=[1,1,0,1])
    m.drawmeridians(np.arange(-180,180,30),labels=[1,1,0,1])

    if ui:
        print(f'plotRADSdataWorldMap Done! [t={t()} s]')


if __name__=="__main__":

    ui=True
    
    t_0 = time.time()


    startday = 1
    n = 7

    #getRADStimeAllSat(22, 7, ui=True)

    #fig, ax = plt.subplots()
    #plotRADStimeWorldMap(ax, 1, 7, ui=True)
    #plt.show()

    #'''
    # Error handling
    import logging
    logging.basicConfig(filename='RADS/getRADSgridAllSat_memory/errors.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger=logging.getLogger(__name__)
    
    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    #extents = (-80, -10, 0, 60)

    
    # For calculating whole 2022
    n = 7
    ui=True
    t_0 = time.time()
    for i in range(1, 365, 7):
        print(f'{bcolors.HEADER}Startday = {i}, n = {n} [{t()} s]{bcolors.ENDC}')
        try:
            
            #fig, ax = plt.subplots()
            #plotRADStimeWorldMap(ax, i, n, ui=ui)
            #plt.show()
            #plt.close(fig)
            
            getRADStimeAllSat(i, n, ui=ui)
        except:
            print(f'{bcolors.FAIL}{bcolors.BOLD}FAILED!{bcolors.ENDC} {bcolors.FAIL}Startday = {i}, n = {n}{bcolors.ENDC}')
            print(f'{bcolors.WARNING}Check errors.log for more info.{bcolors.ENDC}')
            logging.exception(f'Failed on Startday = {i}, n = {n}')
    #'''

    '''
    getRADStimeAllSat(15, 7, ui=True)
    a = getRADSrawAllSat(15, 7, filter_threshold=.5, ui=True)
    sla = a[:, 3]
    sla.sort()
    print(sla.size)
    print(sla[-101:-1])
    print(sla[0:100])
    '''

    '''
    raw_data = getRADSrawAllSat(1, 365, ui=True)
    sla = raw_data[:, 3]

    m = sla.mean()
    s = sla.std()

    print(f'Mean = {m}', f'Standard Deviation = {s}')

    neg_3s = m - 3 * s
    pos_3s = m + 3 * s

    print(f'-3s = {neg_3s}', f'+3s = {pos_3s}')

    plt.hist(sla, bins=1000)
    plt.show()
    '''

    '''
    fig, ax = plt.subplots(1, 2)
    plotRADSrawWorldMap(ax[0], 99, 7, ui=True)
    plotRADStimeWorldMap(ax[1], 99, 7, ui=True)
    plt.show()
    '''

    '''

    fig, ax = plt.subplots()

    a, _, _ = getRADStimeAllSat(startday, n, smoothen=False, ui=ui)
    b, _, _ = getRADStimeAllSat(startday, n, smoothen_sigma=10, ui=ui)

    c = a-b

    m = ax.imshow(c, cmap='seismic')
    plt.show()

    '''

    '''
    #getRADStimeAllSat(36, 7, extents=(-85, 0, 0, 70), ui=ui)
    #getRADStimeAllSat(57, 7, extents=(-85, 0, 0, 70), ui=ui)
    
    fig, ax = plt.subplots(1, 3)

    plotRADSrawWorldMap(ax[0], startday, n, ui=ui)
    plotRADSrawWorldMap(ax[1], 36, n, ui=ui)
    plotRADSrawWorldMap(ax[2], 57, n, ui=ui)

    plt.show()
    '''

    '''
    fig, ax = plt.subplots(2, 2, figsize=[10, 10])

    plotRADSrawWorldMap(ax[0, 0], startday, n)
    plotRADSdataWorldMap(ax[0, 1], startday, n)
    plotRADStimeWorldMap(ax[1, 0], startday, n, smoothen=False, ui=ui)
    plotRADStimeWorldMap(ax[1, 1], startday, n, smoothen_sigma=5., ui=ui)

    ax[1, 0].set_title('Time interpolation')

    fig.savefig('RADS/gaussian_filter_display_example.png', dpi=300)
    '''

    '''
    fig, ax = plt.subplots(2, 3, figsize=[15, 10])

    plotRADStimeWorldMap(ax[0, 0], 1, n, smoothen_sigma=1., ui=ui)
    plotRADStimeWorldMap(ax[0, 1], 8, n, smoothen_sigma=1., ui=ui)
    plotRADStimeWorldMap(ax[0, 2], 15, n, smoothen_sigma=1., ui=ui)
    plotRADStimeWorldMap(ax[1, 0], 22, n, smoothen_sigma=1., ui=ui)
    plotRADStimeWorldMap(ax[1, 1], 29, n, smoothen_sigma=1., ui=ui)
    plotRADStimeWorldMap(ax[1, 2], 43, n, smoothen_sigma=1., ui=ui)

    #fig.savefig('RADS/gaussian_filter_sigma_example_1.png', dpi=300)
    plt.show()
    '''

    '''
    

    sla, longitude, latitude = getRADStimeAllSat(startday, n, ui=ui)

    overlay = globe.is_land(latitude, longitude)

    print(overlay)
    '''
    

    '''
    fig, ax = plt.subplots(1, 2)

    plotRADSdataWorldMap(ax[0], startday, n, ui=ui)
    plotRADStimeWorldMap(ax[1], startday, n, ui=ui)

    plt.show()

    #getRADSgridAllSat(startday, n, ui=ui)
    #getRADStimeAllSat(startday, n, ui=ui)
    '''

    '''

    fig, ax = plt.subplots(1, 2, figsize=[25, 10])
    #fig, ax = plt.subplots(1, 3, figsize=[25, 6]) # for zoom

    extents = (-85, 25, 0, 70)
    #extents = (-80, -30, 30, 50) # for zoom

    plotRADStimeWorldMap(ax[0], startday, n, extents=extents, ui=ui)
    #plotRADSrawWorldMap(ax[0], startday, n, extents=extents, ui=ui)
    #plotRADSdataWorldMap(ax[1], startday, n, extents=extents, ui=ui, interpolation_method='nearest')
    map = plotRADSdataWorldMap(ax[1], startday, n, extents=extents, ui=ui)

    if ui:
        print(f'Plotting... [t={t()} s]')

    fig.tight_layout()

    # COLORBAR PLOT
    # You input the POSITION AND DIMENSIONS RELATIVE TO THE AXES
    x0, y0, width, height = [0.88, 0.05, 1.5, 0.02]

    # and transform them after to get the ABSOLUTE POSITION AND DIMENSIONS
    Bbox = mpl.transforms.Bbox.from_bounds(x0, y0, width, height)
    trans = ax[0].transAxes + fig.transFigure.inverted()
    l, b, w, h = mpl.transforms.TransformedBbox(Bbox, trans).bounds

    # Now just create the axes and the colorbar
    cbaxes = fig.add_axes([l, b, w, h])
    cbar = plt.colorbar(map, cax=cbaxes, orientation='horizontal', label='SLA [m]')

    fig.suptitle(f'RADS {n} day(s)', fontsize=32)

    #plt.show()
    fig.savefig(f'RADS/GridResolutionDisplay/TimeGridTest_{startday}_{n}.png', dpi=300)
    
    '''

    '''
    # Error handling
    import logging
    logging.basicConfig(filename='RADS/plotRADSrawWorldMap_memory/errors.log', level=logging.WARNING, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger=logging.getLogger(__name__)
    
    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    
    # For calculating whole 2022
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    n = 7
    ui=True
    t_0 = time.time()
    for i in range(1, 365, 7):
        print(f'{bcolors.HEADER}Startday = {i}, n = {n} [{t()} s]{bcolors.ENDC}')
        try:
            fig, ax = plt.subplots(figsize=[8, 7])
            map = plotRADSrawWorldMap(ax, i, n, ui=ui)
            if ui:
                print(f'Plotting... [t={t()} s]')

            # COLORBAR PLOT
            # normalization of the colormap, min=-1m max=+1m
            norm = mpl.colors.Normalize(-1, 1)
            map = mpl.cm.ScalarMappable(norm=norm, cmap='seismic')
            cbar = plt.colorbar(map, ax=ax, orientation='horizontal', label='SLA [m]', aspect=30, fraction=0.034, pad=0.05)

            fig.tight_layout()

            #plt.show()
            fig.savefig(f'RADS/plotRADSrawWorldMap_memory/plotRADSrawWorldMap_{i}_{n}.png', dpi=300)
            plt.close()
        except:
            print(f'{bcolors.FAIL}{bcolors.BOLD}FAILED!{bcolors.ENDC} {bcolors.FAIL}Startday = {i}, n = {n}{bcolors.ENDC}')
            print(f'{bcolors.WARNING}Check errors.log for more info.{bcolors.ENDC}')
            logging.exception(f'Failed on Startday = {i}, n = {n}')


    '''


    '''

    data = np.load("RADS\getRADSrawAllSat_memory_OLD\getRADSrawAllSat_1_365.npy")
    #print(data)
    data_av = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)

    plt.scatter(data[:, 0], data[:, 1], c=data[:, 3], cmap='seismic')
    #plt.hist(data[:, 3], bins=1000, log=True)
    #plt.boxplot(data[:, 3])
    plt.show()

    print(data_av[3], data_std[3], 2/data_std[3])
    '''

    if ui:
        print(f'Done! [t={t()} s]')
    