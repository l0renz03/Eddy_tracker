# imports
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.basemap import Basemap
from math import sin, sqrt, radians
import pandas as pd
from scipy.spatial.distance import cdist
from RADSgetgrid import RADSgrid
import random
from scipy.spatial.distance import cdist
import GridResolutionDisplay as GRD
import netCDF4 as nc
from paths import *
import os
from global_land_mask import globe


def reduce_size(sla_grid, lon_grid, lat_grid, index_extents: tuple = (0, 250, 0, 250)):
    i1,i2,i3,i4 = index_extents
    sla_grid = sla_grid[i1:i2,i3:i4]
    lon_grid = lon_grid[i3:i4]
    lat_grid = lat_grid[i1:i2]
    return sla_grid, lon_grid, lat_grid


def coriolis_param(latitude: float):
    ''' Compute the coriolis parameter for given latitude (in degrees). '''
    f = 2 * 7.292 * 10 ** (-5) * sin(radians(latitude))
    return f

def velocity_field(data_grid: np.array, lat_grid, lon_grid,  plot = False, print_ = False):
    '''

    Will take data_grid as input (np.array) and compute u, v. u, v outputed as pd.dataframes index by lattitude and longitude.
    u = -dh/dy * g/corelis_param(lat)
    u = dh/dx * g/corelis_param(lat)

    '''
    #Data_grid : np.array
    #Frame : pd.dataframe


    #Take gradient Specifically, for a 2D array, the returned tuple contains two arrays:
    # the first one represents the gradient along the rows (i.e., the y-axis)
    # and the second one represents the gradient along the columns (i.e., the x-axis)
    u,v = np.gradient(data_grid)

    #Correct sign of u
    u = -u # THIS WAS THE PROBLEM WITH THE GEOSTROPHIC CURRENT
    v = -v

    f = coriolis_param
    g = 9.80665 # Gravity acceleration [m/s2]

    #Make the velocities dataframes:
    u_frame = pd.DataFrame(u, index = lat_grid, columns = lon_grid)
    v_frame = pd.DataFrame(v, index = lat_grid, columns = lon_grid)

    row_idx = u_frame.index.values
    column_idx = u_frame.columns.values

    # Makes adjusts u and v by the coriolis parameter

    for i in row_idx: #Latitude
        for j in column_idx: #Longitude
            if print_:
                print(f"Point at lat. = {i:.2f} and lon. = {j:.2f} is {v_frame.loc[i, j]:.5f}")
            if not np.isnan(v_frame.loc[i,j]):
                v_frame.loc[i,j] = v_frame.loc[i,j] * (g/f(i))

    for i in row_idx: # Latitude
        for j in column_idx:  # Longitude
            if not np.isnan(u_frame.loc[i,j]):  
                u_frame.loc[i, j] = u_frame.loc[i,j] * (g / f(i))


    if plot:

        plot_sla(data_grid, "SLA over U-V", "Longitude", "Latitude", lat_grid, lon_grid,quiver=True, u_frame=u_frame, v_frame= v_frame )

    return u_frame, v_frame

def okuboweiss(u_frame,v_frame, lat_grid, lon_grid, plot = False, print_ = False, ow_cut = 1.1, week = 0):
    #Convert to numpy arrays:
    u = u_frame.values
    v = v_frame.values
    #Specifically, for a 2D array, the returned tuple contains two arrays:
    # the first one represents the gradient along the rows (i.e., the y-axis)
    # and the second one represents the gradient along the columns (i.e., the x-axis)
    dudy, dudx = np.gradient(u)
    dudx = -dudx
    dvdy, dvdx = np.gradient(v)
    dvdx = -dvdx

    sn = dudx - dvdy
    ss = dvdx + dudy
    omega = dvdx - dudy

    okubo_weiss = ss ** 2 + sn ** 2 - omega ** 2
    # print(okubo_weiss)
    # print(okubo_weiss.shape)

    if plot:

        plot_sla(okubo_weiss, "Okubo-Weiss Parameter", "Longitude", "Latitude", lat_grid, lon_grid, colorbar_label='OW-Parameter', week = week)

    #Compute the variance of the okubo-weiss to set a threshhold:
    okubo_weiss_1D = okubo_weiss.reshape(-1)
    var = np.nanvar(okubo_weiss_1D)
    # print(f'Variance = {var}')
    iseddy = np.zeros_like(okubo_weiss)
    ow_cutoff = -1 * ow_cut
    iseddy[okubo_weiss < ow_cutoff * sqrt(var)] = 1 #WE NEED TO USE STANDARD DEVIATION NOT THE FULL VAR
    # print(np.all(iseddy == 0))
    # print(iseddy)

    if plot:

        nan_mask = np.isnan(okubo_weiss)
        iseddy[nan_mask] = np.nan
        name = f"Eddy point location (OW < {ow_cutoff}*stdv)"
        plot_sla(iseddy, name , "Longitude", "Latitude", lat_grid, lon_grid,cmap='bwr',
                 colorbar_label='Blue --> No | Red --> Yes', OW=ow_cutoff, week = week)

    return iseddy

def check_map(p_lat_start, p_lat_stop, p_lon_start, p_lon_stop):
    """Print the map of the world between given pixel coordinates (-85<--0 for lon and 70<--0 for lat for example). This is to help visualozation"""
    lon_start = (p_lon_start-850)/10
    lon_stop = (p_lon_stop - 850) / 10
    lat_start = (700-p_lat_start)/10
    lat_stop = (700-p_lat_stop)/10
    print(lon_start, lon_stop, lat_start, lat_stop)
    # Create a new map
    m = Basemap(projection='cyl', llcrnrlon=lon_start, llcrnrlat=lat_stop, urcrnrlon=lon_stop, urcrnrlat=lat_start,
                resolution='c')

    # Draw coastlines, countries, and parallels/meridians
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(range(-90, 91, 30), labels=[True, False, False, False], linewidth=0.5, color='gray')
    m.drawmeridians(range(-180, 181, 60), labels=[False, False, False, True], linewidth=0.5, color='gray')

    # Show the map
    # plt.show()
    plt.savefig(os.path.join(images_path, 'test map.png'))

def plot_sla(data_grid: np.array, Title: str, x_axis: str, y_axis: str, lat, lon , cmap = 'RdBu',
             u_frame = pd.DataFrame, v_frame = pd.DataFrame, quiver = False, colorbar_label = 'Sea Level Anomaly',
             OW = 0, week = 0, landmask = False, additional_mask = None):

    '''Plotting function for numpy arrays. Quiver if you plot u-v velocities.'''
    if landmask:
        longitude_grid, latitude_grid = np.meshgrid(lon, lat, indexing='xy')
        # print(len(longitude_grid), np.shape(longitude_grid), np.shape(latitude_grid))
        overlay = globe.is_land(latitude_grid, longitude_grid)
        data_grid[overlay==True] = np.nan

    lat = np.round(lat, 1)
    lon = np.round(lon, 1)
    fig, ax = plt.subplots(1, 1)
    cmap = plt.cm.get_cmap(cmap)
    cmap.set_bad('grey')
    if quiver :
        ax.quiver(u_frame.values, v_frame.values)

    im = ax.imshow(data_grid, cmap= cmap)

    if type(additional_mask) != type(None):
        cmap_dict = {1: '#000000', 0: '#ff000000'}
        cmap2 = ListedColormap([cmap_dict[i] for i in range(2)])
        ax.imshow(additional_mask, cmap2)

    # ax.title.set_text(Title)

    y_size, x_size = np.shape(data_grid)

    y_step = int(y_size / 10)
    x_step = int(x_size / 10)

    ax.set_xticks(np.arange(0, x_size, x_step))
    ax.set_yticks(np.arange(0, y_size, y_step))
    ax.set_xticklabels(lon[::x_step], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(lat[::y_step])
    ax.set_xlabel(x_axis, fontsize=12)
    ax.set_ylabel(y_axis, fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label(colorbar_label)

    if OW != 0:
        Title = f"Eddy point location OW {OW}"

    grid_size_km = 0.1 * data_grid.shape[0] / 10  # assuming 1 degree is 111.1 km

    # add a label for the scale in kilometers
    #plt.xlabel(f'Scale: {grid_size_km} km')

    km = round(0.1 * data_grid.shape[1] * 0.1 * 111.1, -2)

    if data_grid.shape[1] * 0.1 * 111.1 <= 400:
        km = 50

    if km != 0:
        # add a black line to represent the 50 km scale
        line_y = data_grid.shape[0] * 0.95
        line_x_start = data_grid.shape[1] * 0.90 - km / (0.1 * 111.1)
        line_x_end = data_grid.shape[1] * 0.90
        x_text = line_x_start + (line_x_end - line_x_start) / 2
        ax.text(x_text, line_y - data_grid.shape[1] * 0.05, f'{km} [km]', ha='center', va='center')
        plt.plot([line_x_start, line_x_end], [line_y, line_y], 'k-', linewidth=1)
        plt.plot([line_x_start, line_x_start], [line_y - 1, line_y + 1], 'k-', linewidth=1)
        plt.plot([line_x_end, line_x_end], [line_y - 1, line_y + 1], 'k-', linewidth=1)

    plt.subplots_adjust(bottom=0.20)
    # plt.savefig(os.path.join("RADS/Overleaf_images/", (f'{Title}_Small).png')
    # plt.savefig(os.path.join("RADS/Overleaf_images/", (f'{Title}_Small).svg')
    # Specify the folder path
    folder_path = folder_creation + f'Overleaf_images/week{week}'

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(folder_path, f'{Title}.png'), bbox_inches='tight', pad_inches = 0)
    plt.savefig(os.path.join(folder_path, f'{Title}.svg'), bbox_inches='tight', pad_inches = 0)

    # plt.show()

def grouping(matrix: np.array, Print = False):
    # Initialize an empty list to store the groups of ones
    one_groups = []

    # Define a recursive function to perform DFS on the graph
    def dfs(node, group):
        # Add the current node to the group
        group.append(node)
        # Mark the node as visited by setting its value to 0
        matrix[node[0]][node[1]] = 0
        # Check the neighboring nodes and recursively call dfs on unvisited one nodes
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                x, y = node[0] + i, node[1] + j
                if x < 0 or y < 0 or x >= len(matrix) or y >= len(matrix[0]):
                    continue
                if matrix[x][y] == 1:
                    dfs((x, y), group)

    # Loop over the matrix and find the connected components of ones using DFS
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                group = []
                dfs((i, j), group)
                one_groups.append(group)

    print_one_group = []
    unpacked_one_group = []
    fig, ax = plt.subplots(1,1)
    for group in one_groups:
        print_group = []
        unpacked_group = []
        for point in group:
            c, r = point
            c = np.size(matrix, axis = 0) - 1 - c
            print_point = [r,c]
            print_group.append(print_point)
            # Create a scatter plot of the points

            #Unpack tuples
            r , c = point
            unpacked_group.append([r,c])
        if Print:
            plt.scatter([point[0] for point in print_group], [point[1] for point in print_group])
        print_one_group.append(print_group)
        unpacked_one_group.append(unpacked_group)

    plt.savefig(os.path.join(images_path, "ENCLOSED.PNG"))
    if Print:
        plt.show()

    # Print the groups of ones
    # print("Groups of ones:")
    # for group in unpacked_group:
    #     print(group) 
    return unpacked_one_group

class Eddy():
    def __init__(self, group, sla_grid, lat_grid, lon_grid, u, v, sst_grid = None):

        '''
        An eddy should be a list of coordinates of eddy points, for example [ [0, 1], [0, 2], [0, 3] ] can be an eddy consiting of 3 points.
        One should perform the methods of this class on such objects.

        group :     a list of coordinates of eddy points (like in the example)
        sla_grid :  list of sla values
        lat_grid :
        lon_grid:
        '''
        
        self.group = group
        self.sla_grid = sla_grid
        self.area = len(group)
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.no_points = len(group)
        self.area = self.no_points * (11.132 * 11.113 * 1000 * 1000)  # in m^2
        self.u_grid = u
        self.v_grid = v
        self.sst_grid = sst_grid

    def centre(self):
        r_sum = 0
        c_sum = 0
        # print(f"Group = {self.group}")
        # print(f"No. of points = {self.no_points}, {len(self.group)}")
        # for coord in self.group:
        #     r_sum += coord[0]
        #     c_sum += coord[1]
        for r, c in self.group:
            r_sum += r
            c_sum += c
        centre_r = r_sum/self.no_points
        centre_c = c_sum/self.no_points
        return centre_r, centre_c
    
    def area_units(self):
        return self.area*(11.132*11.113*1000*1000) # in m^2

    def get_grid_average(self):
        cell_size = 1
        points = self.group
        grid = self.sla_grid

        # Step 1: Convert the list of coordinates to grid cells
        cells = [(int(point[0] / cell_size), int(point[1] / cell_size)) for point in points]

        # Step 2: Calculate the sum of the values in the cells
        sum_values = np.sum(grid[[cell[0] for cell in cells], [cell[1] for cell in cells]])

        # Step 3: Calculate the average value
        num_cells = len(cells)
        average_value = sum_values / num_cells

        return average_value
    
    def get_grid_average_SST(self):
        cell_size = 1
        points = self.group
        grid = self.sst_grid

        # Step 1: Convert the list of coordinates to grid cells
        cells = [(int(point[0] / cell_size), int(point[1] / cell_size)) for point in points]

        # Step 2: Calculate the sum of the values in the cells
        sum_values = np.sum(grid[[cell[0] for cell in cells], [cell[1] for cell in cells]])

        # Step 3: Calculate the average value
        num_cells = len(cells)
        average_value = sum_values / num_cells

        return average_value
    
    def eddy_sla(self):
        if self.get_grid_average() < 0:
            eddy_type_sla = "n"
        else:
            eddy_type_sla = "p"
        return eddy_type_sla
    
    def eddy_circ(self):
        dudy, dudx = np.gradient(self.u_grid)
        dvdy, dvdx = np.gradient(self.v_grid)
        curl = dvdx[int(self.centre()[0]), int(self.centre()[1])] - dudy[int(self.centre()[0]), int(self.centre()[1])]
        if curl > 0:
            eddy_type_circ = "ac"    # cyclonic
        else:
            eddy_type_circ = "c"   # anticyclonic
        return eddy_type_circ
    
    def eddy_ssta(self):
        r, c = self.centre()
        sla_ones = np.ones(self.sla_grid.shape) #Takes a grid of ones the same shape as sla_grid so that NaNs dont annoy us.
        # Create a mask to select the points within the maximum radius
        #mask = cdist([(r, c)], np.argwhere(self.sla_grid != 0)) <= self.largest_radius()
        distance = cdist([(r, c)], np.argwhere(sla_ones))
        mask = distance  <= self.largest_radius()*1.5
        mask = np.reshape(mask, self.sla_grid.shape)

        # if self.get_grid_average() < 0:
        #     # Select the relevant points that are smaller than the average SLA
        #     relevant_mask = np.logical_and(mask, self.sla_grid < self.get_grid_average_SST())
        # else:
        #     # Select the relevant points that are larger than the average SLA
        #     relevant_mask = np.logical_and(mask, self.sla_grid > self.get_grid_average_SST())

        # Apply the mask to the grid values to select the relevant points
        np.where(mask==True, 1, 0)
        group = np.argwhere(mask)
        fake_eddy = Eddy(group, self.sla_grid, self.lat_grid, self.lon_grid, self.u_grid, self.v_grid, self.sst_grid)
        modified_SST = fake_eddy.get_grid_average_SST()
        if modified_SST < self.get_grid_average_SST():
            eddy_type_ssta = "wc" # extended eddy cooler, so core must be warmer
        else:
            eddy_type_ssta = "cc" # vice versa

        return eddy_type_ssta

    def largest_radius(self):
        x, y = self.centre()
        XA = np.array([[x, y], [x, y]])
        XB = np.array([[point[0], point[1]] for point in self.group])
        # print("XA", XA)
        # print("XB", XB)
        distances = cdist(XA, XB)
        largest_radius = np.amax(distances)

        return largest_radius

    def search_within_radius(self, update: bool):
        '''Searchest within largest radius of eddy and updates points based on SLA and returns the new list and update the list in the class if update = True.'''
        # Extract the (x, y) coordinates of the given point indices
        r, c = self.centre()
        sla_ones = np.ones(self.sla_grid.shape) #Takes a grid of ones the same shape as sla_grid so that NaNs dont annoy us.
        # Create a mask to select the points within the maximum radius
        #mask = cdist([(r, c)], np.argwhere(self.sla_grid != 0)) <= self.largest_radius()
        distance = cdist([(r, c)], np.argwhere(sla_ones))
        mask = distance  <= self.largest_radius()
        mask = np.reshape(mask, self.sla_grid.shape)

        if self.get_grid_average() < 0:
            # Select the relevant points that are smaller than the average SLA
            relevant_mask = np.logical_and(mask, self.sla_grid < self.get_grid_average())
        else:
            # Select the relevant points that are larger than the average SLA
            relevant_mask = np.logical_and(mask, self.sla_grid > self.get_grid_average())

        # Apply the mask to the grid values to select the relevant points
        relevant_values = self.sla_grid[relevant_mask]

        # Return the indices of the relevant points that are smaller than or larger than the average SLA
        corrected_points = np.argwhere(relevant_mask)
        # print(type(corrected_points))
        # print(corrected_points)
        # print(self.group)
        # print(type(self.group))
        updated_group = sorted(corrected_points.tolist() + self.group)

        if update:
            self.group = updated_group
            self.no_points = len(self.group)


        return updated_group

    def print_eddy(self, plot: bool, title):
        '''Print the eddy on a map. This will be the corrected eddy if search_within_radius(update = True) has been called.'''
        mask_size = np.shape(self.sla_grid)
        mask = np.zeros(mask_size)
        for r, c in self.group:
            mask[r, c] = 1
        # plot centre
        #print(self.centre())
        r, c = self.centre()
        r = int(r)
        c = int(c)
        # mask[r, c] = 0.5
        if plot:
            plot_sla(mask, title, 'Longitude', 'Lattitude', self.lat_grid, self.lon_grid,
                 colorbar_label='Blue --> Yes | Red --> No')
        pass

    def print_corrected_eddy(self, plot: bool, title):
        mask_shape = np.shape(self.sla_grid)
        mask = np.zeros(mask_shape)
        for r, c in self.search_within_radius(update=False):
            mask[r,c] = 1
        if plot:
            plot_sla(mask, title, 'Longitude', 'Lattitude', self.lat_grid, self.lon_grid, colorbar_label= 'Blue --> Yes | Red --> No')
        pass

    def size_filter(self, threshold):
        small = False
        radius = self.largest_radius()*(11.132+11.113)/2
        if radius < threshold:
            small = True
        return small

    def depth_filter(self, depth_array_filter):
        shallow = False
        for i in range(len(self.group)):
            point_lat, point_lon = self.group[i]
            filter_value = depth_array_filter[point_lat, point_lon]
            if filter_value == 0:
                shallow = True
        return shallow

    def eddy_output_list(self , coord=True):
        # ex_list = [[(2, 2), 0.50, 5], [(42, 53), 10.0, 15]], coordinates, area, amplitude
        centre = self.centre()
        if coord:
            c1, c2 = centre
            extents = (c1, c1, c2, c2)
            coord_ext = ext_to_coord(extents)[2]
            c1, c1, c2, c2 = coord_ext

            # lat_centre, lon_centre = self.centre()
            # lat_ratio = lat_centre/len(self.sla_grid)
            # lon_ratio = lon_centre/len(self.sla_grid[1])
            # lat_centre = self.lat_grid[0] + lat_ratio*(self.lat_grid[len(self.lat_grid)-1] - self.lat_grid[0])
            # lon_centre = self.lon_grid[0] + lon_ratio*(self.lon_grid[len(self.lon_grid)-1] - self.lon_grid[0])
            # centre = (c1, c2)
        output = [centre, self.area, self.get_grid_average()]
        return output
    
#### END OF CLASS DEF ####

def plot_characteristics(plot_type, weekly_list: list, week: int, ow: float, labels = False):
    '''
    Plot = 1: plots amplitude of eddies
    Plot = 2 plots area of eddies
    Plot = 3 plots a scatter plot of area over amplitude'''
    folder = folder_creation + f'Overleaf_images/week{week}'
    # if plot_type == 1:
    #     amplitude_list = sorted([round(sublist[2], 2) for sublist in weekly_list])
    #     fig, ax = plt.subplots()
    #     ax.grid(zorder=1)
    #     colors = ['navy' if val >= 0 else 'red' for val in amplitude_list]
    #     p = ax.bar(range(len(amplitude_list)), amplitude_list, color=colors, zorder=2)
    #
    #     if labels:
    #         ax.bar_label(p, label_type='edge')
    #     ax.set_ylabel('Eddy Amplitude [m]')
    #     ax.set_xlabel('Individual Eddies [-]')
    #     #ax.title.set_text(f'Eddy Amplitudes Over Week {week} OW -{ow}')
    #     ax.set_xticks([])
    #     title = f'Eddy_Amplitude Week {week} OW -{ow}'
    #     plt.savefig(os.path.join(folder, f'{title}.svg'), bbox_inches='tight', pad_inches = 0)
    #     plt.savefig(os.path.join(folder, f'{title}.png'), bbox_inches='tight', pad_inches = 0)
    #     plt.show()
    #     return amplitude_list

    if plot_type == 1:
        amplitude_list = sorted([round(sublist[2], 2) for sublist in weekly_list])
        fig, ax = plt.subplots()
        ax.grid(zorder=1)
        colors = ['navy' if val >= 0 else 'red' for val in amplitude_list]
        p = ax.bar(range(len(amplitude_list)), amplitude_list, color=colors, zorder=2)

        if labels:
            ax.bar_label(p, label_type='edge')
        ax.set_ylabel('Eddy Amplitude [m]')
        ax.set_xlabel('Individual Eddies [-]')
        x_ticks = range(len(amplitude_list))  # Set x-axis ticks for every second number
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([])
        ax.xaxis.set_tick_params(rotation=90)

        title = f'Eddy_Amplitude Week {week} OW -{ow}'
        plt.savefig(os.path.join(folder, f'{title}.svg'), bbox_inches='tight', pad_inches=0)
        plt.savefig(os.path.join(folder, f'{title}.png'), bbox_inches='tight', pad_inches=0)
        # plt.show()
        return amplitude_list

    if plot_type == 2:
        area_list = sorted([sublist[1] for sublist in weekly_list])
        area_list = np.array(area_list)
        area_list = area_list * 10 ** (-6) * 10 ** (-3)
        area_list = np.round(area_list, 2)
        fig, ax = plt.subplots()
        ax.grid(zorder=1)

        # Get the indices of sublists where the amplitude is negative
        negative_indices = [index for index, sublist in enumerate(weekly_list) if sublist[2] < 0]

        # Create a list of colors for the bars
        colors = ['navy' if index not in negative_indices else 'red' for index in range(len(area_list))]

        p = ax.bar(range(len(area_list)), area_list, color=colors, zorder=2)

        if labels:
            ax.bar_label(p, label_type='edge')

        ax.set_ylabel('Eddy Area [10^3 * km^2]')
        ax.set_xlabel('Individual Eddies [-]')
        #ax.title.set_text(f'Eddy Area Over Week {week} OW -{ow}')
        x_ticks = range(len(area_list))  # Set x-axis ticks for every second number
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([])
        ax.xaxis.set_tick_params(rotation=90)
        title = f'Eddy_Area Week {week} OW -{ow}'
        plt.savefig(os.path.join(folder, f'{title}.svg'), bbox_inches='tight', pad_inches = 0)
        plt.savefig(os.path.join(folder, f'{title}.png'), bbox_inches='tight', pad_inches = 0)
        # plt.show()
        return area_list

    if plot_type == 3:
        amplitude_list = sorted([sublist[2] for sublist in weekly_list])
        area_list = sorted([sublist[1] for sublist in weekly_list])
        area_list = np.array(area_list)
        area_list = area_list * 10 ** (-6) * 10 ** (-3)

        fig, ax = plt.subplots()

        coefficients = np.polyfit(area_list, amplitude_list, 1)
        trendline_y = np.polyval(coefficients, area_list)
        print(f'Linear trendline coeffs: {coefficients}')
        plt.plot(area_list, trendline_y, color='navy', label = 'Linear Trendline')

        residuals = amplitude_list - trendline_y
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((amplitude_list - np.mean(amplitude_list)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f'R^2 = {r_squared}')

        log_x = np.log(area_list)
        coefficients2 = np.polyfit(log_x, amplitude_list, 1)
        trendline_log = coefficients2[0] * np.log(area_list) + coefficients2[1]
        print(f'Log trendline coeffs: {coefficients2}')
        plt.plot(area_list, trendline_log, color = 'red', label = 'Logarithmic Trendline')

        residuals = amplitude_list - trendline_log
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((amplitude_list - np.mean(amplitude_list)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f'R^2 = {r_squared}')

        ax.scatter(area_list, amplitude_list, marker = '^', c = amplitude_list)
        ax.set_ylabel('Eddy Amplitude [m]')
        ax.set_xlabel('Eddy Area [10^3 * km^2]')
        #ax.title.set_text(f'Eddy Amplitude over Area Over Week {week} OW -{ow}')
        ax.legend()

        title = f'Eddy_Amplitude_over_Area Week {week} OW -{ow}'
        plt.savefig(os.path.join(folder, f'{title}.svg'), bbox_inches='tight', pad_inches = 0)
        plt.savefig(os.path.join(folder, f'{title}.png'), bbox_inches='tight', pad_inches=0)

        # plt.show()
        pass

def data_grid_selection(extents: tuple, start, step , verbose = False, plot = False, week = 0):
    '''----------'''
    sla_grid, longitude_grid, latitude_grid = GRD.getRADStimeAllSat(start, step, smoothen_sigma=2., dir_memory= dir_memory)
    sla_grid = sla_grid.T
    sla_grid = np.flipud(sla_grid)

    if verbose:
        print(type(sla_grid))
        print(sla_grid.shape)  # (700, 1100) means that we have latitude as rows and longitude as columns.

    # Transform lon/lat grids into 1d arrays used for dataframe indexing.
    lat_grid = np.flip(latitude_grid[0, :])  # This is x_grid: [-85-->20]
    lat_grid = np.round(lat_grid, 2)

    lon_grid = longitude_grid[:, 0]  # This is y_grid [70-->0]
    lon_grid = np.round(lon_grid, 2)

    if verbose:
        print(f"-----------------  Lattitude (N/S) -----------------")
        print(lat_grid)
        print(f"-----------------  Longitude (W/E) -----------------")
        print(lon_grid)

    if plot:
        print("Plot of the full grid.")
        plot_sla(sla_grid, "Complete SLA map", "Longitude", "Latitude", lat_grid, lon_grid, week=week)

    # Reduce the size of the numpy array while testing:
    # Test = long -60,-65 lat: 45,40
    # extents = (250,300,300,350) #SET SIZE
    #extents = (300, 350, 350, 400)  # Indexed lat then lon.
    sla_grid, lon_grid, lat_grid = reduce_size(sla_grid, lon_grid, lat_grid, index_extents=extents)

    if plot:
        #Print checking map
        p_lat_start, p_lat_stop, p_lon_start, p_lon_stop = extents
        check_map(p_lat_start, p_lat_stop, p_lon_start, p_lon_stop)

        #Plot SLA:
        plot_sla(sla_grid, "Partial SLA map", "Longitude", "Latitude", lat_grid, lon_grid, week = week)

    return sla_grid, lon_grid, lat_grid

def sla_pipeline(sla_grid, lon_grid, lat_grid, ow_cut = 1.1, uv_print = False, uv_plot = False, ow_print = False, ow_plot = False, plot_grouping = False, week = 0):
    '''Pipeline : sla_grid --> u,v --> Okubo-Weiss --> Grouping --> one_group'''

    # u and v velocities from SLA:
    u_frame, v_frame = velocity_field(sla_grid, lat_grid, lon_grid, print_= uv_print, plot= uv_plot)

    #Compute dataframes of
    iseddy = okuboweiss(u_frame, v_frame, lat_grid, lon_grid, plot = ow_plot , print_ = ow_print, ow_cut = ow_cut, week = week)

    one_group = grouping(iseddy, Print=plot_grouping)

    u = u_frame.to_numpy()
    v = v_frame.to_numpy()

    return one_group, u, v

def eddy_pipeline(eddies_group, sla_grid, extents, depths_array_filter,  T: bool, u: np.array, v: np.array, plot: bool, week=0, sst_grid = None):
    centres = []
    averages = []
    largest_radii = []
    areas = []
    output_list = []
    filtered_list = []
    small_count = 0
    shallow_count = 0
    eddy_mask_size = np.shape(sla_grid)
    eddy_mask = np.zeros(eddy_mask_size)
    lon_grid, lat_grid, ext = ext_to_coord(extents)

    sst_list_circ_c = []
    sst_list_circ_ac = []
    sst_dict_circ = dict(c = sst_list_circ_c, ac = sst_list_circ_ac)

    sst_list_sla_p = []
    sst_list_sla_n = []
    sst_dict_sla = dict(p = sst_list_sla_p, n = sst_list_sla_n)

    sst_list_ssta_cc = []
    sst_list_ssta_wc = []
    sst_dict_ssta = dict(cc = sst_list_ssta_cc, wc = sst_list_ssta_wc)

    no_of_eddies = len(eddies_group)

    # loop through all the eddies present in one synpotic view, raw detection
    for i, eddy in enumerate(eddies_group):

        # an eddy is made into an object
        eddy = Eddy(eddy, sla_grid, lat_grid, lon_grid, u, v, sst_grid=sst_grid)

        # centre located
        centres.append(eddy.centre())

        # average SLA determined
        averages.append(eddy.get_grid_average())

        # largest radius used for correction
        largest_radii.append(eddy.largest_radius())

        # Correct the size of the eddy - refined and updated
        eddy.search_within_radius(update = True)

        # update and store parameters too with refined eddy
        centres[i] = eddy.centre()
        averages[i] = eddy.get_grid_average()
        largest_radii[i] = eddy.largest_radius()
        areas.append(eddy.area)

        # eddy classification
        eddy_type_sla = eddy.eddy_sla()
        eddy_type_circ = eddy.eddy_circ()
        eddy_type_ssta = eddy.eddy_ssta()

        # Filters

        small = eddy.size_filter(threshold=35) # km

        shallow = eddy.depth_filter(depths_array_filter)

        if shallow == True:
            shallow_count += 1
        if small == True:
            small_count += 1

        if (small == False) and (shallow == False):

            for r, c in eddy.group:
                if eddy_type_circ == "c":
                    eddy_mask[r, c] = 0.5
                else:
                    eddy_mask[r, c] = 1

            output_list.append(eddy.eddy_output_list())

            if T:
                if eddy_type_circ == "c":
                    sst_list_circ_c.append(eddy.get_grid_average_SST())
                else:
                    sst_list_circ_ac.append(eddy.get_grid_average_SST())

                if eddy_type_sla == "p":
                    sst_list_sla_p.append(eddy.get_grid_average_SST())
                else:
                    sst_list_sla_n.append(eddy.get_grid_average_SST())

                if eddy_type_ssta == "wc":
                    sst_list_ssta_wc.append(eddy.get_grid_average_SST())
                else:
                    sst_list_ssta_cc.append(eddy.get_grid_average_SST())

        else:
            filtered_list.append(eddy.eddy_output_list())

    print("Too small: ", small_count, "Too shallow: ", shallow_count)
    plot_sla(eddy_mask, "Cumulative_filtered_eddies_"+str(week), "Longitude", "Latitude", lat_grid, lon_grid, week=week, cmap="bwr", landmask=True, colorbar_label='Blue --> No | Red --> Yes')
    
    # debug - showing a zoomed-in plot
    smallgrid = sla_grid[130:190, 320:380]
    smalleddymask = eddy_mask[130:190, 320:380]
    smallgrid_u = u[130:190, 320:380]*2
    smallgrid_v = v[130:190, 320:380]*2
    smallsst = sst_grid[130:190, 320:380]

    smalllat = lat_grid[130:190]
    smalllon = lon_grid[320:380]

    smallgrid_u = pd.DataFrame(smallgrid_u, index = smalllat, columns = smalllon)
    smallgrid_v = pd.DataFrame(smallgrid_v, index = smalllat, columns = smalllon)
    # c-ac debug
    # plot_sla(smallgrid, "SLA over U-V"+str(week), "Longitude", "Latitude", smalllat, smalllon, week=week, quiver=True, u_frame=smallgrid_u, v_frame= smallgrid_v )
    # plot_sla(smalleddymask, "Cumulative_filtered_eddies_small_"+str(week), "Longitude", "Latitude", smalllat, smalllon, week=week, cmap="bwr", landmask=True, colorbar_label='Blue --> Not eddy | Red --> Anticyclonic | White --> Cyclonic')
    # cc-wc debug
    # plot_sla(smallsst, "Temperature_eddy_correlation_small"+str(week), "Longitude", "Latitude", smalllat, smalllon, cmap="RdBu_r", additional_mask=None, week=week, colorbar_label="SST [K]")

    return output_list, eddy_mask, sst_dict_circ, sst_dict_sla, sst_dict_ssta, no_of_eddies

def ext_to_coord(extents: tuple):
    lat_TL, lat_BR, lon_TL, lon_BR = extents # max is (0, 700, 0, 1100)

    # coord values
    lat_TL_coord = 70 - (lat_TL/700)*70
    lat_BR_coord = 70 - (lat_BR/700)*70
    lon_TL_coord = -85 + (lon_TL/1100)*110
    lon_BR_coord = -85 + (lon_BR/1100)*110
    coord_ext = (lat_TL_coord, lat_BR_coord, lon_TL_coord, lon_BR_coord)

    # 1D coord grids along lon and lat axes
    lon_grid = np.linspace(lon_TL_coord, lon_BR_coord, int(abs(lon_TL-lon_BR)) )
    lat_grid = np.linspace(lat_TL_coord, lat_BR_coord, int(abs(lat_TL-lat_BR)) )
    # print("x - lon: ", len(lon_grid))
    # print("y - lat: ", len(lat_grid))

    return lon_grid, lat_grid, coord_ext

def depth_grid(depth_path, extents):
    fn = depth_path
    ds = nc.Dataset(fn) # positve is surface, negative is depth below sea
    
    # lat from 0 to 70
    # lon from -85 to 25
    lat_TL, lat_BR, lon_TL, lon_BR = extents # extents = (100, 420, 50, 750)
    # print(ds['elevation'][:]) # top left of this print is at bottom left corner of the actual grid, bottom left of this is a value in like canada
    depths_array = np.zeros( ( (lat_BR-lat_TL), (lon_BR-lon_TL) ) )
    lat_arr = np.arange(lat_TL, lat_BR, 1)
    lon_arr = np.arange(lon_TL, lon_BR, 1)

    for i_arr, i in enumerate(lat_arr):
        i_index = 24*i
        lat_index = len(ds["lat"]) - i_index - 1
        for j_arr, j in enumerate(lon_arr):
            lon_index = 24*j
            depths_array[i_arr, j_arr] = ds['elevation'][lat_index, lon_index]

    plot_sla(depths_array, "Sea depths", "Longitude", "Latitude", lat_arr, lon_arr)

    depths_array_filter = depths_array
    depths_array_filter[depths_array_filter > -1000] = 0

    lon_grid = ext_to_coord(extents)[0]
    lat_grid = ext_to_coord(extents)[1]

    plot_sla(depths_array_filter, "Filter map", "Longitude", "Latitude", lat_grid, lon_grid)

    return depths_array, depths_array_filter


def SST_grid(path: str, extents: tuple, weeks: list):
    print("\n", "Loading temperature data...")
    array_list = []
    lat_TL, lat_BR, lon_TL, lon_BR = extents
    weeks_str = [str(x) for x in weeks]
    for index in range(len(weeks_str)):
        if len(weeks_str[index]) == 1:
            weeks_str[index] = "0"+weeks_str[index]

    for i, filename in enumerate(os.listdir(path)):
        if str(filename[-6:-4]) in weeks_str:
            print(filename)
            f = os.path.join(path, filename)
            SST_grid = np.load(f)
            SST_grid = np.flip(SST_grid, 0)

            SST_array = np.zeros( ( 700, 1100 ) )

            for k in range(700):
                k_index = 10*k
                for l in range(1100):
                    l_index = 10*l
                    SST_array[k, l] = SST_grid[k_index, l_index]

            SST_array = SST_array[lat_TL:lat_BR, lon_TL:lon_BR]
            # correction for faulty data
            SST_array[SST_array < 220] = np.nan
            SST_array[SST_array > 350] = np.nan
            array_list.append(SST_array)
        else:
            array_list.append(0)
    
    print("Temperature data loaded")

    return array_list

def load_avg_sst(path: str):
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        avg_array = np.load(f)
        if filename[9]=="1":
            if filename[10:12]=="cy":
                c1 = avg_array
            elif filename[10:12]=="ac":
                ac1 = avg_array
            elif filename[10:12]=="po":
                p1 = avg_array
            elif filename[10:12]=="ne":
                n1 = avg_array
            elif filename[10:12]=="wc":
                wc1 = avg_array
            elif filename[10:12]=="cc":
                cc1 = avg_array
        else:
            if filename[10:12]=="cy":
                c2 = avg_array
            elif filename[10:12]=="ac":
                ac2 = avg_array
            elif filename[10:12]=="po":
                p2 = avg_array
            elif filename[10:12]=="ne":
                n2 = avg_array
            elif filename[10:12]=="wc":
                wc2 = avg_array
            elif filename[10:12]=="cc":
                cc2 = avg_array
    results = dict(c1 = c1, c2 = c2, ac1 = ac1, ac2 = ac2, p1 = p1, p2 = p2, n1 = n1, n2 = n2, wc1 = wc1, wc2 = wc2, cc1 = cc1, cc2 = cc2)
    return results

def quantify_avg_sst(results: dict, cat_a: str, cat_b: str): # category means either "circ", "SLA" or "SSTA"
    keys_dict = dict(circ = ["c1", "c2", "ac1", "ac2"], SLA = ["p1", "p2", "n1", "n2"], SSTA = ["wc1", "wc2", "cc1", "cc2"])
    keys_list_a = keys_dict[cat_a]
    keys_list_b = keys_dict[cat_b]
    diff_type1 = 0
    diff_type2 = 0
    for i in [0, 1]:                # type 1
        arr_a = results[keys_list_a[i]]
        arr_b = results[keys_list_b[i]]
        for j in range(len(arr_a)):
            value1 = arr_a[j]
            value2 = arr_b[j]
            diff_type1 += abs(value1-value2)
    for i in [2, 3]:                # type 2
        arr_a = results[keys_list_a[i]]
        arr_b = results[keys_list_b[i]]
        for j in range(len(arr_a)): 
            value1 = arr_a[j]
            value2 = arr_b[j]
            diff_type2 += abs(value1-value2)
    return diff_type1, diff_type2


# debugging:
if __name__=="__main__":
    debug = "disznósajt"
    res = load_avg_sst(results_path)
    d1, d2 = quantify_avg_sst(res, "circ", "SLA")
    print(d1)
    print(d2)
    # TODO:
    # Test and run last two functions here