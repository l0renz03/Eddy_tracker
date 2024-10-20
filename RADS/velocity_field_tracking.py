import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
from rads_funcs import *
import matplotlib as mpl
import os
import scipy as sp
import pandas as pd
import GridResolutionDisplay as GRD


# directory of data
filedir = 'RADS/example/'
#filedir = 'RADS/example/'

def get_data_for_tracker(start_week,end_week):
    mpl.rc('figure', max_open_warning = 0)
    ### Setup ###
    extents = (100, 420, 50, 750)   
    lon, lat, ext = ext_to_coord(extents)
    weeks = 2
    ow = 1.2 # The negative is taken later
    T = False # option to deal with the temperature stuff
    if T:
        sst_array_list = SST_grid(temperature_path, extents, limit=weeks)

    avg_list = []
    eddy_sst_list_c = []
    eddy_sst_list_ac = []
    week_list = []
    ticks = []
    labels = []
    # create plot labels:
    for w in range(weeks):
        ticks.append(w+1)
        labels.append('Week '+str(w+1))

    for start_day in range(1, (weeks*7) + 1, 7):
        week = start_day//7 + 1
        print("\n","Week: ", week)

        sla_grid, lon_grid, lat_grid = data_grid_selection(extents, start_day, 7, plot=False, week=week)

        week_list.append(int(week))

        #Create folder for imgages:
        folder_to_create = folder_creation + f'Overleaf_images/week{week}'
        os.makedirs(folder_to_create, exist_ok=True)

        one_group = sla_pipeline(sla_grid, lon_grid, lat_grid, ow_cut= ow, uv_plot = False, ow_plot=False, plot_grouping=False, week = week)

        depths_array, depths_array_filter = depth_grid(depth_path, extents)
        
        if T:
            temp_array = sst_array_list[int(week-1)]
            avg = np.nanmean(temp_array)
            avg_list.append(avg)
            
            output, sst_dict, eddy_mask = eddy_pipeline(one_group, sla_grid, extents, depths_array_filter, plot = False, week=week, sst_grid = temp_array)
            
            plot_sla(sst_array_list[week-1], "Temperature_eddy_correlation", "Longitude", "Latitude", lat, lon, cmap="RdBu_r", additional_mask=eddy_mask, week=week, colorbar_label="SST [K]")

            sst_arr_c = np.array(sst_dict["a"])
            sst_arr_ac = np.array(sst_dict["ac"])
            eddy_sst_list_c.append(np.mean(sst_arr_c))
            eddy_sst_list_ac.append(np.mean(sst_arr_ac))

        else:
            output = eddy_pipeline(one_group, sla_grid, extents, depths_array_filter, plot = False, week=week)[0]

        if T:
            plot_characteristics(1, output, week, ow = ow) #Amplitude
            plot_characteristics(2, output, week, ow = ow) #Area
            plot_characteristics(3, output, week, ow = ow) #Amplitude over area

        plt.clf()
    if T:
        plt.plot(week_list, eddy_sst_list_c, label="Cyclonic eddies")
        plt.plot(week_list, eddy_sst_list_ac, label="Anticyclonic eddies")
        plt.plot(week_list, avg_list, label="Overall SST average")
        plt.xticks(ticks, labels)
        plt.xlabel("Week number")
        plt.ylabel("SST [K]")
        plt.legend()
        plt.savefig(folder_creation+"Overleaf_images/week0/sst_graph.png", bbox_inches='tight', pad_inches = 0)
        plt.savefig(folder_creation+"Overleaf_images/week0/sst_graph.svg", bbox_inches='tight', pad_inches = 0)

    print("\n", "FINISHED,",len(output),"eddies identified")
    return output

if __name__=="__main__":
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
    import os
    from rads_funcs import *
    import matplotlib as mpl
    import os
    import scipy as sp
    import pandas as pd
    import GridResolutionDisplay as GRD

    # directory of data
    filedir = 'RADS/example/'
    # filedir = 'RADS/example/'
    # dir_memory=r"C:\Users\Misko\OneDrive\Documents\D01\AE2224-I_D01\RADS\getRADSgridAllSat_memory\\"
    extents = (300, 350, 350, 400)

    sla_grid, lon_grid, lat_grid = data_grid_selection(extents, 1, 7, plot=False)

    one_group = sla_pipeline(sla_grid, lon_grid, lat_grid, uv_plot=False, ow_plot=False, plot_grouping=False)

    output = eddy_pipeline(one_group, sla_grid, lat_grid, lon_grid, plot=False)

    #print(output)

    print("FINISHED")