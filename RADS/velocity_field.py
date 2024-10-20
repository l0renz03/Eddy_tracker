from rads_funcs import *

# directory of data
filedir = 'RADS/example/'
#filedir = 'RADS/example/'

# Entire Grid: extents = (100, 420, 50, 750) --> 60;28 N -80;-10 W
# Small grid (for method) extents = (100, 420, 50, 750)

if __name__ == "__main__":
    mpl.rc('figure', max_open_warning = 0)
    mpl.use('Agg')
    ### Setup ###
    extents = (100, 420, 50, 750)   
    lon, lat, ext = ext_to_coord(extents)

    # weeks to analyze - either specific weeks given by the list 
    # or all weeks from week 1 up to a limit (incl.; set this to 52 for full data)
    weeks = np.array([27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52])        # Exact weeks
    weeks_limit = 26     # Weeks up to this limit
    weeks = np.arange(1, weeks_limit+1, 1)  

    ow = 1.2 # The negative is taken later
    T = True # option to deal with the temperature stuff
    res = True # option to overwrite results with plots and with saving arrays
    start_days = weeks*7-6
    if T:
        sst_array_list = SST_grid(temperature_path, extents, weeks)

    avg_list = []

    eddy_sst_list_c = []
    eddy_sst_list_c_cumulative = []
    eddy_sst_list_ac = []
    eddy_sst_list_ac_cumulative = []

    eddy_sst_list_cc = []
    eddy_sst_list_cc_cumulative = []
    eddy_sst_list_wc = []
    eddy_sst_list_wc_cumulative = []

    eddy_sst_list_p = []
    eddy_sst_list_p_cumulative = []
    eddy_sst_list_n = []
    eddy_sst_list_n_cumulative = []

    total_no_eddies = 0
    no_c = 0
    no_ac = 0
    no_p = 0
    no_n = 0
    no_cc = 0
    no_wc = 0

    week_list = []
    ticks = []
    labels = []
    # create plot labels:
    for w in weeks:
        ticks.append(w)
        if w % 2 != 0:
            labels.append('W'+str(w))
        else:
            labels.append(" ")

    for start_day in start_days:
        week = start_day//7 + 1
        print("\n","Week: ", week)

        sla_grid, lon_grid, lat_grid = data_grid_selection(extents, start_day, 7, plot=False, week=week)

        week_list.append(int(week))

        #Create folder for imgages:
        folder_to_create = folder_creation + f'Overleaf_images/week{week}'
        os.makedirs(folder_to_create, exist_ok=True)

        one_group, u, v = sla_pipeline(sla_grid, lon_grid, lat_grid, ow_cut= ow, uv_plot = False, ow_plot=False, plot_grouping=False, week = week)

        depths_array, depths_array_filter = depth_grid(depth_path, extents)
        
        if T:
            temp_array = sst_array_list[int(week-1)]
            avg = np.nanmean(temp_array)
            avg_list.append(avg)
            
            output, eddy_mask, sst_dict_circ, sst_dict_sla, sst_dict_ssta, no_eddies = eddy_pipeline(one_group, sla_grid, extents, depths_array_filter, T, u, v, plot = False, week=week, sst_grid = temp_array)
            total_no_eddies += no_eddies

            plot_sla(temp_array, "Temperature_eddy_correlation", "Longitude", "Latitude", lat, lon, cmap="RdBu_r", additional_mask=eddy_mask, week=week, colorbar_label="SST [K]") 

            # temperature based on circulation
            sst_arr_c = np.array(sst_dict_circ["c"])
            sst_arr_ac = np.array(sst_dict_circ["ac"])
            eddy_sst_list_c_cumulative.append(np.sum(sst_arr_c))
            eddy_sst_list_ac_cumulative.append(np.sum(sst_arr_ac))
            no_c += len(sst_arr_c)
            no_ac += len(sst_arr_ac)
            eddy_sst_list_c.append(np.mean(sst_arr_c))
            eddy_sst_list_ac.append(np.mean(sst_arr_ac))
            # temperature based on sla
            sst_arr_p = np.array(sst_dict_sla["p"])
            sst_arr_n = np.array(sst_dict_sla["n"])
            eddy_sst_list_p_cumulative.append(np.sum(sst_arr_p))
            eddy_sst_list_n_cumulative.append(np.sum(sst_arr_n))
            no_p += len(sst_arr_p)
            no_n += len(sst_arr_n)
            eddy_sst_list_p.append(np.mean(sst_arr_p))
            eddy_sst_list_n.append(np.mean(sst_arr_n))
            # temperature based on ssta
            sst_arr_cc = np.array(sst_dict_ssta["cc"])
            sst_arr_wc = np.array(sst_dict_ssta["wc"])
            eddy_sst_list_cc_cumulative.append(np.sum(sst_arr_cc))
            eddy_sst_list_wc_cumulative.append(np.sum(sst_arr_wc))
            no_cc += len(sst_arr_cc)
            no_wc += len(sst_arr_wc)
            eddy_sst_list_cc.append(np.mean(sst_arr_cc))
            eddy_sst_list_wc.append(np.mean(sst_arr_wc))

        else:
            output = eddy_pipeline(one_group, sla_grid, extents, depths_array_filter, T, u, v, plot = False, week=week)[0]

        if T:
            plot_characteristics(1, output, week, ow = ow) #Amplitude
            plot_characteristics(2, output, week, ow = ow) #Area
            plot_characteristics(3, output, week, ow = ow) #Amplitude over area

        plt.clf()
    if T and res:
        avg_c = np.ones(len(weeks))*sum(eddy_sst_list_c_cumulative)/no_c
        avg_ac = np.ones(len(weeks))*sum(eddy_sst_list_ac_cumulative)/no_ac
        avg_p = np.ones(len(weeks))*sum(eddy_sst_list_p_cumulative)/no_p
        avg_n = np.ones(len(weeks))*sum(eddy_sst_list_n_cumulative)/no_n
        avg_cc = np.ones(len(weeks))*sum(eddy_sst_list_cc_cumulative)/no_cc
        avg_wc = np.ones(len(weeks))*sum(eddy_sst_list_wc_cumulative)/no_wc

        print(avg_c[0])
        print(avg_ac[0])
        print(avg_p[0])
        print(avg_n[0])
        print(avg_cc[0])
        print(avg_wc[0])

        plt.plot(week_list, eddy_sst_list_c, label="Cyclonic Eddies", color='orange', marker='.')
        plt.plot(week_list, eddy_sst_list_ac, label="Anticyclonic Eddies", color='blue', marker='.')
        plt.plot(week_list, avg_list, label="Overall SST average", color='plum', marker='.')
        plt.plot(week_list, avg_c, label = "Average temperature of cyclonic eddies", color='orangered')
        plt.plot(week_list, avg_ac, label = "Average temperature of anticyclonic eddies", color='darkblue')
        plt.ylim(284, 304)
        plt.xticks(ticks, labels)
        plt.xlabel("Week number")
        plt.ylabel("SST [K]")
        plt.legend()
        plt.savefig(folder_creation+"Overleaf_images/week0/sst_graph_circ_01.png", bbox_inches='tight', pad_inches = 0)
        plt.savefig(folder_creation+"Overleaf_images/week0/sst_graph_circ_01.svg", bbox_inches='tight', pad_inches = 0)

        plt.clf()

        plt.plot(week_list, eddy_sst_list_p, label="Positive SLA Eddies", color='orange', marker='.')
        plt.plot(week_list, eddy_sst_list_n, label="Negative SLA Eddies", color='blue', marker='.')
        plt.plot(week_list, avg_list, label="Overall SST average", color='plum', marker='.')
        plt.plot(week_list, avg_p, label = "Average temperature of positive SLA eddies", color='orangered')
        plt.plot(week_list, avg_n, label = "Average temperature of negative SLA eddies", color='darkblue')
        plt.ylim(284, 304)
        plt.xticks(ticks, labels)
        plt.xlabel("Week number")
        plt.ylabel("SST [K]")
        plt.legend()
        plt.savefig(folder_creation+"Overleaf_images/week0/sst_graph_SLA_01.png", bbox_inches='tight', pad_inches = 0)
        plt.savefig(folder_creation+"Overleaf_images/week0/sst_graph_SLA_01.svg", bbox_inches='tight', pad_inches = 0)

        plt.clf()

        plt.plot(week_list, eddy_sst_list_wc, label="Warm Core Eddies", color='orange', marker='.')
        plt.plot(week_list, eddy_sst_list_cc, label="Cold Core Eddies", color='blue', marker='.')
        plt.plot(week_list, avg_list, label="Overall SST average", color='plum', marker='.')
        plt.plot(week_list, avg_cc, label = "Average temperature of cold-core eddies", color='darkblue')
        plt.plot(week_list, avg_wc, label = "Average temperature of warm-core eddies", color='orangered')
        plt.ylim(284, 304)
        plt.xticks(ticks, labels)
        plt.xlabel("Week number")
        plt.ylabel("SST [K]")
        plt.legend()
        plt.savefig(folder_creation+"Overleaf_images/week0/sst_graph_SSTA_01.png", bbox_inches='tight', pad_inches = 0)
        plt.savefig(folder_creation+"Overleaf_images/week0/sst_graph_SSTA_01.svg", bbox_inches='tight', pad_inches = 0)

        print("saving results...")
        # np.save("eddy_sst_1c", np.array(eddy_sst_list_c))
        # np.save("eddy_sst_1ac", np.array(eddy_sst_list_ac))
        # np.save("eddy_sst_1p", np.array(eddy_sst_list_p))
        # np.save("eddy_sst_1n", np.array(eddy_sst_list_n))
        # np.save("eddy_sst_1cc", np.array(eddy_sst_list_cc))
        # np.save("eddy_sst_1wc", np.array(eddy_sst_list_wc))

    print("\n", "FINISHED")

    # TODO:
    # check if velocity field is correct
    # color plots accordingly
    # split into two or four (if 4 same as in app)
    # ---
