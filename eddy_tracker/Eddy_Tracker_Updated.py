import numpy as np
import matplotlib.pyplot as plt
import math


#Sample array to test the code. Will return 2D numpy arrays for each week representing random sample values.
N=50 #Number of sample center points
Week1, Week2, Week3, Week4, Week5 = 10* (np.random.random((N,2))),10* (np.random.random((N,2))),10* (np.random.random((N,2))),10* (np.random.random((N,2))),10* (np.random.random((N,2)))


#get centre points from multiple weeks. Smaller random set to test 

centrePointsWeek1 = [[2,2], [42,53], [12,18]]
centrePointsWeek2 = [[4,5], [11,20], [44,52]]
centrePointsWeek3 = [[3,4], [12,21], [43,51]]
centrePointsWeek4 = [[2,3], [13,21], [43,50]]
centrePointsWeek5 = [[1,2], [15,21], [45,49]]

centrePoints = [centrePointsWeek1, centrePointsWeek2, centrePointsWeek3, centrePointsWeek4, centrePointsWeek5]

#get information on radius of eddies
eddieRadiiWeek1 = [0.50, 10.0, 5.00]
eddieRadiiWeek2 = [0.51, 4.99, 10.2]
eddieRadiiWeek3 = [0.50, 4.97, 10.3]
eddieRadiiWeek4 = [0.49, 4,95, 10.4]
eddieRadiiWeek5 = [0.48, 4.92, 10.5]


#compare the centre point from the previous week to all the centre points from the next week

# distHistory = [[] for x in range(len(centrePointsWeek1))]   #Initialise list of lists containing position of eddies overtime

# for i in range(len(centrePointsWeek1)):                     #Add initial positions of eddies in the list
#     distHistory[i].append(centrePointsWeek1[i])

# ###CHANGE CODE TO TAKE INTO ACCOUNT THAT NEW EDDIES ARE GOING TO APPEAR
    
# for i in range(len(centrePoints) - 1):  # Loop through all of the weeks to calculate the distances between all of the centre points for all weeks
#     initWeek = centrePoints[i]  # Define the previous week and the next week
#     afterWeek = centrePoints[i + 1]
#     initDist = []  # List cointaining lists

#     IcentreIndex = 0  # Index of the eddy being taken into account

#     for Icentre in initWeek:  # For each centre point in the previous week, we will find the next centre point
#         dist = []  # Initialise a list to store all of the distances between the old centre point and all new ones
#         index = 0

#         for Acentre in afterWeek:  # For each new centre point in the new week
#             distance = math.sqrt((Acentre[0] - Icentre[0]) ** 2 + (
#                         Acentre[1] - Icentre[1]) ** 2)  # Calculate the distance to the old centre point

#             if distance <= 5:  # If the distance if within a reasonable limit
#                 dist.append((distance, index))  # The distance and new data point is considered as a possible next point

#             index += 1
#         #print(afterWeek[min(dist)[1]])
#         distHistory[IcentreIndex].append(afterWeek[min(dist)[1]]) #Appends new position of the centre of the eddy

#         initDist.append(dist)  # Add the distances to a larger list
#         IcentreIndex += 1

    
#     print(initDist) 

# #initWeek the same one 
# #new afterWeek, will have the the centre coordinates matching with initWeek, so that at index[0], in both lists the eddy
# #will be the same


# lis = [[[0,1], [0,2],[0,3]], [[2,1], [4,4],[3,7]], [[5,1], [8,4],[4,7]]]        #Sample list again

# #Attempt to plot
# eddy1 = lis[1]      #Choose the eddy 
# x_cord=[]
# y_cord = []
# for week in eddy1:
#     x_cord.append(week[0])
#     y_cord.append(week[1])
        

# plt.plot(x_cord,y_cord)
# plt.title("Sample plot of one eddy")
# plt.show()


# WHAT TO ADD:

# Display all eddy paths on one graph
# If new eddies appear
# If eddies disappear (Check 3 weeks after to make sure an eddy has disappeared)
# Filter out eddies that last less than 30 days
# If the cross over/collide
#Method from research paper for collision instances

#New Code

def HDdistance (mainEddy, candidates):
    closestEddyDist = 100000                    #Set a really high closestEddyDist to initialise the variable
    for i in range(len(candidates)):            #Loop through all of the candidates
        HDdist = math.sqrt((mainEddy[0]-candidates[i][0])**2 + (mainEddy[1]-candidates[i][1])**2)   #Find the distances for each candidate
        if HDdist < closestEddyDist:            #If a candidate is closer to the main eddy that the others, it becomes the closestEddy
            closestEddy = candidates [i]

    return closestEddy                          #The final closest eddy from the candidates is returned.

# print (HDdistance([20, 20], [[15, 17], [5, 5], [30, 20], [21, 22]]))


def check_eddies(mainEddy, newEddies): 
    candidates = []
    mainEddy_long = (mainEddy[0])[1]

    mainEddy_lat = mainEddy[0][0]

    mainEddy_area = mainEddy[1]
    mainEddy_amp = mainEddy[2]
    for newEddy in newEddies:
        newEddy_long = newEddy[0][1] 
        newEddy_lat = newEddy[0][0]
        newEddy_area = newEddy[1] 
        newEddy_amp = newEddy[2]
        
        if (abs(mainEddy_long - newEddy_long) <= 1.5) and ((abs(mainEddy_lat - newEddy_lat) <= 1.35)): #1.35deg latitude = 150 km  
                                 #1.5 deg long = 150 km these values are temporary                              
    
            if ((newEddy_area >= 0.25*mainEddy_area) and (newEddy_area <= 2.75*mainEddy_area)):  #check if newEddy is in correct range for area
                if ((newEddy_amp >= 0.25*mainEddy_amp) and (newEddy_amp <= 2.75*mainEddy_amp)):  #check if newEddy is in correct range for amplitude
                    candidates.append(newEddy)       
                                      
    return candidates # returns list containing potential eddies







ex_list = [[[(2, 2), 0.50, 5], [(42, 53), 10.0, 15], [(12, 18), 5.00, 4]], [[(2.2, 1.9), 0.51, 4.95], [(42.1, 52.8), 10.2, 15.1], [(11.95, 18.1), 4.99, 4.2]], [[(2.3, 2), 0.51, 4.95], [(42.2, 52.9), 10.2, 15.1]], [], [[(42.3, 52.8), 10.1, 15.2], [(2.35, 2.1), 0.52, 4.97], [(70.35, 90.1), 0.52, 4.97]]]


search_range = 150 #Sample 
ex_output = [[(22,33,1), (44,33,2)], []]
output = [] #Output initialization
current_week_counter = 1 #Variable

for week in ex_list:

    for eddy in week:
        eddy_list = [[eddy, current_week_counter]]               #list containing all position of specific eddy through time
        
        disappeared = False
        new_week_counter = current_week_counter + 1
        for new_week in ex_list[(current_week_counter):]:       #loop through all weeks to find position of eddy
                 
            main_eddy = eddy_list[-1][0] 
                                      #get last position of eddy that has been found for analysis
            candidates = check_eddies(main_eddy, new_week)           #get candidates 
            
            if len(candidates) == 1:                                  #if only one candidate found, append
                eddy_list.append([candidates[0], new_week_counter])
                disappeared = False
                new_week.remove(candidates[0])                      #Delete eddy once it has been added
                #Delete eddy
              
            elif len(candidates) > 1:                                 #if more than one candidate, append the one with smallest distance
                closest_eddy = HDdistance(main_eddy, candidates)
                eddy_list.append([closest_eddy, new_week_counter])
                disappeared = False
                new_week.remove(closest_eddy)
                #Delete eddy
                
            elif len(candidates) == 0:
                if disappeared == True:
                    break
                else:
                    disappeared = True
            new_week_counter += 1   
        output.append(eddy_list)  
          
    current_week_counter += 1
                
for i in output:
    #print(output)
    print("eddy",i)



#Plotting 
for eddy in output:
    x_cords=[]
    y_cords=[]
    for week in eddy:
        x = week[0][0][0]
        y = week[0][0][1]
        x_cords.append(x)
        y_cords.append(y)
    plt.plot(x_cords,y_cords)
plt.savefig("filepath.svg", format = 'svg', dpi=300)


 # addidional_week_eddy = ex_list[(current_week_counter + 1)]
                # new_candidates = check_eddies(main_eddy,addidional_week_eddy)
                    
                # if len(new_candidates) == 1:                                  #if only one candidate found, append
                #     eddy_list.append(new_candidates[0], new_week_counter + 1)
                #     #Delete eddy
                    
                # elif len(new_candidates) > 1:                               #if more than one candidtate, append the one with smallest distance
                #     closest_eddy = HDdistance(main_eddy, candidates)
                #     eddy_list.append(closest_eddy, new_week_counter + 1)
                #     #Delete eddy
                    
                # elif new_candidates == False:
                #     break 
                
        
    # ####
    # old = [] #Eddies in week
    # new=[] #Eddies in the next week
    # newEddies = ex_list[week+1] #Returns properties of eddies the next week
    # oneweek=ex_list[week] #returns eddies on the week

    # for eddy in oneweek:
    #     mainEddy=list(eddy) #Returns each eddy in the week
    #     candidates=check_eddies(mainEddy,newEddies) #Implement function to identify potential canditiates
    #     if len(candidates)==1:#Only one potential eddy
    #         old.append(mainEddy)
    #         new.append(candidates) 
    #     elif len(candidates)>1:#Multiple possible eddies
    #         old.append(mainEddy)
    #         closest_eddy = HDdistance(mainEddy,candidates)
    #         new.append(closest_eddy)
    #     elif candidates==False: #No eddy found, try for next week
    #         if week<len(ex_list-1):
    #             newEddies=ex_list[week+2]
    #             candidates2=check_eddies(mainEddy,newEddies)
    #             if len(candidates2)==1: #One eddy found
    #             old.append(mainEddy)
    #             new.append(candidates)
    #             elif len(candidates2)>1: # multiple eddies found
    #                 old.append(mainEddy)
    #                 closest_eddy = HDdistance(mainEddy,candidates2)
    #                 new.append(closest_eddy)
    #             else: #Eddy is dead
    #                 old.append(mainEddy)
    #                 new.append("Dead")
    #         else: break
    #     else: #New eddy
    #         old.append("None")
    #         new.append(mainEddy) #New eddy
    
        
            
    



