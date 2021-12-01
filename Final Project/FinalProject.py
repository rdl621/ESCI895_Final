#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:13:10 2021

@author: rlevea
"""

#%%
#External libraries
# This cell imports libraries that this code uses

import numpy as np                   # functions for data analysis 
import pandas as pd                  # functions for data frames
from matplotlib import pyplot as plt
import datetime 

#%% Constants

startdate = datetime.datetime(1969, 1, 1)
enddate = datetime.datetime(2020, 12, 31)

headerlist = ['Date Time', 'Water Level', 'I', 'L']

#%% Ingesting, organizing, and filling missing water level data

waterlevelfiles =  ['olc68to77.csv', 'olc78to87.csv', 'olc88to97.csv', 'olc98to07.csv',
               'olc08to17.csv', 'olc18to20.csv', 'Cape68to77.csv', 'Cape78to87.csv',
               'Cape88to97.csv', 'Cape98to07.csv', 'Cape08to17.csv', 'Cape18to20.csv',
               'Oz68to77.csv', 'Oz78to87.csv', 'Oz88to97.csv', 'Oz98to07.csv', 
               'Oz08to17.csv', 'Oz18to20.csv', 'Roch68to77.csv', 'Roch78to87.csv', 
               'Roch88to97.csv', 'Roch98to07.csv', 'Roch08to17.csv', 'Roch18to20.csv', 
               ]

dscgfiles = ['niagdscg.csv', 'lawdscg.csv', 'genndscg.csv', 'blkdscg.csv', 'ozdscg.csv']

prcpfile = ['ozprcp.csv', 'rochprcp.csv', 'wtprcp.csv']

#%% Importing Water Level Data

olcdflvl0 = pd.read_csv(waterlevelfiles[0], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
olcdflvl0.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

olcdflvl1 = pd.read_csv(waterlevelfiles[1], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
olcdflvl1.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

olcdflvl2 = pd.read_csv(waterlevelfiles[2], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
olcdflvl2.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

olcdflvl3 = pd.read_csv(waterlevelfiles[3], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
olcdflvl3.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

olcdflvl4 = pd.read_csv(waterlevelfiles[4], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
olcdflvl4.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

olcdflvl5 = pd.read_csv(waterlevelfiles[5], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
olcdflvl5.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

#Joining into one site dataframe, trimming to the date, renaming columns
olcdflvl = pd.concat([olcdflvl0, olcdflvl1, olcdflvl2, olcdflvl3, olcdflvl4, 
                      olcdflvl5], axis=0, join='outer', ignore_index=False)
olcdflvl = olcdflvl[startdate:enddate]
olcdflvl.rename(columns = {'Water Level' : 'Olcott'}, inplace = True)

##############################################################################  Breaks between sites

capedflvl0 = pd.read_csv(waterlevelfiles[6], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
capedflvl0.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

capedflvl1 = pd.read_csv(waterlevelfiles[7], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
capedflvl1.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

capedflvl2 = pd.read_csv(waterlevelfiles[8], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
capedflvl2.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

capedflvl3 = pd.read_csv(waterlevelfiles[9], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
capedflvl3.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

capedflvl4 = pd.read_csv(waterlevelfiles[10], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
capedflvl4.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

capedflvl5 = pd.read_csv(waterlevelfiles[11], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
capedflvl5.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

#Joining into one site dataframe, trimming to the dates, renaming columns
capedflvl = pd.concat([capedflvl0, capedflvl1, capedflvl2, capedflvl3, capedflvl4,
                       capedflvl5], axis=0, join='outer', ignore_index=False)
capedflvl = capedflvl[startdate:enddate]
capedflvl.rename(columns = {'Water Level' : 'Cape Vincent'}, inplace = True)

##############################################################################  Breaks between sites

ozdflvl0 = pd.read_csv(waterlevelfiles[12], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
ozdflvl0.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

ozdflvl1 = pd.read_csv(waterlevelfiles[13], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
ozdflvl1.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

ozdflvl2 = pd.read_csv(waterlevelfiles[14], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
ozdflvl2.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

ozdflvl3 = pd.read_csv(waterlevelfiles[15], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
ozdflvl3.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

ozdflvl4 = pd.read_csv(waterlevelfiles[16], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
ozdflvl4.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

ozdflvl5 = pd.read_csv(waterlevelfiles[17], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
ozdflvl5.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

#Joining into one site dataframe, trimming to the dates, renaming columns
ozdflvl = pd.concat([ozdflvl0, ozdflvl1, ozdflvl2, ozdflvl3, ozdflvl4,
                       ozdflvl5], axis=0, join='outer', ignore_index=False)
ozdflvl = ozdflvl[startdate:enddate]
ozdflvl.rename(columns = {'Water Level' : 'Oswego'}, inplace = True)

##############################################################################  Breaks between sites

rochdflvl0 = pd.read_csv(waterlevelfiles[18], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
rochdflvl0.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

rochdflvl1 = pd.read_csv(waterlevelfiles[19], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
rochdflvl1.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

rochdflvl2 = pd.read_csv(waterlevelfiles[20], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
rochdflvl2.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

rochdflvl3 = pd.read_csv(waterlevelfiles[21], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
rochdflvl3.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

rochdflvl4 = pd.read_csv(waterlevelfiles[22], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
rochdflvl4.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

rochdflvl5 = pd.read_csv(waterlevelfiles[23], delimiter=',', comment='#', header=0, 
                 parse_dates=['Date Time'], index_col= 'Date Time', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
rochdflvl5.rename(columns=lambda x: x.strip(), inplace=True) #stripping spaces out of names

#Joining into one site dataframe, trimming to the dates, renaming columns
rochdflvl = pd.concat([rochdflvl0, rochdflvl1, rochdflvl2, rochdflvl3, rochdflvl4,
                       rochdflvl5], axis=0, join='outer', ignore_index=False)
rochdflvl = rochdflvl[startdate:enddate]
rochdflvl.rename(columns = {'Water Level' : 'Rochester'}, inplace = True)

#Keeping only water level columns which has been named with site names
olcdflvl = olcdflvl[['Olcott']]
capedflvl = capedflvl[['Cape Vincent']]
ozdflvl = ozdflvl[['Oswego']]
rochdflvl = rochdflvl[['Rochester']]

#resamlping to ensure all days are covered, and linearly interpolating any missing data
olcdflvl = olcdflvl.resample('1D').interpolate('linear') #large stretch from mid '99 to mid '00 missing
capedflvl = capedflvl.resample('1D').interpolate('linear')
ozdflvl = ozdflvl.resample('1D').interpolate('linear')
rochdflvl = rochdflvl.resample('1D').interpolate('linear')

#Joining into one main dataframe with all waterlevel data
waterlevel = pd.concat([olcdflvl, capedflvl, ozdflvl, rochdflvl], axis = 1) 

#Calculating standard dev
stdev = waterlevel.std(axis = 1)

#Getting average lake level
avgLL = pd.DataFrame()
avgLL['avg'] = waterlevel.mean(axis = 1)

maximumwldiff = max(stdev) #ft
avgwldiff = stdev.mean()
maximumwldiffdate = datetime.datetime(2000, 2, 18)

#%% Importing Discharge data

niagdscg = pd.read_csv(dscgfiles[0], delimiter='\t', comment='#', header=1, 
                 parse_dates=['20d'], index_col= '20d', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
niagdscg = niagdscg[['14n']]
niagdscg.rename(columns={'14n':'Q Niag'}, inplace = True)


genndscg = pd.read_csv(dscgfiles[2], delimiter='\t', comment='#', header=1, 
                 parse_dates=['20d'], index_col= '20d', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
genndscg = genndscg[['14n']]
genndscg.rename(columns={'14n':'Q Genn'}, inplace = True)


blackdscg = pd.read_csv(dscgfiles[3], delimiter='\t', comment='#', header=1, 
                 parse_dates=['20d'], index_col= '20d', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
blackdscg = blackdscg[['14n']]
blackdscg.rename(columns={'14n':'Q Black'}, inplace = True)


ozdscg = pd.read_csv(dscgfiles[4], delimiter='\t', comment='#', header=1, 
                 parse_dates=['20d'], index_col= '20d', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
ozdscg = ozdscg[['14n']]
ozdscg.rename(columns={'14n':'Q Oz'}, inplace = True)


lawrencedscg = pd.read_csv(dscgfiles[1], delimiter='\t', comment='#', header=1, 
                 parse_dates=['20d'], index_col= '20d', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
lawrencedscg = lawrencedscg[['14n']]
lawrencedscg['14n'] = lawrencedscg['14n']*86400
lawrencedscg.rename(columns={'14n':'Q Out'}, inplace = True)


discharge = pd.concat([niagdscg, lawrencedscg, genndscg, blackdscg, ozdscg], axis = 1)

discharge['qsum'] = discharge.iloc[:, [0, 2, 3, 4]].sum(axis = 1)
discharge['Q In'] = discharge['qsum']*86400 #cubic feet per day
discharge['Q Difference'] = discharge['Q In'] - discharge['Q Out']
#%% Importing precip data

ozprcp = pd.read_csv(prcpfile[0], delimiter=',', comment='#', header=0, 
                 parse_dates=['DATE'], index_col= 'DATE', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
ozprcp = ozprcp[['PRCP']]
ozprcp.rename(columns={'PRCP':'Oz P'}, inplace = True)
ozprcp = ozprcp.resample('1D').asfreq().fillna(0) #filling NaNs with 0

rochprcp = pd.read_csv(prcpfile[1], delimiter=',', comment='#', header=0, 
                 parse_dates=['DATE'], index_col= 'DATE', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
rochprcp = rochprcp[['PRCP']]
rochprcp.rename(columns={'PRCP':'Roch P'}, inplace = True)
rochprcp = rochprcp.resample('1D').asfreq().fillna(0) #filling NaNs with 0

wtprcp = pd.read_csv(prcpfile[2], delimiter=',', comment='#', header=0, 
                 parse_dates=['DATE'], index_col= 'DATE', na_values = 
                  [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])
wtprcp = wtprcp[['PRCP']]
wtprcp.rename(columns={'PRCP':'WT P'}, inplace = True)
wtprcp = wtprcp.resample('1D').asfreq().fillna(0) #filling NaNs with 0

precipitation = pd.concat([ozprcp, rochprcp, wtprcp], axis = 1)

precipitation['total'] = precipitation.sum(axis = 1)

#llx, lly = np.polyfit(avgLL.index.view('int64')/1e9, avgLL['avg'], 1)

#%%

fig, (ax) = plt.subplots()

ax.plot(avgLL, color = 'r')
ax2 = ax.twinx()
ax2.plot(discharge['Q Difference'], linestyle = 'dotted')
ax3 = ax.twinx()
ax3.plot(precipitation['total'], color = 'k')

#ax.plot(llx, lly)

#%% Defining flood events and years

#plan 2014 was agreed to and enacted in 2016, previous plan was 1958D which was
# in use since 1963 (floods 1973-1998 all plan 1958D)
floodyear = ['1952', '1973', '1976', '1983', '1998', '2017', '2019']

for i in floodyear:
    floodyear.append(pd.to_datetime(i))



#%%

# def timeseriesplot(plotframe):
#     fig, (axes)= plt.subplots (5, sharex=True)
    
#     axes[0].plot(plotframe['Olcott'], linestyle = '-',
#                     label ='olc')
#     axes[0].set_ylabel('olc')
    
#     axes[1].plot(plotframe['Rochester'], linestyle = '-',
#                    label='Roch')
#     axes[1].set_ylabel('Roch')
    
#     axes[2].plot(plotframe['Oswego'], linestyle = '-',
#                    label ='oz')
#     axes[2].set_ylabel('oz')
    
#     axes[3].plot(plotframe['Cape Vincent'], linestyle = '-',
#                    label ='Cape')
#     axes[3].set_ylabel('Cape')
    
#     axes[4].plot(plotframe['Average Lake Level'], linestyle = '-', 
#                    label ='Avg') 
#     axes[4].set_ylabel('Average')


# timeseriesplot(waterlevel)



#%%
# #%%
# fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex = True)

# ax1.plot(olcdflvl, label = 'Olcott')
# ax1.set_ylim(243, 250)

# ax2.plot(capedflvl, label = 'Cape Vincent')
# ax2.set_ylim(243, 250)

# ax3.plot(ozdflvl, label = 'Oswego')
# ax3.set_ylim(243, 250)

# ax4.plot(rochdflvl, label = 'Rochester')
# ax4.set_ylim(243, 250)

# ax5.plot(waterlevel['Average Lake Level'])
# ax5.set_ylim(243, 250)

# # fig, (ax1) = plt.subplots(1, 1, sharex = True)

# # ax1.plot(olcdflvl, olcdflvl.index, capedflvl, ozdflvl, rochdflvl, linewidth = 0.25)
# # ax1.set_ylim(243, 250)

# #fig.legend()

# plt.show()



# fig, (axes)= plt.subplots (5, sharex=True)

# axes[0].plot(waterlevel['Olcott'], linestyle = '-',
#                 label ='olc')
# axes[0].set_ylabel('olc')

# axes[1].plot(waterlevel['Rochester'], linestyle = '-',
#                label='Roch')
# axes[1].set_ylabel('Roch')

# axes[2].plot(waterlevel['Oswego'], linestyle = '-',
#                label ='oz')
# axes[2].set_ylabel('oz')

# axes[3].plot(waterlevel['Cape Vincent'], linestyle = '-',
#                label ='Cape')
# axes[3].set_ylabel('Cape')

# axes[4].plot(waterlevel, linestyle = '-', 
#                label ='All') 
# axes[4].set_ylabel('All')

# axes[4].plot(waterlevelavg, linestyle = 'dashdot', color = 'r')


#%% Showing problem

# olcdflvl0 = pd.read_csv(waterlevelfiles[0], delimiter=',', comment='#', 
                 # parse_dates=['Date Time'], names= ['Date Time', 'Water Level'], index_col= 'Date Time', na_values = 
                  # [2.0181e+11, 2.01902e+11, -9999, 9999, 'NaN', 'Ice', 'Eqp'])

# olcdflvl0.drop(columns={'I', 'L'}, inplace=True)

#%%

# olcdflvl0 = pd.read_csv(waterlevelfiles[0], delimiter=',', header = None) 

# olcdflvl0 = pd.read_csv(waterlevelfiles[0]) 

# olcdflvl0 = olcdflvl0.to_csv(waterlevelfiles[0], header = headerlist, index = False)

# olcdflvl0 = olcdflvl0.drop(columns={2, 3})

# olcdflvl0.set_index(0, inplace=True)




