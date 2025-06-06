#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:21:20 2023
Generate fraction of gridMET gridcell in each US ecozone
@author: liuming
"""
import pandas as pd
import datetime
from datetime import timedelta
from pathlib import Path
import struct
from pathlib import Path
import sys 

 
#process data

in_gridid_lat_lon = sys.argv[1] #"/home/liuming/mnt/hydronas3/Projects/Rangeland/gridmet_file_list.csv"
#GRID_ID,lat,lon
in_county_gridmet_count_fraction = sys.argv[2] #"/home/liuming/mnt/hydronas3/Projects/Rangeland/county_gridmet_range_count_fraction.csv"
#county,gridmet,count,county_all,fraction

in_gridmet_path = sys.argv[3] #"/home/liuming/mnt/hydronas3/Projects/Rangeland/gridmet_monthly_indices/"

out_climate_indicex = sys.argv[4] #"/home/liuming/mnt/hydronas3/Projects/Rangeland/county_gridmet_mean_indices.csv"

#read gridmet filename
gridmet_info = dict()
with open(in_gridid_lat_lon,'r') as f:
    for line in f:
        a = line.rstrip().split(',')
        if 'GRID_ID' not in a:
            #print('grid:' + a[0] + '\tlat:' + a[1] + '\tlon:' + a[2])
            gridmet_info[a[0]] = [a[1],a[2]]
            
#read and generate indexes
#gridmet_data = dict()
gridmet_indexes = dict()
county_indices = dict()
county_total_fraction = dict()

def set_in_spei_range(x):
    y = x
    if x >= 3:
        y = 3
    elif x <= -3:
        y = -3
    return y
import gc
with open(in_county_gridmet_count_fraction,'r') as f:
    for line in f:
        a = line.rstrip().split(',')
        if 'gridmet' not in a:
            county = a[0]
            gridmet = a[1]
            #process all gridmet grid cells
            #if gridmet not in gridmet_indexes:
            gridmet_indexes = pd.DataFrame()
            if True:
                #read and process this gridmet indices
                if gridmet in gridmet_info:
                    #filename = in_gridmet_path + 'data_' + gridmet_info[gridmet][0] + '_' + gridmet_info[gridmet][1]
                    filename = in_gridmet_path + 'spei_' + gridmet + '.csv'
                    if Path(filename).is_file():
                        #print('filename:' + filename)
                        #vic_metdata = read_VIC_binary(filename)
                        #gridmet_data[a[1]] = vic_metdata
                        gridmet_indexes = pd.read_csv(filename,header=0,dtype={'year': int,'month': int})
                        for col in gridmet_indexes.columns:
                            if 'spei' in col:
                                gridmet_indexes[col] = gridmet_indexes[col].apply(set_in_spei_range)
                        #print('read done!')
            #added into county table
            #for checking total fraction equal one
            fraction = float(a[4])
            if county not in county_total_fraction:
                county_total_fraction[county] = fraction
            else:
                county_total_fraction[county] += fraction
                
            #add gridmet fractional indices into county table
            if county not in county_indices and not gridmet_indexes.empty:
                county_indices[county] = gridmet_indexes.copy()
                county_indices[county]['total_fraction'] = fraction
                for col in county_indices[county].columns:
                    if col not in ['year','month','total_fraction']:
                        county_indices[county][col] = fraction * county_indices[county][col]
            elif not gridmet_indexes.empty:
                #add new gridmet indices with weighted by fraction
                if True:
                    for col in county_indices[county].columns:
                        if col not in ['year','month','total_fraction']:
                            county_indices[county][col] = county_indices[county][col] + fraction * gridmet_indexes[col]
                    county_indices[county]['total_fraction'] = county_indices[county]['total_fraction'] + fraction                          
            #print('county:' + county + '\tgridmet:' + gridmet)
            del gridmet_indexes
            gc.collect()
                            
                            
print('Done processing data!')
#merge all conties into one data frame
all_county_indices = pd.DataFrame()
for county in county_indices:
    if county in county_total_fraction:
        if county_total_fraction[county] < 0.9999 or county_total_fraction[county] > 1.0001:
            print("Warning: county:" + county + " total fraction is not 1!")
    for col in county_indices[county].columns:
        if col not in ['year','month','total_fraction']:
            county_indices[county][col] = county_indices[county][col] / county_indices[county]['total_fraction']
    county_indices[county]['county'] = county
    if all_county_indices.empty:
        all_county_indices = county_indices[county]
    else:
        all_county_indices = all_county_indices.append(county_indices[county])
#export
all_county_indices[['year','month']] = all_county_indices[['year','month']].astype(int)
all_county_indices.to_csv(out_climate_indicex,index=False,float_format='%.3f')
print('Done!')