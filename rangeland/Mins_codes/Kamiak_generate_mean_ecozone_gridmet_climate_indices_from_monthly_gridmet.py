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

#VIC binary
VIC_scale = {'PREC' : 40,
             'TMAX' : 100,
             'TMIN' : 100,
             'WIND' : 100,
             'QAIR' : 10000,
             'SHORTWAVE' : 40,
             'RMAX' : 100,
             'RMIN' : 100}
vic_columns = ['date']
for col in VIC_scale:
    vic_columns.append(col)
 
#process data

in_gridid_lat_lon = sys.argv[1] #"/home/liuming/mnt/hydronas3/Projects/Rangeland/gridmet_file_list.csv"
#GRID_ID,lat,lon
in_ecozone_gridmet_count_fraction = sys.argv[2] #"/home/liuming/mnt/hydronas3/Projects/Rangeland/ecozone_gridmet_range_count_fraction.csv"
#ecozone,gridmet,count,ecozone_all,fraction

in_gridmet_path = sys.argv[3] #"/home/liuming/mnt/hydronas3/Projects/Rangeland/gridmet_monthly_indices/"

out_climate_indicex = sys.argv[4] #"/home/liuming/mnt/hydronas3/Projects/Rangeland/ecozone_gridmet_mean_indices.csv"

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
ecozone_indices = dict()
ecozone_total_fraction = dict()
with open(in_ecozone_gridmet_count_fraction,'r') as f:
    for line in f:
        a = line.rstrip().split(',')
        if 'gridmet' not in a:
            ecozone = a[0]
            gridmet = a[1]
            #process all gridmet grid cells
            if gridmet not in gridmet_indexes:
                #read and process this gridmet indices
                if gridmet in gridmet_info:
                    #filename = in_gridmet_path + 'data_' + gridmet_info[gridmet][0] + '_' + gridmet_info[gridmet][1]
                    filename = in_gridmet_path + 'gridmet_' + gridmet + '.csv'
                    if Path(filename).is_file():
                        print('filename:' + filename)
                        #vic_metdata = read_VIC_binary(filename)
                        #gridmet_data[a[1]] = vic_metdata
                        gridmet_indexes[gridmet] = pd.read_csv(filename,header=0)
                        #print('read done!')
            #added into ecozone table
            #for checking total fraction equal one
            fraction = float(a[4])
            if ecozone not in ecozone_total_fraction:
                ecozone_total_fraction[ecozone] = fraction
            else:
                ecozone_total_fraction[ecozone] += fraction
                
            #add gridmet fractional indices into ecozone table
            if ecozone not in ecozone_indices and gridmet in gridmet_indexes:
                ecozone_indices[ecozone] = gridmet_indexes[gridmet]
                for col in ecozone_indices[ecozone].columns:
                    if col not in ['year','month']:
                        ecozone_indices[ecozone][col] = fraction * ecozone_indices[ecozone][col]
            else:
                #add new gridmet indices with weighted by fraction
                if gridmet in gridmet_indexes and gridmet in gridmet_indexes:
                    for col in ecozone_indices[ecozone].columns:
                        if col not in ['year','month']:
                            ecozone_indices[ecozone][col] = ecozone_indices[ecozone][col] + fraction * gridmet_indexes[gridmet][col]
            #print('ecozone:' + ecozone + '\tgridmet:' + gridmet)
                            
                            
print('Done processing data!')
#merge all conties into one data frame
all_ecozone_indices = pd.DataFrame()
for ecozone in ecozone_indices:
    if ecozone in ecozone_total_fraction:
        if ecozone_total_fraction[ecozone] < 0.9999:
            print("Warning: ecozone:" + ecozone + " total fraction is not 1!")
    ecozone_indices[ecozone]['ecozone'] = ecozone
    if all_ecozone_indices.empty:
        all_ecozone_indices = ecozone_indices[ecozone]
    else:
        all_ecozone_indices = all_ecozone_indices.append(ecozone_indices[ecozone])
#export
all_ecozone_indices.to_csv(out_climate_indicex,index=False,float_format='%.3f')
print('Done!')
