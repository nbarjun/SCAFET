import numpy as np
import xarray as xr
import pandas as pd
import time
from glob import glob
import metpy.calc as mpcalc
from tqdm import tqdm
from datetime import datetime, timedelta, date
import dask
import platform

import object_properties as obp
import ridge_detection as rd
import object_filtering as obf
import object_tracking as obt


folder = '/Volumes/RAID5/Data/Atmosp_Rivers/ESGF/CESM2/'
grid_area = xr.open_mfdataset('{}areacella_fx_CESM2_historical_r6i1p1f1_gn.nc'.format(folder))\
                    ['areacella'].to_dataset(name='cell_area').load()
land_mask =  xr.open_mfdataset('{}sftlf_fx_CESM2_historical_r6i1p1f1_gn.nc'.format(folder))
land_mask = land_mask['sftlf'].where(land_mask['sftlf']>.5)*0.+1.
land_mask = land_mask.squeeze().transpose('lat','lon').to_dataset(name='islnd').load()

smooth_scale = 2e6
angle_threshold = 45
shape_index = [0.375,1]
min_length = 2000e3
min_area = 2e11
min_duration = 12
max_distance_per_tstep = 500e3
shape_eccentricity = [0.5,1.0]
lat_mask = [-20,20]
lon_mask = [360,0]

properties = obp.object_properties2D(grid_area,land_mask,min_length,min_area,\
                    smooth_scale,angle_threshold,min_duration,max_distance_per_tstep,\
                    shape_index,shape_eccentricity,\
                    lon_mask,lat_mask)

exps='CESM2.ssp370'
infolder='/Volumes/RAID5/Data/Atmosp_Rivers/ESGF/{}/Output/'.format(exps[:5])
outfolder='/Volumes/RAID5/Data/Atmosp_Rivers/ESGF/{}/'.format(exps[:5])
arfiles = sorted(glob(infolder+'arObjects.{}.*.nc'.format(exps)))
prfiles=sorted(glob(infolder+'arProps.{}.*.nc'.format(exps)))

print(arfiles)
print('\n')
print(prfiles)

ar_objects = xr.open_mfdataset(arfiles,combine='by_coords')
ar_props = xr.open_mfdataset(prfiles,combine='by_coords')


# Tracking
properties.obj['Min_Duration']=1
properties.obj['Max_Distance']=4000

# latlon = ['mx_lat1','mx_lon1']
latlon = ['wclat','wclon']

tracker = obt.Tracker(latlon,properties)
tracked = tracker.apply_tracking(ar_props,ar_objects)


saved_props = properties.obj.copy()
del saved_props['Min_Lon_Points']
del saved_props['Min_Lat_Points']
del saved_props['Map_Lons']
del saved_props['Map_Lats']
del saved_props['isglobal']

object_properties = tracked[0]
encod = {}
for v in object_properties.data_vars:
    encod[v]={'zlib':True,'complevel':9,'shuffle':True}
for p in saved_props:
    object_properties.attrs[p] = saved_props[p]

object_properties.attrs['Created By']='Arjun Babu Nellikkattil (arjunbabun@pusan.ac.kr)'
object_properties.attrs['Detected Using']='Scalable Feature Extraction and Tracking SCAFET in {}'\
                                                        .format(platform.platform())
object_properties.attrs['Date Created']=datetime.now().strftime("%Y/%m/%d, %H:%M:%S")   
prfile = '{}SCAFET_Tracked_props.{}.nc'.format(outfolder,exps)
object_properties.to_netcdf('{}'.\
            format(prfile),encoding=encod)
print('Saved {}'.format(prfile))

objects = tracked[1].to_dataset(name='object')
encod = {}
for v in objects.data_vars:
    encod[v]={'zlib':True,'complevel':9,'shuffle':True}
for p in saved_props:
    objects.attrs[p] = saved_props[p]

objects.attrs['Created By']='Arjun Babu Nellikkattil (arjunbabun@pusan.ac.kr)'
objects.attrs['Detected Using']='Scalable Feature Extraction and Tracking SCAFET in {}'\
                                                        .format(platform.platform())
objects.attrs['Date Created']=datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
obfile = '{}SCAFET_Tracked_objs.{}.nc'.format(outfolder,exps)
objects.to_netcdf('{}'.\
                format(obfile),encoding=encod)
print('Saved {}'.format(obfile))

