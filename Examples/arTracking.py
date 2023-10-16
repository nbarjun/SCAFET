# from dask.distributed import Client
# client = Client(scheduler_file='/proj/arjun/MPI/scheduler.json')  # set up local cluster on your laptop
# client

import numpy as np
import xarray as xr
import pandas as pd
import time
from glob import glob
import metpy.calc as mpcalc
from metpy.units import units
from importlib import reload
import metpy.xarray as mxr
import itertools
from geopy import distance
from tqdm import tqdm
from datetime import datetime, timedelta, date
import dask

import object_properties as obp
import ridge_detection as rd
import object_filtering as obf
import object_tracking as obt

ivt_grid_area = xr.open_dataset('/proj/arjun/CESM/Atmosp_Rivers/GuanWaliserV1/river2/grid_area_cesm.nc')
ivt_land = xr.open_dataset('/proj/arjun/CESM/Atmosp_Rivers/GuanWaliserV1/river2/land_mask_PD.nc')

reload(obp)
smooth_scale = 2e6
angle_threshold = 45
shape_index = [0.375,1]
min_length = 2000e3
min_area = 2e12
min_duration = 1
max_distance_per_tstep = 1000e3
shape_eccentricity = [0.75,1.]
lat_mask = [-20,20]
lon_mask = [360,0]

properties = obp.object_properties2D(ivt_grid_area,ivt_land,min_length,min_area,\
                    smooth_scale,angle_threshold,min_duration,max_distance_per_tstep,\
                    shape_index,shape_eccentricity,\
                    lon_mask,lat_mask)

outfolder = '/proj/arjun/CESM/Atmosp_Rivers/SCAFET/Output/'
for exp in ['PD','2xCO2','4xCO2']:
    print('Tracking {}'.format(exp))
    folder = '/proj/arjun/CESM/Atmosp_Rivers/SCAFEX_V1/{}/'.format(exp)
    ofiles = sorted(glob(folder+'ar_objs*.nc'))
    pfiles = sorted(glob(folder+'ar_props*.nc'))
    
    # Opening Files
    ar_objects = xr.open_mfdataset(ofiles,combine='by_coords')
    ar_props = xr.open_mfdataset(pfiles,combine='by_coords')
    
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
    prfile = '{}arTracked_props_{}.nc'.format(outfolder,exp)
    object_properties.to_netcdf('{}'.\
                    format(prfile),encoding=encod)
    print('Saved {}'.format(prfile))
        
    objects = tracked[1].to_dataset(name='object')
    encod = {}
    for v in objects.data_vars:
            encod[v]={'zlib':True,'complevel':9,'shuffle':True}
    for p in saved_props:
        objects.attrs[p] = saved_props[p]
    obfile = '{}arTracked_objs_{}.nc'.format(outfolder,exp)
    objects.to_netcdf('{}'.\
                        format(obfile),encoding=encod)
    print('Saved {}'.format(obfile))