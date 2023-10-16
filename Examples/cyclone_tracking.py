# from dask.distributed import Client
# client = Client(scheduler_file='/Users/mac/ICCP/MPI/scheduler.json')

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

latslice = slice(90,-90)
lonslice = slice(0,360)

ivt_grid_area = xr.open_dataset('/Volumes/RAID5/Data/Reanalysis_Data/ERA5/grid_area_era5.nc')
            # .sel(latitude=latslice,longitude=lonslice)
ivt_grid_area = ivt_grid_area.rename({'longitude':'lon'})
ivt_grid_area = ivt_grid_area.rename({'latitude':'lat'})
ivt_grid_area = ivt_grid_area.reindex\
    (lat=list(reversed(ivt_grid_area.lat)))
ivt_land = xr.open_dataset('/Volumes/RAID5/Data/Reanalysis_Data/ERA5/land_sea_mask_erai.nc')
            # .sel(lat=slice(latslice.stop,latslice.start),lon=lonslice)


reload(obp)
smooth_scale = 1.5e6
angle_threshold = 45
shape_index = [0.625,1]
min_length = 20e3
min_area = 1e11
min_duration = 6
max_distance_per_tstep = 1000e3
shape_eccentricity = [0.0,1.0]
lat_mask = [-0,0]
lon_mask = [360,0]

properties = obp.object_properties2D(ivt_grid_area,ivt_land,min_length,min_area,\
                    smooth_scale,angle_threshold,min_duration,max_distance_per_tstep,\
                    shape_index,shape_eccentricity,\
                    lon_mask,lat_mask)

# Opening data
folder = '/Volumes/RAID5/Data/SCAFET/cyclonesf/'
outfolder = '/Volumes/RAID5/Data/SCAFET/cyclonest/'

ofiles = sorted(glob(folder+'cycObjects*.nc'))
pfiles = sorted(glob(folder+'cycProps*.nc'))
# print(len(pfiles))
# print(len(ofiles))

ar_objects = xr.open_mfdataset(ofiles,combine='by_coords')
ar_props = xr.open_mfdataset(pfiles,combine='by_coords')

# Tracking
properties.obj['Min_Duration']=8
properties.obj['Max_Distance']=500

# latlon = ['mx_lat1','mx_lon1']
latlon = ['wclat','wclon']
tracker = obt.Tracker(latlon,properties)
tracked = tracker.apply_tracking(ar_props,ar_objects)

exp='2019'
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
prfile = '{}cycTracked_props_ERA5_{}.nc'.format(outfolder,exp)
object_properties.to_netcdf('{}'.\
                format(prfile),encoding=encod)
print('Saved {}'.format(prfile))

objects = tracked[1].to_dataset(name='object')
encod = {}
for v in objects.data_vars:
        encod[v]={'zlib':True,'complevel':9,'shuffle':True}
for p in saved_props:
    objects.attrs[p] = saved_props[p]
obfile = '{}cycTracked_objs_ERA5_{}.nc'.format(outfolder,exp)
objects.to_netcdf('{}'.\
                    format(obfile),encoding=encod)
print('Saved {}'.format(obfile))
