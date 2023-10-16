# from dask.distributed import Client
# client = Client(scheduler_file='/Users/arjun/MPI/scheduler.json')
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

import ridge_detection as rd
import object_filtering as obf
import object_properties as obp

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

def check_filenames(uf,vf):
    ul=uf.split('/')[-1][-11:]
    vl=vf.split('/')[-1][-11:]
    return ul, ul == vl


folder = '/Volumes/RAID5/Data/Reanalysis_Data/ERA5'
ufiles = sorted(glob(folder+'/u10/*u10*.nc'))[224:228]
vfiles = sorted(glob(folder+'/v10/*v10*.nc'))[224:228]
# rvfiles = sorted(glob(folder+'*rv*.nc'))
outfolder1 = '/Volumes/RAID5/Data/'
outfolder2 = '/Volumes/RAID5/Data/'

latslice = slice(70,10)
lonslice = slice(250,340)
tslice = slice(0,5)

for i in tqdm(range(0,1)):
    ufile=ufiles[i]
    vfile=vfiles[i]
    ofile,flag=check_filenames(ufile,vfile)

    if flag:
        print('Opening File {}'.format(ofile))
        u=xr.open_dataset(ufile).isel(time=tslice)#.sel(latitude=latslice,longitude=lonslice)
        v=xr.open_dataset(vfile).isel(time=tslice)#.sel(latitude=latslice,longitude=lonslice)
        ws = np.sqrt(u['u10']**2+v['v10']**2)
        # Primary Field
        rv_unf = mpcalc.vorticity(u['u10'], v['v10'])
        rv = rv_unf.metpy.dequantify().to_dataset(name='rv')
        rv = rv.rename({'longitude':'lon'})
        rv = rv.rename({'latitude':'lat'})
        rv = rv.reindex\
            (lat=list(reversed(rv.lat)))
        cyc = rv*np.sign(rv.lat)

        stime = time.time()
        rdetect = rd.ridgeDetector(cyc)
        vor = rdetect.apply_smoother(cyc,properties)
        print('Finished smoothing in {} seconds'.format(time.time()-stime))

        stime = time.time()
        vor = vor.rename({'rv':'mag'})
        cyc = cyc.rename({'rv':'mag'})
        cyc_us = cyc.where((cyc.mag>0)).fillna(0)
        cyc_sm = vor.where((vor.mag>0)).fillna(0)
        # Detect Ridges
        ridges = rdetect.apply_ridge_detection(cyc_sm,properties)
        print('Finished ridge extraction in {} seconds'.format(time.time()-stime))

        ridges = ridges.drop('metpy_crs')
        encod = {}
        for v in ridges.data_vars:
            encod[v]={'zlib':True,'complevel':9,'shuffle':True}
        pfile = '{}cycRidges{}'.format(outfolder1,ofile)
        ridges.to_netcdf('{}'.\
                        format(pfile),encoding=encod)
        # Secondary Field
        del u
        del v
        ws = ws.rename({'longitude':'lon'})
        ws = ws.rename({'latitude':'lat'})
        ws = ws.reindex\
            (lat=list(reversed(ws.lat)))

        props_mag = xr.concat([ws.expand_dims('Channel'),\
                    cyc_us.mag.expand_dims('Channel')], dim='Channel')
        props_mag = props_mag.to_dataset(name='mag')
        del ws
        encod = {}
        for v in props_mag.data_vars:
            encod[v]={'zlib':True,'complevel':9,'shuffle':True}

        pfile = '{}cycProps{}'.format(outfolder1,ofile)
        props_mag.to_netcdf('{}'.\
                        format(pfile),encoding=encod)

        # Filtering
        stime = time.time()
        obfilter = obf.filterObjects(ridges)
        filtered = obfilter.apply_filter(ridges,\
                    props_mag,['max_intensity','mean_intensity'],
                    [10,2e-5],properties,'ridges')
        print('Finished Filtering in {} seconds'.format(time.time()-stime))

        obfile = '{}cycObjects{}'.format(outfolder2,ofile)
        prfile = '{}cycProps{}'.format(outfolder2,ofile)

        saved_props = properties.obj.copy()
        del saved_props['Min_Lon_Points']
        del saved_props['Min_Lat_Points']
        del saved_props['Map_Lons']
        del saved_props['Map_Lats']
        del saved_props['isglobal']

        encod = {}
        # object_properties = filtered[0].drop('metpy_crs').isel(index=slice(0,100))
        object_properties = filtered[0].isel(index=slice(0,100))

        for v in object_properties.data_vars:
            encod[v]={'zlib':True,'complevel':9,'shuffle':True}
        for p in saved_props:
            object_properties.attrs[p] = saved_props[p]

        object_properties.to_netcdf('{}'.\
                        format(prfile),encoding=encod)
        print('Saved {}'.format(prfile))

        encod = {}
        # objects = filtered[1].drop('metpy_crs').drop('e')
        objects = filtered[1].drop('e')
        for v in objects.data_vars:
            encod[v]={'zlib':True,'complevel':9,'shuffle':True}
        for p in saved_props:
            objects.attrs[p] = saved_props[p]

        objects.to_netcdf('{}'.\
                        format(obfile),encoding=encod)
        print('Saved {}'.format(obfile))
