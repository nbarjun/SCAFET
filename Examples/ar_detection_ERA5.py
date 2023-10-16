# Modules required for computation
import numpy as np
import xarray as xr
import pandas as pd
import time
from glob import glob
from importlib import reload
import sys
import dask
from tqdm import tqdm

# Modules for feature extraction
scafex_folder = '/home/arjun/Atmosp_Rivers/SCAFEX/scafet/'
sys.path.append(scafex_folder)
import object_properties as obp
import ridge_detection as rd
import object_filtering as obf

# Import the dask library for parallisation
from dask_mpi import initialize
initialize()
from dask.distributed import Client
client = Client()

#--------------Grid Area and Land Mask----------------------#
ivt_grid_area = xr.open_dataset('/proj/arjun/ERA25/grid_area_era5.nc')#.sel(latitude=latslice)
ivt_grid_area = ivt_grid_area.rename({'longitude':'lon'})
ivt_grid_area = ivt_grid_area.rename({'latitude':'lat'})
ivt_grid_area = ivt_grid_area.reindex\
    (lat=list(reversed(ivt_grid_area.lat)))
ivt_land = xr.open_dataset('/proj/arjun/ERA25/land_sea_mask_erai.nc')#.sel(lat=latslice1)
#------------------------------------------------------------#

smooth_scale = 2e6
angle_threshold = 45
shape_index = [0.375,1]
min_length = 2000e3
min_area = 2e12
shape_eccentricity = [0.75,1.]
lat_mask = [-20,20]
lon_mask = [360,0]
properties = obp.object_properties2D(ivt_grid_area,ivt_land,min_length,min_area,\
                        smooth_scale,angle_threshold,shape_index,shape_eccentricity,\
                        lon_mask,lat_mask)
#----------------------------------------------------------------------#
#---------------Get Precipitation Filenames----------------------------#
#----------------------------------------------------------------------#
pptfiles =sorted(glob('/proj/arjun/ERA25/ppt_daily/*.nc'))[:240]
# pptfiles = pptfiles[-1:]
#----------------------------------------------------------------------#       
tslice = slice(10,15)
ivt_folder = '/proj/arjun/ERA25/ivt_daily/'
ivtfiles=sorted(glob(ivt_folder+'ivt_*.nc'))[:240]
# ivtfiles=ivtfiles[-1:]
# tslice = slice('2019-12-21','2019-12-31')
outfolder = '/proj/arjun/CESM/Atmosp_Rivers/SCAFEX_V1/ERA5/'

for f in tqdm(range(len(ivtfiles))):
    # Reading IVT data
    ivtdata = xr.open_dataset(ivtfiles[f])#.sel(time=tslice)
    ivtdata = ivtdata.rename({'longitude':'lon'})
    ivtdata = ivtdata.rename({'latitude':'lat'})
    ivtdata = ivtdata.reindex\
        (lat=list(reversed(ivtdata.lat))).drop('time_bnds')
    # Reading PRECT data
    prect = xr.open_dataset(pptfiles[f])['mtpr']#.sel(time=tslice)
    prect = prect.where(prect>0).fillna(0)*86400

    # Reading PRECT Data-----------------#
    prect = prect.rename({'longitude':'lon'})
    prect = prect.rename({'latitude':'lat'})
    prect = prect.reindex\
        (lat=list(reversed(prect.lat))).to_dataset(name='mag')
    
    # Define Ridge Detector
    print('Reading ',ivtfiles[f])
    rdetect = rd.ridgeDetector(ivtdata)
    ivtmag = rd.calc_magnitude(ivtdata.ivtx, ivtdata.ivty).to_dataset(name='mag')
    # Apply Smoother
    start = time.time()
    ivts = rdetect.apply_smoother(ivtdata,properties)
    print('Finished smoothing in {} seconds'.format(time.time() - start))
    # For vector data
    ivts = ivts.rename({'ivtx':'u'})
    ivts = ivts.rename({'ivty':'v'})
    
    # Detect the ridges
    start = time.time()
    ridges = rdetect.apply_ridge_detection(ivts,properties)
    print('Finished ridge extraction in {} seconds'.format(time.time() - start))
    
    # Apply object filter and get object properties
    start = time.time()
    # Combine the needed properties
    objProps = xr.concat([ivtmag.expand_dims('Channel'),\
            prect.expand_dims('Channel')],dim='Channel')
    obfilter = obf.filterObjects(ridges)
    filtered = obfilter.apply_filter(ridges,objProps,properties,'ridges')
    print('Finished object filtering in {} seconds'.format(time.time() - start))

    #----------Save the netcdf files------#
    ofile = '{}ar_objs{}.nc'.format(outfolder,\
                    ivtfiles[f].split('/')[-1][3:-3])
    pfile = '{}ar_props{}.nc'.format(outfolder,\
                    ivtfiles[f].split('/')[-1][3:-3])
    
    saved_props = properties.obj.copy()
    del saved_props['Min_Lon_Points'] 
    del saved_props['Min_Lat_Points'] 
    del saved_props['Map_Lons'] 
    del saved_props['Map_Lats'] 
    del saved_props['isglobal'] 

    encod = {}
    object_properties = filtered[0].isel(index=slice(0,30))
    for v in object_properties.data_vars:
        encod[v]={'zlib':True,'complevel':9,'shuffle':True}
    for p in saved_props:
        object_properties.attrs[p] = saved_props[p]
    
    object_properties.to_netcdf('{}'.\
                    format(pfile),encoding=encod)
    print('Saved {}'.format(pfile))

    encod = {}
    objects = filtered[1].drop('crs').drop('e')
    for v in objects.data_vars:
        encod[v]={'zlib':True,'complevel':9,'shuffle':True}
    for p in saved_props:
        objects.attrs[p] = saved_props[p]
        
    objects.to_netcdf('{}'.\
                    format(ofile),encoding=encod)
    print('Saved {}'.format(ofile))