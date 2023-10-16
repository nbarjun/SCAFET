# from dask.distributed import Client
# client = Client(scheduler_file="/Users/mac/ICCP/MPI/scheduler.json")

import xarray as xr
from glob import glob
import pandas as pd
import numpy as np
import scipy.integrate as integrate
import time
import platform
from datetime import datetime
import calendar

import object_properties as obp
import ridge_detection as rd
import object_filtering as obf
import object_tracking as obt
# client.upload_file('ridge_detection.py')
   
def calcIVT_simp(u,v,q,plev):
    ivtx=u*q
    #Integrate trapezoidal
    ivtx=(-1/9.81)*integrate.simpson(ivtx,x=plev,axis=0)
    ivty=v*q
    #Integrate trapezoidal
    ivty=(-1/9.81)*integrate.simpson(ivty,x=plev,axis=0)
    return (ivtx,ivty)   

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
min_area = 2e12
min_duration = 1
max_distance_per_tstep = 1000e3
shape_eccentricity = [0.75,1.]
lat_mask = [-20,20]
lon_mask = [360,0]

properties = obp.object_properties2D(grid_area,land_mask,min_length,min_area,\
                    smooth_scale,angle_threshold,min_duration,max_distance_per_tstep,\
                    shape_index,shape_eccentricity,\
                    lon_mask,lat_mask)

    
outfolder = '/Volumes/RAID5/Data/Atmosp_Rivers/ESGF/CESM2/Output/'
expname = 'CESM2.ssp585'

var_names = ['ua','va','hus','pr']
datafolder  = '/Volumes/RAID5/Data/Atmosp_Rivers/ESGF/CESM2'
files = {}
for v in var_names:
    if v == 'pr':
        files[v] = sorted(glob('{}/{}/{}_day_CESM2_ssp585_*.nc'.format(datafolder,v,v)))[-3:]
    else:
        files[v] = sorted(glob('{}/{}/{}_day_CESM2_ssp585_*.nc'.format(datafolder,v,v)))[-3:]
        
years = np.arange(2075,2100,5)
for yr in years:
    if yr==2000:
        year = '{}0101-{}1231'.format(yr,yr+5)
        tslice = slice('{}-01-01'.format(yr),'{}-12-31'.format(yr+5))
    else:
        year = '{}0101-{}1231'.format(yr,yr+4)
        tslice = slice('{}-01-01'.format(yr),'{}-12-31'.format(yr+4))
   
    udata = xr.open_mfdataset(files['ua'])['ua'].sel(time=tslice)\
                    .chunk({'time':365,'plev':-1,'lat':-1,'lon':-1})
    vdata = xr.open_mfdataset(files['va'])['va'].sel(time=tslice)\
                        .chunk({'time':365,'plev':-1,'lat':-1,'lon':-1})
    qdata = xr.open_mfdataset(files['hus'])['hus'].sel(time=tslice)\
                        .chunk({'time':365,'plev':-1,'lat':-1,'lon':-1})
    prdata = xr.open_mfdataset(files['pr'])['pr'].sel(time=tslice)\
                        .chunk({'time':365,'lat':-1,'lon':-1})
    
    ivtx,ivty = xr.apply_ufunc(calcIVT_simp,udata.fillna(0),vdata.fillna(0),\
                     qdata.fillna(0),qdata['plev'],vectorize=True,dask='parallelized',\
                     input_core_dims=[['plev','lat','lon'],['plev','lat','lon'],['plev','lat','lon'],['plev']],\
                     output_core_dims=[['lat','lon'],['lat','lon']],\
                     output_dtypes =[ 'float32', 'float32'])
    ivtdata = ivtx.to_dataset(name='u')
    ivtdata['v'] = ivty
    print('Finshed IVT calculation for {}'.format(year))
    
    rdetect = rd.ridgeDetector(ivtdata)
    ivtmag = rd.calc_magnitude(ivtdata.u, ivtdata.v).to_dataset(name='mag')
    
    
    # Apply Smoother
    start = time.time()
    ivts = rdetect.apply_smoother(ivtdata,properties)
    print('Finished smoothing in {} seconds'.format(time.time() - start))
    
    
    # Detect the ridges
    start = time.time()
    ridges = rdetect.apply_ridge_detection(ivts,properties)
    print('Finished ridge extraction in {} seconds'.format(time.time() - start))
    
    
    # Apply object filter and get object properties
    start = time.time()
    prect = (prdata*86400).expand_dims('Channel').to_dataset(name='mag')
    # Combine the needed properties
    props_mag = xr.concat([ivtmag.expand_dims('Channel'),prect],dim='Channel')
    obfilter = obf.filterObjects(ridges)
    filtered = obfilter.apply_filter(ridges.astype(int),\
                    props_mag.chunk({'Channel':-1}),['mean_intensity','mean_intensity'],
                    [0,1],properties,'ridges')
    print('Finished object filtering in {} seconds'.format(time.time() - start))


    # Save Files
    objects_outfile='{}arObjects.{}.{}.nc'.format(outfolder,expname,\
                                      year)
    props_outfile='{}arProps.{}.{}.nc'.format(outfolder,expname,\
                                          year)    

    saved_props = properties.obj.copy()
    del saved_props['Min_Lon_Points']
    del saved_props['Min_Lat_Points']
    del saved_props['Map_Lons']
    del saved_props['Map_Lats']
    del saved_props['isglobal']

    encod = {}
    try:
        object_properties = filtered[0].drop('metpy_crs').isel(index=slice(0,30))
    except:
        object_properties = filtered[0].isel(index=slice(0,30))

    for v in object_properties.data_vars:
        encod[v]={'zlib':True,'complevel':9,'shuffle':True}
    for p in saved_props:
        object_properties.attrs[p] = saved_props[p]
    object_properties.attrs['Created By']='Arjun Babu Nellikkattil (arjunbabun@pusan.ac.kr)'
    object_properties.attrs['Detected Using']='Scalable Feature Extraction and Tracking SCAFET in {}'\
                                                            .format(platform.platform())
    object_properties.attrs['Date Created']=datetime.now().strftime("%Y/%m/%d, %H:%M:%S")


    object_properties.to_netcdf('{}'.format(props_outfile),encoding=encod)
    print('Saved {}'.format(props_outfile))

    encod = {}
    try:
        objects = filtered[1].drop('e').drop('metpy_crs')
    except:
        objects = filtered[1]

    for v in objects.data_vars:
        encod[v]={'zlib':True,'complevel':9,'shuffle':True}
    for p in saved_props:
        objects.attrs[p] = saved_props[p]

    objects.attrs['Created By']='Arjun Babu Nellikkattil (arjunbabun@pusan.ac.kr)'
    objects.attrs['Detected Using']='Scalable Feature Extraction and Tracking SCAFET in {}'\
                                                            .format(platform.platform())
    objects.attrs['Date Created']=datetime.now().strftime("%Y/%m/%d, %H:%M:%S")

    objects.to_netcdf('{}'.format(objects_outfile),encoding=encod)
    print('Saved {}'.format(objects_outfile))
