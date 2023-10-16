# Modules required for computation
import numpy as np
import xarray as xr
import pandas as pd
import time
from glob import glob
from importlib import reload
import sys
import os
import dask
from tqdm import tqdm
from datetime import datetime, timedelta, date
import platform

# Modules for feature extraction
scafet_folder = '/home/arjun/Atmosp_Rivers/UHR/scafetfiles/'
sys.path.append(scafet_folder)
import object_properties as obp
import ridge_detection as rd
import object_filtering as obf

def check_timestamp(fivt,fppc,fppl):
    t1=fivt.split('/')[-1].split('.')[-2]
    t2=fppc.split('/')[-1].split('.')[-2]
    t3=fppl.split('/')[-1].split('.')[-2]
    return t1,(t1==t2)&(t2==t3)

def check_timestamp_none(fivt):
    t1=fivt.split('/')[-1].split('.')[-2]
    return t1, True

var_names = ['PRECC','PRECL']#,'T','OMEGA']#,'QRL','RELHUM','CLOUD']
dataFolder= '/proj/shared_data/cesm_hires/'
experiments= ['BC5.ne120_t12.pop62.lagrangian.off','BC5.ne120_t12.pop62.lagrangian.off.2xCO2',\
             'BC5.ne120_t12.pop62.lagrangian.off.4xCO2']
exnames = ['PD','2xCO2','4xCO2']

fnames={}
for exp,exn in zip(experiments,exnames ):
    fnames[exn]={}
    for vn in var_names:
        flista=sorted(glob(dataFolder+exp+'/atm/day/'+vn+'/'+vn+'.day.*.nc'))
        flistb=sorted(glob(dataFolder+exp+'/atm/day/'+vn+'/'+vn+'.day.*.1x1.nc'))
        flistb = sorted(list(set(flista)-set(flistb)))
        if exn=='PD':
            fnames[exn][vn]=sorted(flistb)[-5:]
            print(fnames[exn][vn])
        elif exn=='2xCO2':
            fnames[exn][vn]=sorted(flistb)[95:100]
            print(fnames[exn][vn])
        elif exn=='4xCO2':
            fnames[exn][vn]=sorted(flistb)[95:100]
            print(fnames[exn][vn])
        del flistb

ivtfolder = '/proj/arjun/CESM/Misc_Data/IVT/'
expnames = ['PD','2xCO2','4xCO2']
exnames = ['pd','2c','4c']
ivtfiles={}
for exp,exn in zip(expnames,exnames):
    ivtfiles[exp]=sorted(glob('{}{}/ivt.{}*.nc'.format(ivtfolder,exp,exn)))[-5:]

ivt_grid_area = xr.open_dataset('/proj/arjun/SCAFET/Grids/grid_area_cesm.nc')
ivt_land = xr.open_dataset('/proj/arjun/SCAFET/Grids/land_mask_PD.nc')

smooth_scale = 2e6
angle_threshold = 45
shape_index = [0.375,1]
min_length = 2000e3
min_area = 2e12
shape_eccentricity = [0.75,1.]
lat_mask = [-20,20]
lon_mask = [360,0]
min_duration = 1
max_distance_per_tstep = 1000e3

# ivt_land = ivt_land.transpose()
properties = obp.object_properties2D(ivt_grid_area,ivt_land,min_length,min_area,\
                    smooth_scale,angle_threshold,min_duration,max_distance_per_tstep,\
                    shape_index,shape_eccentricity,\
                    lon_mask,lat_mask)

outfolder='/proj/arjun/CESM/Atmosp_Rivers/SCAFET/Sensitivity/P5/'
for exps in list(ivtfiles.keys()):
    if not os.path.exists('{}{}'.format(outfolder,exps)):
        os.makedirs('{}{}'.format(outfolder,exps))
    if not os.path.exists('{}{}/Output'.format(outfolder,exps)):
        os.makedirs('{}{}/Output'.format(outfolder,exps))
        
    for f in tqdm(range(len(ivtfiles[exps]))):
        ifile,flag=check_timestamp_none(ivtfiles[exps][f])
        if flag:
            # Reading IVT data
            ivtdata = xr.open_dataset(ivtfiles[exps][f])#.isel(time=tslice)
            # Reading PRECT data
            precc = xr.open_dataset(fnames[exps]['PRECC'][f])['PRECC']#.isel(time=tslice)
            precl = xr.open_dataset(fnames[exps]['PRECL'][f])['PRECL']#.isel(time=tslice)

            prect = precc+precl
            prect = (prect.where(prect>0).fillna(0)*86400*1000).to_dataset(name='mag')

            # Define Ridge Detector
            print('Reading ',ivtfiles[exps][f])
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
            # Secondary field
            props_mag = xr.concat([ivtmag['mag'].expand_dims('Channel'),\
                    prect['mag'].expand_dims('Channel')],dim='Channel')
            # props_mag = xr.concat([ivtmag['mag'].expand_dims('Channel')],dim='Channel')
            props_mag = props_mag.to_dataset(name='mag')

            # Filtering
            obfilter = obf.filterObjects(ridges)
            filtered = obfilter.apply_filter(ridges,\
                        props_mag,['mean_intensity','mean_intensity'],
                        [0,5],properties,'ridges')
            # filtered = obfilter.apply_filter(ridges,\
            #             props_mag,['mean_intensity'],
            #             [0],properties,'ridges')
            print('Finished object filtering in {} seconds'.format(time.time() - start))


            objects_outfile='{}{}/Output/arObjects.{}.{}.nc'.format(outfolder,exps,exps,\
                                              ivtfiles[exps][f].split('/')[-1].split('.')[-2])
            props_outfile='{}{}/Output/arProps.{}.{}.nc'.format(outfolder,exps,exps,\
                                              ivtfiles[exps][f].split('/')[-1].split('.')[-2])

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
                continue

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
        else:
            print(ivtfiles[exps][f])
            print(fnames[exps]['PRECC'][f])
            print(fnames[exps]['PRECL'][f])
            raise ValueError('The corresponding Files doesnt match')