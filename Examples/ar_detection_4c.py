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

ivt_grid_area = xr.open_dataset('/proj/arjun/CESM/Atmosp_Rivers/GuanWaliserV1/river2/grid_area_cesm.nc')
ivt_land = xr.open_dataset('/proj/arjun/CESM/Atmosp_Rivers/GuanWaliserV1/river2/land_mask_PD.nc')

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
var_names = ['PRECC','PRECL']
dataFolder= '/proj/shared_data/cesm_hires/'
experiments= ['BC5.ne120_t12.pop62.lagrangian.off','BC5.ne120_t12.pop62.lagrangian.off.2xCO2',\
             'BC5.ne120_t12.pop62.lagrangian.off.4xCO2']
exnames = ['PD','2xCO2','4xCO2']
fnames={}
for exp,exn in zip(experiments[2:3],exnames[2:3]):
    fnames[exn]={}
    for vn in var_names:
        flista=sorted(glob(dataFolder+exp+'/atm/day/'+vn+'/'+vn+'.day.*.nc'))
        flistb=sorted(glob(dataFolder+exp+'/atm/day/'+vn+'/'+vn+'.day.*.1x1.nc'))
        flistb = sorted(list(set(flista)-set(flistb)))
        if exn=='PD':
            fnames[exn][vn]=sorted(flistb)[120:140]
        elif exn=='2xCO2':
            fnames[exn][vn]=sorted(flistb)[80:100]
        elif exn=='4xCO2':
            fnames[exn][vn]=sorted(flistb)[80:100]
        del flistb
        
#----------------------------------------------------------------------#       
# tslice = slice(10,15)
ivt_folder = '/proj/arjun/CESM/Misc_Data/IVT/'
ivtfiles = sorted(glob(ivt_folder+'*.4c.*.nc'))[:20]
pptfiles = fnames['4xCO2']

outfolder = '/proj/arjun/CESM/Atmosp_Rivers/SCAFEX_V1/4xCO2/'
for f in tqdm(range(len(ivtfiles))):
    # Reading IVT data
    ivtdata=xr.open_dataset(ivtfiles[f])#.sel(time=tslice)
    # Reading PRECT data
    precc = xr.open_dataset(pptfiles['PRECC'][f])['PRECC']#.sel(time=tslice)
    precl = xr.open_dataset(pptfiles['PRECL'][f])['PRECL']#.sel(time=tslice)
    prect = precc.where(precc>0).fillna(0)+precl.where(precl>0).fillna(0)
    prect = (prect*86400*1000).expand_dims('Channel').to_dataset(name='mag')
    
    print('Reading ',ivtfiles[f])
    rdetect = rd.ridgeDetector(ivtdata)
    ivtmag = rd.calc_magnitude(ivtdata.ivtx, ivtdata.ivty).to_dataset(name='mag')

    start = time.time()
    ivts = rdetect.apply_smoother(ivtdata,properties)
    print('Finished smoothing in {} seconds'.format(time.time() - start))
    # For vector data
    ivts = ivts.rename({'ivtx':'u'})
    ivts = ivts.rename({'ivty':'v'})
    start = time.time()
    ridges = rdetect.apply_ridge_detection(ivts,properties)
    print('Finished ridge extraction in {} seconds'.format(time.time() - start))
    
    # Apply object filter and get object properties
    start = time.time()
    # Combine the needed properties
    objProps = xr.concat([ivtmag.expand_dims('Channel'),prect],dim='Channel')
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