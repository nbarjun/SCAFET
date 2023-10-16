Iimport numpy as np
import xarray as xr
import pandas as pd
import time
from glob import glob
import metpy.calc as mpcalc
from metpy.units import units
from importlib import reload
import metpy.xarray as mxr
from tqdm import tqdm
import pop_tools
from scipy.ndimage import gaussian_filter
# Modules for feature extraction
import sys
scafex_folder = '/home/arjun/SCAFET/scafet/'
sys.path.append(scafex_folder)
import object_properties as obp
import ridge_detection as rd
import object_filtering as obf
import object_tracking as obt

tslice=slice(0,2)
latslice = slice(0,60)
lonslice = slice(110,180)
#-------------Function to Calcualte Gradient-----------------#
def calc_gradient(data,grid):
    for v in ['ULAT','ULONG','DXT','DYT',\
            'DXU','DYU','TAREA','UAREA','REGION_MASK']:
        data[v] = grid[v] 

    metrics = {
    ("X",): ["DXT"],  # X distances
    ("Y",): ["DYT"],  # Y distances
    ("X", "Y"): ["TAREA"],
    }

    # here we get the xgcm compatible dataset
    gridxgcm, dsxgcm = pop_tools.to_xgcm_grid_dataset(
        data, periodic=False,metrics=metrics,
        boundary={"X": "extend", "Y": "extend", "Z": "extend"})
    
    for coord in ["nlat", "nlon"]:
        if coord in dsxgcm.coords:
            dsxgcm = dsxgcm.drop_vars(coord)
            
    dt_dx = gridxgcm.diff(dsxgcm['mag'], 'X')
    dt_dy = gridxgcm.diff(dsxgcm['mag'], 'Y')
    return ((gridxgcm.interp(dt_dx, 'X')/dsxgcm['DXT'])**2 +\
            ((gridxgcm.interp(dt_dy,'Y')**2)/dsxgcm['DYT'])) ** 0.5

#------------Grid Information---------------#
hires_grid = pop_tools.get_grid('POP_tx0.1v3')

gridinfo = hires_grid.where((hires_grid.TLAT>latslice.start)&(hires_grid.TLAT<latslice.stop)\
              &(hires_grid.TLONG>lonslice.start)&(hires_grid.TLONG<lonslice.stop)\
              &(hires_grid.TLONG>=0)&(hires_grid.TLAT>=0))\
            .dropna('nlat',how='all').dropna('nlon',how='all')

#---------------Define Propertiesof Front-----------#
smooth_scale = 30e3
shape_index = [0.375,1]
angle_threshold= 45
min_length = 200e3
min_area = 1e9
min_duration = 1
max_distance_per_tstep = 1000e3
shape_eccentricity = [0.5,1.0]
lat_mask = [-0,0]
lon_mask = [360,0]

properties = obp.object_properties_tgrid(gridinfo,min_length,min_area,\
                    smooth_scale,angle_threshold,min_duration,max_distance_per_tstep,\
                    shape_index,shape_eccentricity,lon_mask,lat_mask)

#------------READ IVT Data-----------------#
folder = '/proj/shared_data/cesm_hires/BC5.ne120_t12.pop62.lagrangian.off/ocn/day/SST/'
flista = sorted(glob(folder+'SST.*.nc'))
flistb = sorted(glob(folder+'SST.*0.1x0.1.nc'))
flistc = sorted(glob(folder+'SST.*1x1.nc'))
flist = sorted(list(set(flista)-set(flistb)-set(flistc)))[130:140]

outfolder='/proj/arjun/SCAFET/sstFront_core/Output/'
for i in tqdm(range(len(flist))):
    print('Opening File: {}'.format(flist[i].split('/')[-1]))
    sst = xr.open_dataset(flist[i]).isel(time=tslice)
    sst['TLAT'] = hires_grid['TLAT']
    sst['TLONG'] = hires_grid['TLONG']
    
    sst_us = sst.where((sst.TLAT>latslice.start)&(sst.TLAT<latslice.stop)\
              &(sst.TLONG>lonslice.start)&(sst.TLONG<lonslice.stop)\
              &(sst.TLONG>=0)&(sst.TLAT>=0)).dropna('time',how='all')\
            .dropna('nlat',how='all').dropna('nlon',how='all').where(gridinfo.KMT!=0)
    
    #------------Smoothing SST data-----------#
    stime = time.time()
    rdetect = rd.ridgeDetector(sst_us)
    sst_sm = rdetect.apply_smoother_tgrid(sst_us['SST'],properties)
    print('Finished Smoothing in {} seconds'.format(time.time()-stime))

    #------------Gradient Calculation-----------#
    sst_grad = calc_gradient(sst_sm.to_dataset(name='mag'),gridinfo)
    
    #------------Ridge Extraction-----------#
    stime = time.time()
    ridges = rdetect.apply_ridge_detection_tgrid(\
            sst_grad.to_dataset(name='mag'),properties)
    print('Finished Ridge Extraction in {} seconds'.format(time.time()-stime))

    #-----------Filtering----------------#
    stime = time.time()
    props_mag = sst_grad.to_dataset(name='mag')

    obfilter = obf.filterObjects(ridges)
    filtered = obfilter.apply_filter_tgrid(ridges,\
                    props_mag,['mean_intensity'],
                    [1e-4],properties,'cores')
    print('Finished Filtering in {} seconds'.format(time.time()-stime))
    #----------Saving Ridge File-----------#
    ridges = ridges
    encod = {}
    for v in ridges.data_vars:
        encod[v]={'zlib':True,'complevel':9,'shuffle':True}
    pfile = '{}sstRidges_{}'.format(outfolder,flist[i].split('/')[-1][8:])
    ridges.to_netcdf('{}'.\
                    format(pfile),encoding=encod)

    #----------Saving Magnitude File-----------#
    encod = {}
    for v in props_mag.data_vars:
        encod[v]={'zlib':True,'complevel':9,'shuffle':True}
    
    mfile = '{}sstGrad_mag_{}'.format(outfolder,flist[i].split('/')[-1][8:])
    props_mag.to_netcdf('{}'.\
                    format(mfile),encoding=encod)
    
    saved_props = properties.obj.copy()
    del saved_props['Map_Lons']
    del saved_props['Map_Lats']
    del saved_props['isglobal']

    #------------Saving Object Properties------------#
    encod = {}
    object_properties = filtered[0].isel(index=slice(0,100))
    for v in object_properties.data_vars:
        encod[v]={'zlib':True,'complevel':9,'shuffle':True}
    for p in saved_props:
        object_properties.attrs[p] = saved_props[p]
    
    pfile = '{}sstFront_props_{}'.format(outfolder,flist[i].split('/')[-1][8:])
    object_properties.to_netcdf('{}'.\
                    format(pfile),encoding=encod)
    print('Saved {}'.format(pfile))
    
    #------------Saving Object Properties------------#
    encod = {}
    objects = filtered[1]
    for v in objects.data_vars:
        encod[v]={'zlib':True,'complevel':9,'shuffle':True}
    for p in saved_props:
        objects.attrs[p] = saved_props[p]
        
    sfile = '{}sstFront_objects_{}'.format(outfolder,flist[i].split('/')[-1][8:])
    objects.to_netcdf('{}'.\
                    format(sfile),encoding=encod)
    print('Saved {}'.format(sfile))