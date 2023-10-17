import numpy as np
import xarray as xr
import pandas as pd
from skimage.exposure import rescale_intensity
from geopy.distance import geodesic
from skimage.measure import label, regionprops_table, regionprops
from skimage.segmentation import clear_border
import pop_tools

class filterObjects:    
    def apply_filter(self,ridges,data,minthres,values,properties,option):
        properties.minvals = {}
        if (self.ndims == 2):
            for n in range(len(values)):
                properties.minvals['{}-{}'.format(minthres[n],str(n+1))]=values[n]
            return filter_objects2D(ridges,data,properties,option)
#             return properties
        if (self.ndims == 3):
            for n in range(len(values)):
                properties.minvals['{}-{}'.format(minthres[n],str(n+1))]=values[n]              
            return filter_objects3D(ridges,data,properties,option)

    def apply_filter_tgrid(self,ridges,data,minthres,values,properties,option):
        properties.minvals = {}
        if (self.ndims == 2):
            for n in range(len(values)):
                properties.minvals['{}-{}'.format(minthres[n],str(n+1))]=values[n]
            return filter_objects_tgrid2D(ridges,data,properties,option)
#             return properties
        if (self.ndims == 3):
            for n in range(len(values)):
                properties.minvals['{}-{}'.format(minthres[n],str(n+1))]=values[n]              
            return filter_objects_tgrid3D(ridges,data,properties,option)

    def apply_filter_ugrid(self,ridges,data,minthres,values,properties,option):
        properties.minvals = {}
        if (self.ndims == 2):
            for n in range(len(values)):
                properties.minvals['{}-{}'.format(minthres[n],str(n+1))]=values[n]
            return filter_objects_ugrid3D(ridges,data,properties,option)

        if (self.ndims == 3):
            for n in range(len(values)):
                properties.minvals['{}-{}'.format(minthres[n],str(n+1))]=values[n]              
            return filter_objects_ugrid3D(ridges,data,properties,option)

    def __init__(self,data,ndims=None):
        self.ndims = len(list(data.dims))-1 if ndims==None else ndims        


def filter_objects_tgrid2D(ridge,data,props,option):
    # Creating the compatible grid and data
    for v in ['ULAT','ULONG','DXT','DYT',\
            'DXU','DYU','TAREA','UAREA','REGION_MASK']:
        data[v] = props.grid[v] 

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
    del gridxgcm
    
    # Selecting whether to use Cores or Ridges
    if option == 'cores':
        ridges = ridge.cores
    elif option == 'ridges':
        ridges = ridge.ridges
    # Broadcasting the grid area
    grid_area = (xr.broadcast(ridges,dsxgcm['TAREA'])[1])*1e-4

    # Wrap the array if we are using global data
    if props.obj['isglobal']:
        magnitude = wrapped(dsxgcm['mag'])
        ridges = wrapped(ridges.fillna(0))
        grid_area = wrapped(grid_area)
    else:
        magnitude = dsxgcm['mag'].copy()
        ridges = ridges.fillna(0)
    
    # Remove small objects and enusure object continuity across borders
    ridges = xremove_objects_tgrid(ridges,props)
    dims = list(data.mag.dims)

    if 'Channel' in dims:
        field_props = magnitude.copy()
    else:
        field_props = magnitude.expand_dims('Channel')

    grid_area = grid_area.expand_dims('Channel').assign_coords({'Channel':[0]})
    field_props = field_props.assign_coords({'Channel':range(1,len(field_props['Channel'])+1)})

    field_props = xr.concat([grid_area,field_props],dim='Channel')
    field_props = field_props.transpose(...,'Channel')

    objs= regionprops_table(ridges.isel(time=0).astype(int).values,\
                properties=['label'])
    n= int(len(objs['label'])-len(objs['label'])*.1)
    ridge_objs = xr.apply_ufunc(filter_tgrid,ridges.astype(int),field_props,n,props,\
        input_core_dims=[['nlat_t','nlon_t'],['nlat_t','nlon_t','Channel'],[],[]],\
        output_core_dims=[['index','props'],['nlat_t','nlon_t'],['props']],vectorize=True,\
                            dask='parallelized')
    vlist = ridge_objs[2].isel(time=0).values

    ridge_dset = ridge_objs[0].sel(props=0).to_dataset(name='label')
    for c in range(len(vlist)-1):
        ridge_dset[vlist[c+1]] = ridge_objs[0].isel(props=c+1)
    ridge_dset = ridge_dset.assign_coords(index=np.arange(1,n+1,1))

    if props.obj['isglobal']:
        filtered_objs = unwrap(ridge_objs[1].fillna(0)).to_dataset(name='object')
    else:
        filtered_objs = ridge_objs[1].fillna(0).to_dataset(name='object')

    filtered_objs['core'] = filtered_objs['object']*ridge['cores']

    object_properties = ridge_dset.dropna(dim='index',how='all')
    return ridge_dset, filtered_objs

def filter_tgrid(ridge_objects,magnitudes,n,props):
    objs= regionprops_table(ridge_objects.astype(int),\
                intensity_image=magnitudes, properties=['label','bbox','centroid',\
              'mean_intensity','min_intensity','max_intensity',\
              'perimeter','eccentricity','bbox_area','area'],\
                extra_properties=(max_loc,))
    objs = pd.DataFrame(objs)
    #Get and remove the duplicate objects
    removables_duplicates = objs[objs.duplicated(subset=\
                list(objs.columns[7:]), keep='first')]
    objs_noduplicates = pd.concat([objs,removables_duplicates])\
                        .drop_duplicates(keep=False)
    removables_small = objs_noduplicates.where(objs_noduplicates['area']<props.obj['Min_Area_Points'])
    objs_noduplicates = pd.concat([objs_noduplicates,removables_small])\
                        .drop_duplicates(keep=False)
#     pd.set_option('display.max_rows', None)
#     print(objs_noduplicates)
    #Map geographic Information to labelled regions
    mapped_objects = map_coordinates_tgrid(objs_noduplicates,props)
#     print(mapped_objects.iloc[0])
#     pd.set_option('display.max_rows', None)
#     pd.set_option('display.max_columns', None)
#     print(mapped_objects)
    
#     Filter for
#     1. Minimum Area
#     2. Eccentricity
#     3. Equator Objects
    removables_shape = mapped_objects.where((mapped_objects['eccentricity']<props.obj['Eccentricity'][0])|\
                    (mapped_objects['eccentricity']>props.obj['Eccentricity'][1])|\
                    (mapped_objects['Obj_Area']<props.obj['Min_Area'])|\
                    ((mapped_objects['wclat']>props.obj['Lat_Mask'][0])&(mapped_objects['wclat']<props.obj['Lat_Mask'][1])&\
                    ((np.minimum(mapped_objects['lat-0'],mapped_objects['lat-1'])>props.obj['Lat_Mask'][0])|\
                    (np.maximum(mapped_objects['lat-0'],mapped_objects['lat-1'])<props.obj['Lat_Mask'][1])))|\
                    ((mapped_objects['wclon']>props.obj['Lon_Mask'][0])&(mapped_objects['wclon']<props.obj['Lon_Mask'][1])))
    

    removables_minthresh = []
    for v in props.minvals:
        removables_minthresh.append(mapped_objects.where(mapped_objects[v]<props.minvals[v]))

    objs_shape_area = pd.concat([mapped_objects, removables_shape, *removables_minthresh])\
                        .drop_duplicates(keep=False).dropna()
    
    #Find the length of objects
    length_mapped = map_length(objs_shape_area,props)
    removables_length = length_mapped.where(length_mapped.length<props.obj['Min_Length'])

    #Remove short objects
    objs_long = pd.concat([length_mapped, removables_length])\
                    .drop_duplicates(keep=False)
    
#     print(pd.concat(removables_minthresh)['label'])
    #Combine all removable object labels
    all_removables = np.unique(np.concatenate((removables_duplicates.label.values,\
                removables_small.label.values, removables_shape.label.values,\
                removables_length.label.values,\
                pd.concat(removables_minthresh)['label'].values),axis=0))
    
    #All the objects
    filtered_objects = objs_long.dropna(subset=objs_long.columns)
    filtered_objects = filtered_objects.reset_index(drop=True)
    
    #Remove the filtered objects
    filtered_objs =  remove_objects(ridge_objects.astype(int),all_removables)

#     relabelled_objects = label_rivs_tgrid(np.nan_to_num(filtered_objs))
    
#     print(len(filtered_objects.index.values))
#     print(len(filtered_objects['label'].values))
    
#     print((filtered_objects.index.values+1))
#     print((filtered_objects['label'].values))

    #Relabel object masks to match properties
    relabelled_objects =  remove_and_replace_labels(\
            np.nan_to_num(filtered_objs).astype(int),\
            filtered_objects['label'].values,filtered_objects.index.values+1)
#     print(filtered_objects['label'].values,filtered_objects.index.values)
    
    nandframe = pd.DataFrame(np.empty((n-len(filtered_objects),len(objs_long.columns)))*np.nan,\
                columns=filtered_objects.columns,index=np.arange(len(filtered_objects),n,1))
#     print(nandframe)
    filled_filtered = pd.concat([filtered_objects,nandframe])

    return filled_filtered, relabelled_objects, xr.DataArray(filled_filtered.columns)

def filter_objects2D(ridge,data,properties,option):
    # Selecting whether to use Cores or Ridges
    if option == 'core':
        ridges = ridge.core
    elif option == 'ridges':
        ridges = ridge.ridges
    # Broadcasting the grid area
    grid_area = xr.broadcast(ridges,properties.grid['grid_area']['cell_area'])[1]
    # Wrap the array if we are using global data
    if properties.obj['isglobal']:
        magnitude = wrapped(data.mag)
        ridges = wrapped(ridges.fillna(0))
        grid_area = wrapped(grid_area)
    else:
        magnitude = data.mag.copy()
        ridges = ridges.fillna(0)
    # Remove small objects and enusure object continuity across borders
    ridges = xremove_objects(ridges,properties)
    dims = list(data.mag.dims)

    if 'Channel' in dims:
        field_props = magnitude.copy()
    else:
        field_props = magnitude.expand_dims('Channel')
        
    grid_area = grid_area.expand_dims('Channel').assign_coords({'Channel':[0]})
    field_props = field_props.assign_coords({'Channel':range(1,len(field_props['Channel'])+1)})

#     print(grid_area)
    field_props = xr.concat([grid_area,field_props],dim='Channel')
    field_props = field_props.transpose(...,'Channel')
    
#     dims.remove('Channel')
#     ridges = ridges.transpose(*dims)
#     ridges = label_rivs(ridges)
    
#     field_props = magnitude
#     print(field_props)
#     print(np.shape(ridges.isel(time=0)))
#     print(np.shape(field_props.isel(time=0)))

    objs= regionprops_table(ridges.isel(time=0).astype(int).values,\
                properties=['label'])
    n= int(len(objs['label'])-len(objs['label'])*.1)
    # print(objs)
    # print(n)
    
    ridge_objs = xr.apply_ufunc(filter_region_props,ridges.astype(int),field_props,n,properties,\
            input_core_dims=[['lat','lon'],['lat','lon','Channel'],[],[]],\
            output_core_dims=[['index','props'],['lat','lon'],['props']],vectorize=True,\
                    dask='parallelized')
    
    vlist = ridge_objs[2].isel(time=0).values

    ridge_dset = ridge_objs[0].sel(props=0).to_dataset(name='label')
    for c in range(len(vlist)-1):
        ridge_dset[vlist[c+1]] = ridge_objs[0].isel(props=c+1)
    ridge_dset = ridge_dset.assign_coords(index=np.arange(1,n+1,1))
    
    if properties.obj['isglobal']:
        filtered_objs = unwrap(ridge_objs[1].fillna(0)).to_dataset(name='object')
    else:
        filtered_objs = ridge_objs[1].fillna(0).to_dataset(name='object')
        
    filtered_objs['core'] = filtered_objs['object']*ridge['core']
    
    object_properties = ridge_dset.dropna(dim='index',how='all')
    return ridge_dset, filtered_objs
    # return ridges.astype(int), field_props


def filter_region_props(ridge_objects,magnitudes,n,props):

    objs= regionprops_table(ridge_objects.astype(int),\
                intensity_image=magnitudes, properties=['label','bbox','centroid',\
              'mean_intensity','min_intensity','max_intensity',\
              'perimeter','eccentricity','bbox_area','area'],\
              extra_properties=(max_loc,))
    
    objs = pd.DataFrame(objs)
    
#     print(objs)

    #Get and remove the duplicate objects
    removables_duplicates = objs[objs.duplicated(subset=\
                list(objs.columns[7:]), keep='first')]
    objs_noduplicates = pd.concat([objs,removables_duplicates])\
                        .drop_duplicates(keep=False)
#     pd.set_option('display.max_rows', None)
#     print(objs_noduplicates)
    #Map geographic Information to labelled regions
    mapped_objects = map_coordinates(objs_noduplicates,props)
#     pd.set_option('display.max_rows', None)
#     pd.set_option('display.max_columns', None)
#     print(mapped_objects)
    
    #Filter for
    # 1. Minimum Area
    # 2. Eccentricity
    # 3. Equator Objects
    removables_shape = mapped_objects.where((mapped_objects['eccentricity']<props.obj['Eccentricity'][0])|\
                    (mapped_objects['eccentricity']>props.obj['Eccentricity'][1])|\
                    (mapped_objects['Obj_Area']<props.obj['Min_Area'])|\
                    ((mapped_objects['wclat']>props.obj['Lat_Mask'][0])&(mapped_objects['wclat']<props.obj['Lat_Mask'][1])&\
                    ((np.minimum(mapped_objects['lat-0'],mapped_objects['lat-1'])>props.obj['Lat_Mask'][0])|\
                    (np.maximum(mapped_objects['lat-0'],mapped_objects['lat-1'])<props.obj['Lat_Mask'][1])))|\
                    ((mapped_objects['wclon']>props.obj['Lon_Mask'][0])&(mapped_objects['wclon']<props.obj['Lon_Mask'][1])))
    

    removables_minthresh = []
    for v in props.minvals:
        removables_minthresh.append(mapped_objects.where(mapped_objects[v]<props.minvals[v]))

    objs_shape_area = pd.concat([mapped_objects, removables_shape, *removables_minthresh])\
                        .drop_duplicates(keep=False).dropna()
    
    #Find the length of objects
    length_mapped = map_length(objs_shape_area,props)
    removables_length = length_mapped.where(length_mapped.length<props.obj['Min_Length'])

    #Remove short objects
    objs_long = pd.concat([length_mapped, removables_length])\
                    .drop_duplicates(keep=False)
    
#     print(pd.concat(removables_minthresh)['label'])
    #Combine all removable object labels
    all_removables = np.unique(np.concatenate((removables_duplicates.label.values,\
                removables_shape.label.values,removables_length.label.values,\
                pd.concat(removables_minthresh)['label'].values),axis=0))
    #Remove the filtered objects
    filtered_objs =  remove_objects(ridge_objects.astype(int),all_removables)

    relabelled_objects = label_rivs(np.nan_to_num(filtered_objs))
    
    
    #All the objects
    filtered_objects = objs_long.dropna(subset=objs_long.columns)
    filtered_objects = filtered_objects.reset_index(drop=True)
    

    nandframe = pd.DataFrame(np.empty((n-len(filtered_objects),len(objs_long.columns)))*np.nan,\
                columns=filtered_objects.columns,index=np.arange(len(filtered_objects),n,1))
#     print(nandframe)
    filled_filtered = pd.concat([filtered_objects,nandframe])

    return filled_filtered, relabelled_objects, xr.DataArray(filled_filtered.columns)

#Function to wrap the data around
def wrapped(data):
    nlons = int(len(data.lon)//1.) #No.of longitudes to be added to right side
    lf_nlons = 0
    pad_lons = np.concatenate([data.lon.values,abs(data.lon.values[:nlons])+360])
    pad_lats = np.concatenate([[-91],data.lat.values,[91]])
    
    pads = []
    coordinates = {}
    for d in data.dims:
        if d == 'lon':
            pads.append((0,nlons))
            coordinates[d] = pad_lons
        elif d == 'lat':
            pads.append((1,1))
            coordinates[d] = pad_lats
        else:
            pads.append((0,0))
            coordinates[d] = data[d]

    #Wrap around in longitudes
    wraped = np.pad(data.values,pad_width=pads,mode='wrap')
    xwrap = xr.DataArray(wraped,coords=coordinates,\
                         dims=data.dims)
    return xwrap

def unwrap(data):
    exdata = data.sel(lon=slice(360,None),lat=slice(-90,90))
    exdata = exdata.assign_coords(lon=exdata.lon-360)
    dumdata = data.sel(lon=slice(exdata.lon.max()+.01,360-0.001))*0.0
    exdata = xr.concat([exdata,dumdata],dim='lon')
    rdata   = data.sel(lon=slice(0,360-0.001),lat=slice(-90,90))
    
    stiched = rdata+exdata
    return stiched

#Function to remove objects touching borders
def xremove_objects(data,props):
    if props.obj['isglobal']:
        data = (data.where(data>0)*0+1).fillna(0)
        removed = xr.apply_ufunc(clear_border,data,input_core_dims=[['lat','lon']],\
                vectorize=True,output_core_dims=[['lat','lon']],dask='allowed')
        removed = removed.where(removed>0).fillna(0)
#         removed = remove_small_objs(removed,props)
        removed = label_rivs(removed)
    else:
#         removed = xr.apply_ufunc(remove_small_objs,removed,props)
        removed = (data.where(data>0)*0+1).fillna(0)
        removed = label_rivs(removed)
    return removed

#Function to remove objects touching borders
def xremove_objects3D(data,props):
    if props.obj['isglobal']:
        data = (data.where(data>0)*0+1).fillna(0)
        removed = xr.apply_ufunc(clear_border,data,input_core_dims=[['lev','lat','lon']],\
                vectorize=True,output_core_dims=[['lev','lat','lon']],dask='allowed')
        removed = removed.sel(lev=slice(props.obj['Map_Plev'].y[0],props.obj['Map_Plev'].y[-1]),\
                        lat=slice(props.obj['Map_Lats'].y[0],props.obj['Map_Lats'].y[-1]))
        
        removed = removed.where(removed>0).fillna(0)
        removed = label_objs3D(removed)
    else:
#         removed = xr.apply_ufunc(remove_small_objs,removed,props)
        removed = (data.where(data>0)*0+1).fillna(0)
        removed = label_objs3D(removed)
    return removed


#Function to remove objects touching borders
def xremove_objects_tgrid(rivs,props):
    if props.obj['isglobal']:
        data = (data.where(data>0)*0+1).fillna(0)
        removed = xr.apply_ufunc(clear_border,data,input_core_dims=[['nlat_t','nlon_t']],\
                vectorize=True,output_core_dims=[['nlat_t','nlon_t']])
        removed = removed.where(removed>0).fillna(0)
#         removed = remove_small_objs(removed,props)
        removed = label_rivs_tgrid(removed)
    else:
#         removed = xr.apply_ufunc(remove_small_objs,removed,props)
        removed = (data.where(data>0)*0+1).fillna(0)
        removed = label_rivs_tgrid(removed)
    return removed

#Function to label rivers
def label_rivs(rivs):
    rlabeled = xr.apply_ufunc(label,rivs,input_core_dims=[['lat','lon']],\
            vectorize=True,output_core_dims=[['lat','lon']],dask='allowed')
    return rlabeled


#Function to label rivers
def label_rivs_tgrid(rivs):
    rlabeled = xr.apply_ufunc(label,rivs,input_core_dims=[['nlat_t','nlon_t']],\
            vectorize=True,output_core_dims=[['nlat_t','nlon_t']])
    return rlabeled

def map_length(objects,props):
    length1 = [geodesic((objects['lat-0'].values[i],objects['lon-0'].values[i]),\
                       (objects['wclat'].values[i],objects['wclon'].values[i])).m\
                           for i in range(len(objects))]
                        
    length2 = [geodesic((objects['wclat'].values[i],objects['wclon'].values[i]),\
                       (objects['lat-1'].values[i],objects['lon-1'].values[i])).m\
                           for i in range(len(objects['label']))]

    objects['length'] = np.array(length1)+np.array(length2)
    
    return objects

def remove_objects(all_objects,label):
    r=all_objects.copy()
    for l in label[~np.isnan(label)]:
        r=np.where(all_objects==l,np.nan,r)
    return r

def max_loc(regionmask, intensity):
    mlocs = np.unravel_index(np.nanargmax(intensity),\
                            intensity.shape)
    return mlocs

# def max_loc(regionmask, intensity):
# #     print(np.shape(intensity[regionmask]))
#     mlocs = []
#     for i in range(1,np.shape(intensity)[-1]):
#         mlocs.append(np.unravel_index(np.nanargmax(intensity[...,i]),\
#                                 intensity[...,i].shape))
# #     print(np.shape(intensity))
#     flat_list = [item for sublist in mlocs for item in sublist]
# #     print(mlocs)
# #     print(flat_list)
#     return flat_list


def remove_and_replace_labels(all_objects,old,new):
    objects=all_objects.copy()
    for o,n in zip(old,new):
        objects[np.where(all_objects==o)]=n
    return objects

def map_coordinates_tgrid(objects,props):
    objects_added = objects.copy()
    objects_added['wclat'] = props.obj['Map_Lats'](objects['centroid-1'].values,\
                                                   objects['centroid-0'].values)
    objects_added['wclon'] = (props.obj['Map_Lons'](objects['centroid-1'].values,\
                                                objects['centroid-0'].values))%360
    
    max_cols = [col for col in objects.columns if 'max_loc' in col]
    for m in range(0,len(max_cols),2):
        objects_added['mx_lat'+str(m//2)] = props.obj['Map_Lats'](objects['max_loc-'+str(m+1)]\
                                            .values+objects['bbox-1'].values,\
                                            objects['max_loc-'+str(m)]\
                                            .values+objects['bbox-0'].values)
        
        objects_added['mx_lon'+str(m//2)] = (props.obj['Map_Lons'](objects['max_loc-'+str(m+1)]\
                                            .values+objects['bbox-1'].values,\
                                            objects['max_loc-'+str(m)]\
                                            .values+objects['bbox-0'].values))%360.
    
    objects_added['lat-0'] = props.obj['Map_Lats'](objects['bbox-1'].values,\
                                                   objects['bbox-0'].values)
    objects_added['lat-1'] = props.obj['Map_Lats'](objects['bbox-3'].values,\
                                                   objects['bbox-2'].values)
    
    objects_added['lon-0'] = (props.obj['Map_Lons'](objects['bbox-1'].values,\
                                                objects['bbox-0'].values))%360.
    objects_added['lon-1'] = (props.obj['Map_Lons'](objects['bbox-3'].values,\
                                                objects['bbox-2'].values))%360.
        
    objects_added['Obj_Area'] = objects['area']*objects['mean_intensity-0']
    return objects_added

def map_coordinates(objects,props):
    objects_added = objects.copy()
    objects_added['wclat'] = props.obj['Map_Lats'](objects['centroid-0'].values)
    objects_added['wclon'] = props.obj['Map_Lons'](objects['centroid-1'].values)%360.
    
    max_cols = [col for col in objects.columns if 'max_loc' in col]
    channels = [n.split('-')[2] for n in max_cols]
    coords = [n.split('-')[1] for n in max_cols]
    
    for co in coords:
        if co=='0':
            eaxis = 'lat'
        elif co=='1':
            eaxis = 'lon'
        else:
            eaxis = 'height'
        
        for ch in channels:
            if co=='0':
                eaxis = 'lat'
                objects_added['mx_{}-{}'.format(eaxis,ch)] = \
                    props.obj['Map_Lats'](objects['max_loc-{}-{}'.format(co,ch)]\
                                            .values+objects['bbox-0'].values)
            elif co=='1':
                eaxis = 'lon'
                objects_added['mx_{}-{}'.format(eaxis,ch)] = \
                    props.obj['Map_Lons'](objects['max_loc-{}-{}'.format(co,ch)]\
                                            .values+objects['bbox-1'].values)%360.
#     for m in range(0,len(max_cols),2):
#         objects_added['mx_lat'+str(m//2)] = props.obj['Map_Lats'](objects['max_loc-'+str(m)]\
#                                             .values+objects['bbox-0'].values)
#         objects_added['mx_lon'+str(m//2)] = props.obj['Map_Lons'](objects['max_loc-'+str(m+1)]\
#                                             .values+objects['bbox-1'].values)%360.
    
    objects_added['lat-0'] = props.obj['Map_Lats'](objects['bbox-0'].values)
    objects_added['lat-1'] = props.obj['Map_Lats'](objects['bbox-2'].values)
    
    objects_added['lon-0'] = props.obj['Map_Lons'](objects['bbox-1'].values)
    objects_added['lon-1'] = props.obj['Map_Lons'](objects['bbox-3'].values)
        
#     objects_added['min_area_size'] = props.obj['Min_Area_Points'](objects_added['wclat'].values)
    objects_added['Obj_Area'] = objects['area']*objects['mean_intensity-0']
    return objects_added

def map_coordinates1(objects,props):
    objects_added = objects.copy()
    objects_added['wclat'] = props.obj['Map_Lats'](objects['weighted_centroid-0'].values)
    
    objects_added['wclon'] = props.obj['Map_Lons'](objects['weighted_centroid-1'].values)%360.
    objects_added['min_area_size'] = props.obj['Min_Area_Points'](objects_added['wclat'].values)
    return objects_added

def map_length(objects,props):
    length1 = [geodesic((objects['lat-0'].values[i],objects['lon-0'].values[i]),\
                       (objects['wclat'].values[i],objects['wclon'].values[i])).m\
                           for i in range(len(objects))]
                        
    length2 = [geodesic((objects['wclat'].values[i],objects['wclon'].values[i]),\
                       (objects['lat-1'].values[i],objects['lon-1'].values[i])).m\
                           for i in range(len(objects['label']))]

    objects['length'] = np.array(length1)+np.array(length2)
    
    return objects

def map_length3D(objects,props):
    length1 = [geodesic((objects['lat-0'].values[i],objects['lon-0'].values[i]),\
                       (objects['clat'].values[i],objects['clon'].values[i])).m\
                           for i in range(len(objects))]
                        
    length2 = [geodesic((objects['clat'].values[i],objects['clon'].values[i]),\
                       (objects['lat-1'].values[i],objects['lon-1'].values[i])).m\
                           for i in range(len(objects['label']))]

    objects['length'] = np.array(length1)+np.array(length2)
    
    return objects

#Function to wrap the data around
def wrapped3Da(data):
    nlons = int(len(data.lon)//1.) #No.of longitudes to be added to right side
    lf_nlons = 0
    #Wrap around in longitudes
    wraped = np.pad(data.values,pad_width=[(0,0),(0,0),(0,0),(lf_nlons,nlons)],mode='wrap')
    wrap_lons = np.concatenate([data.lon.values,abs(data.lon.values[:nlons])+360])

    xwrap = xr.DataArray(wraped,coords={'time':data.time,'lev':data.lev.values,\
                'lon':wrap_lons,'lat':data.lat.values},\
                         dims=['time','lev','lat','lon'])
    return xwrap

#Function to wrap the data around
def wrapped3Db(data):
    nlons = int(len(data.lon)//1.) #No.of longitudes to be added to right side
    lf_nlons = 0
    #Wrap around in longitudes
    wraped = np.pad(data.values,pad_width=[(0,0),(0,0),(0,0),(lf_nlons,nlons)],
                    mode='wrap')
    padded = np.pad(wraped, pad_width=[(0,0),(1,1),(1,1),(0,0)],\
                    mode='constant')
    
    wrap_lons = np.concatenate([data.lon.values,\
                                abs(data.lon.values[:nlons])+360])
    pad_lats = np.concatenate([[-91],data.lat.values,[91]])
    pad_levs = np.concatenate([[data.lev.values[0]+10],data.lev.values,\
                               [data.lev.values[-1]-10]])
    
    xwrap = xr.DataArray(padded,coords={'time':data.time,'lev':pad_levs,\
                'lon':wrap_lons,'lat':pad_lats},\
                         dims=['time','lev','lat','lon'])
    return xwrap

def unwrap(data):
    exdata = data.sel(lon=slice(360,None),lat=slice(-90,90))
    exdata = exdata.assign_coords(lon=exdata.lon-360)
    dumdata = data.sel(lon=slice(exdata.lon.max()+.01,360-0.001))*0.0
    exdata = xr.concat([exdata,dumdata],dim='lon')
    rdata   = data.sel(lon=slice(0,360-0.001),lat=slice(-90,90))
    
    stiched = rdata+exdata
    return stiched

#Function to label rivers
def label_objs3D(rivs):
    rlabeled = xr.apply_ufunc(label,rivs,input_core_dims=[['lev','lat','lon']],\
            vectorize=True,output_core_dims=[['lev','lat','lon']],dask='allowed')
    return rlabeled

def map_coordinates3D(objects,props):
    objects_added = objects.copy()
    objects_added['clev'] = props.obj['Map_Plev'](objects['centroid-0'].values)
    objects_added['clat'] = props.obj['Map_Lats'](objects['centroid-1'].values)
    objects_added['clon'] = props.obj['Map_Lons'](objects['centroid-2'].values)%360.

    max_cols = [col for col in objects.columns if 'max_loc' in col]
    for m in range(0,len(max_cols),3):
        objects_added['mx_lev'+str(m//3)] = props.obj['Map_Plev'](objects['max_loc-'+str(m)]\
                                            .values+objects['bbox-0'].values)
        objects_added['mx_lat'+str(m//3)] = props.obj['Map_Lats'](objects['max_loc-'+str(m+1)]\
                                            .values+objects['bbox-1'].values)
        objects_added['mx_lon'+str(m//3)] = props.obj['Map_Lons'](objects['max_loc-'+str(m+2)]\
                                            .values+objects['bbox-2'].values)%360.
    
    objects_added['lev-0'] = props.obj['Map_Plev'](objects['bbox-0'].values)
    objects_added['lat-0'] = props.obj['Map_Lats'](objects['bbox-1'].values)
    objects_added['lon-0'] = props.obj['Map_Lons'](objects['bbox-2'].values)
    
    objects_added['lev-1'] = props.obj['Map_Plev'](objects['bbox-3'].values-1)
    objects_added['lat-1'] = props.obj['Map_Lats'](objects['bbox-4'].values-1)
    objects_added['lon-1'] = props.obj['Map_Lons'](objects['bbox-5'].values-1)
    
#     objects_added['min_area_size'] = props.obj['Min_Area_Points'](objects_added['wclat'].values)
    objects_added['Obj_Volume'] = objects['area']*objects['mean_intensity-0']
    return objects_added


def filter_props3D(ridge_objects,magnitudes,props,n):
    objs= regionprops_table(ridge_objects.astype(int),\
                intensity_image=magnitudes, properties=['label','bbox','centroid',\
              'mean_intensity','min_intensity','max_intensity',\
              'bbox_area','area'],extra_properties=(max_loc,))
    
    objs = pd.DataFrame(objs)
    #Get and remove the duplicate objects
    removables_duplicates = objs[objs.duplicated(subset=\
                list(objs.columns[10:]), keep='first')]
    objs_noduplicates = pd.concat([objs,removables_duplicates])\
                        .drop_duplicates(keep=False)
#     pd.set_option('display.max_rows', None)
    #Map geographic Information to labelled regions
    mapped_objects = map_coordinates3D(objs_noduplicates,props)
    
    #Filter for
    # 1. Minimum Area
    # 2. Eccentricity
    # 3. Equator Objects
    removables_shape = mapped_objects.where((mapped_objects['Obj_Volume']<props.obj['Min_Volume'])|\
                    ((mapped_objects['clat']>props.obj['Lat_Mask'][0])&\
                     (mapped_objects['clat']<props.obj['Lat_Mask'][1])&\
                    (mapped_objects['clon']>props.obj['Lon_Mask'][0])&\
                     (mapped_objects['clon']<props.obj['Lon_Mask'][1])))
    

    removables_minthresh = []
    for v in props.minvals:
        removables_minthresh.append(mapped_objects.where(mapped_objects[v]<props.minvals[v]))

    objs_shape_area = pd.concat([mapped_objects, removables_shape, *removables_minthresh])\
                        .drop_duplicates(keep=False).dropna()
    #Find the length of objects
    length_mapped = map_length3D(objs_shape_area,props)
    removables_length = length_mapped.where(length_mapped.length<props.obj['Min_Length'])

    #Remove short objects
    objs_long = pd.concat([length_mapped, removables_length])\
                    .drop_duplicates(keep=False)

    all_removables = np.unique(np.concatenate((removables_duplicates.label.values,\
                removables_shape.label.values,removables_length.label.values,\
                pd.concat(removables_minthresh)['label'].values),axis=0))
    
    #Remove the filtered objects
    filtered_objs =  remove_objects(ridge_objects.astype(int),all_removables)    
    
    #All the objects
    filtered_objects = objs_long.dropna(subset=objs_long.columns)
#     print(filtered_objects)
    filtered_objects = filtered_objects.reset_index(drop=True)
    #Relabel object masks to match properties
    relabelled_objects =  remove_and_replace_labels(\
            np.nan_to_num(filtered_objs).astype(int),\
            filtered_objects['label'].values,filtered_objects.index.values+1)
    
    nandframe = pd.DataFrame(np.empty((n-len(filtered_objects),len(objs_long.columns)))*np.nan,\
                columns=filtered_objects.columns,index=np.arange(len(filtered_objects),n,1))
#     print(nandframe)
    filled_filtered = pd.concat([filtered_objects,nandframe])

    return filled_filtered, relabelled_objects, xr.DataArray(filled_filtered.columns)

def filter_objects3D(ridge,data,properties,option):
#     dims = ['time','lev','lat','lon']
#     ridges = wrapped3Da(ridge.ridges).fillna(0)
#     ridges = xremove_objects(ridges,properties).transpose(*dims)
#     ridges = label_rivs3D(ridges)

    # Selecting whether to use Cores or Ridges
    if option == 'core':
        ridges = ridge.core
    elif option == 'ridges':
        ridges = ridge.ridges
    # Broadcasting the grid area
    grid_area = xr.broadcast(ridges,properties.grid['grid_area'])[1]
    grid_zdis = xr.broadcast(ridges,properties.grid['zdistance'])[1]
    cell_volume = grid_area*grid_zdis
    # print(cell_volume)
    # Wrap the array if we are using global data
    if properties.obj['isglobal']:
        magnitude = wrapped3Da(data.mag)
        ridges = wrapped3Db(ridges.fillna(0))
        cell_volume = wrapped3Da(cell_volume)
    else:
        magnitude = data.mag.copy()
        ridges = ridges.fillna(0)
        
    # Remove small objects and enusure object continuity across borders
    ridges = xremove_objects3D(ridges,properties)
    dims = list(data.mag.dims)

    if 'Channel' in dims:
        field_props = magnitude.copy()
    else:
        field_props = magnitude.expand_dims('Channel')
        
    cell_volume = cell_volume.expand_dims('Channel').assign_coords({'Channel':[0]})
    field_props = field_props.assign_coords({'Channel':range(1,len(field_props['Channel'])+1)})

    field_props = xr.concat([cell_volume,field_props],dim='Channel')
    field_props = field_props.transpose(...,'Channel')
    
    
    objs= regionprops_table(ridges.isel(time=0),properties=['label'])
    n= int(len(objs['label'])-len(objs['label'])*.1)

    ridge_objs = xr.apply_ufunc(filter_props3D,ridges,field_props,properties,n,\
            input_core_dims=[['lev','lat','lon'],['lev','lat','lon','Channel'],[],[]],\
            output_core_dims=[['index','props'],['lev','lat','lon'],['props']],\
            vectorize=True,dask='parallelized')

    vlist = ridge_objs[2].isel(time=0).values

    ridge_dset = ridge_objs[0].sel(props=0).to_dataset(name='label')
    for c in range(len(vlist)-1):
        ridge_dset[vlist[c+1]] = ridge_objs[0].isel(props=c+1)
    ridge_dset = ridge_dset.assign_coords(index=np.arange(1,n+1,1))

    if properties.obj['isglobal']:
        filtered_objs = unwrap(ridge_objs[1].fillna(0)).to_dataset(name='object')
    else:
        filtered_objs = ridge_objs[1].fillna(0).to_dataset(name='object')

#     filtered_objs['core'] = filtered_objs['object']*ridge['cores']

    object_properties = ridge_dset.dropna(dim='index',how='all')
    return ridge_dset, filtered_objs, ridge_objs[1]

def unwrap3D(data):
    exdata = data.sel(lon=slice(360,None),lat=slice(-90,90))
    exdata = exdata.assign_coords(lon=exdata.lon-360)
    dumdata = data.sel(lon=slice(exdata.lon.max()+.01,360-0.001))*0.0
    exdata = xr.concat([exdata,dumdata],dim='lon')
    rdata   = data.sel(lon=slice(0,360-0.001),lat=slice(-90,90))
    
    stiched = rdata+exdata
    return stiched

#Function to label rivers
def label_objs_ugrid3D(rivs):
    rlabeled = xr.apply_ufunc(label,rivs,input_core_dims=[['z_t','nlat_u','nlon_u']],\
            vectorize=True,output_core_dims=[['z_t','nlat_u','nlon_u']])
    return rlabeled

#Function to remove objects touching borders
def xremove_objects_ugrid3D(data,props):
    if props.obj['isglobal']:
        data = (data.where(data>0)*0+1).fillna(0)
        removed = xr.apply_ufunc(clear_border,data,input_core_dims=[['z_t','nlat_u','nlon_u']],\
                vectorize=True,output_core_dims=[['z_t','nlat_u','nlon_u']])
        removed = removed.sel(lev=slice(props.obj['Map_Depth'].y[0],props.obj['Map_Depth'].y[-1]),\
                        lat=slice(props.obj['Map_Lats'].y[0],props.obj['Map_Lats'].y[-1]))
        
        removed = removed.where(removed>0).fillna(0)
        removed = label_objs_ugrid3D(removed)
    else:
#         removed = xr.apply_ufunc(remove_small_objs,removed,props)
        removed = (data.where(data>0)*0+1).fillna(0)
        removed = label_objs_ugrid3D(removed)
    return removed

def filter_objects_ugrid3D(ridge,data,props,option):
   # Creating the compatible grid and data
    for v in props.grid.data_vars:
        data[v] = props.grid[v] 
    
    for v in props.grid.coords:
        data[v] = props.grid[v] 
    
    data['DZU']= data['dz']
    data['DZT']= data['dz']
    data['DXT']= data['DXT'].broadcast_like(data['DZT'])
    data['DYT']= data['DYT'].broadcast_like(data['DZT'])
    data['DXU']= data['DXU'].broadcast_like(data['DZU'])
    data['DYU']= data['DYU'].broadcast_like(data['DZU'])
    
    metrics = {
    ("X",): ["DXU", "DXT"],  # X distances
    ("Y",): ["DYU", "DYT"],  # Y distances
    ("Z",): ["DZU", "DZT"],  # Z distances
    ("X", "Y"): ["UAREA", "TAREA"],  # areas, technically not needed
    }

    # here we get the xgcm compatible dataset
    gridxgcm, dsxgcm = pop_tools.to_xgcm_grid_dataset(
        data, periodic=False,metrics=metrics,
        boundary={"X": "extend", "Y": "extend", "Z": "extend"})
    
    for coord in ["nlat", "nlon"]:
        if coord in dsxgcm.coords:
            dsxgcm = dsxgcm.drop_vars(coord)
    del gridxgcm
    
    # Selecting whether to use Cores or Ridges
    if option == 'cores':
        ridges = ridge.cores
    elif option == 'ridges':
        ridges = ridge.ridges
    # Broadcasting the grid area
    grid_volume = (xr.broadcast(ridges,\
                dsxgcm['DXU']*dsxgcm['DYU']*dsxgcm['DZU'])[1])

#     Wrap the array if we are using global data
    if props.obj['isglobal']:
        magnitude = wrapped3Da(dsxgcm['mag'])
        ridges = wrapped3Db(ridges.fillna(0))
        grid_volume = wrapped3Da(grid_volume)
    else:
        magnitude = dsxgcm['mag'].copy()
        ridges = ridges.fillna(0)
    
    # Remove small objects and enusure object continuity across borders
    ridges = xremove_objects_ugrid3D(ridges,props)
    dims = list(data.mag.dims)

    if 'Channel' in dims:
        field_props = magnitude.copy()
    else:
        field_props = magnitude.expand_dims('Channel')

    grid_volume = grid_volume.expand_dims('Channel').assign_coords({'Channel':[0]})
    field_props = field_props.assign_coords({'Channel':range(1,len(field_props['Channel'])+1)})

    field_props = xr.concat([grid_volume,field_props],dim='Channel')
    field_props = field_props.transpose(...,'Channel')

    objs= regionprops_table(ridges.isel(time=0).astype(int).values,\
                properties=['label'])
    n= int(len(objs['label'])-len(objs['label'])*.1)
    
    ridge_objs = xr.apply_ufunc(filter_ugrid3D,ridges.astype(int),field_props,props,n,\
        input_core_dims=[['z_t','nlat_u','nlon_u'],['z_t','nlat_u','nlon_u','Channel'],[],[]],\
        output_core_dims=[['index','props'],['z_t','nlat_u','nlon_u'],['props']],vectorize=True,\
                            dask='parallelized')
    vlist = ridge_objs[2].isel(time=0).values

    ridge_dset = ridge_objs[0].sel(props=0).to_dataset(name='label')
    for c in range(len(vlist)-1):
        ridge_dset[vlist[c+1]] = ridge_objs[0].isel(props=c+1)
    ridge_dset = ridge_dset.assign_coords(index=np.arange(1,n+1,1))

    if props.obj['isglobal']:
        filtered_objs = unwrap(ridge_objs[1].fillna(0)).to_dataset(name='object')
    else:
        filtered_objs = ridge_objs[1].fillna(0).to_dataset(name='object')

    filtered_objs['core'] = filtered_objs['object']*ridge['cores']

    object_properties = ridge_dset.dropna(dim='index',how='all')
    return ridge_dset, filtered_objs

def filter_ugrid3D(ridge_objects,magnitudes,props,n):
    objs= regionprops_table(ridge_objects.astype(int),\
                intensity_image=magnitudes, properties=['label','bbox','centroid',\
              'mean_intensity','min_intensity','max_intensity',\
              'bbox_area','area'],extra_properties=(max_loc,))
    
    objs = pd.DataFrame(objs)
    #Get and remove the duplicate objects
    removables_duplicates = objs[objs.duplicated(subset=\
                list(objs.columns[10:]), keep='first')]
    objs_noduplicates = pd.concat([objs,removables_duplicates])\
                        .drop_duplicates(keep=False)
#     print(objs_noduplicates)
#     pd.set_option('display.max_rows', None)
    #Map geographic Information to labelled regions
    mapped_objects = map_coordinates_ugrid3D(objs_noduplicates,props)
    
    #Filter for
    # 1. Minimum Area
    # 2. Eccentricity
    # 3. Equator Objects
    removables_shape = mapped_objects.where((mapped_objects['Obj_Volume']<props.obj['Min_Volume'])|\
                    ((mapped_objects['clat']>props.obj['Lat_Mask'][0])&\
                     (mapped_objects['clat']<props.obj['Lat_Mask'][1])&\
                    (mapped_objects['clon']>props.obj['Lon_Mask'][0])&\
                     (mapped_objects['clon']<props.obj['Lon_Mask'][1])))
    

    removables_minthresh = []
    for v in props.minvals:
        removables_minthresh.append(mapped_objects.where(mapped_objects[v]<props.minvals[v]))

    objs_shape_area = pd.concat([mapped_objects, removables_shape, *removables_minthresh])\
                        .drop_duplicates(keep=False).dropna()
    #Find the length of objects
    length_mapped = map_length3D(objs_shape_area,props)
    removables_length = length_mapped.where(length_mapped.length<props.obj['Min_Length'])

    #Remove short objects
    objs_long = pd.concat([length_mapped, removables_length])\
                    .drop_duplicates(keep=False)

    all_removables = np.unique(np.concatenate((removables_duplicates.label.values,\
                removables_shape.label.values,removables_length.label.values,\
                pd.concat(removables_minthresh)['label'].values),axis=0))
    
    #Remove the filtered objects
    filtered_objs =  remove_objects(ridge_objects.astype(int),all_removables)    
    
    #All the objects
    filtered_objects = objs_long.dropna(subset=objs_long.columns)
#     print(filtered_objects)
    filtered_objects = filtered_objects.reset_index(drop=True)
    #Relabel object masks to match properties
    relabelled_objects =  remove_and_replace_labels(\
            np.nan_to_num(filtered_objs).astype(int),\
            filtered_objects['label'].values,filtered_objects.index.values+1)
    
    nandframe = pd.DataFrame(np.empty((n-len(filtered_objects),len(objs_long.columns)))*np.nan,\
                columns=filtered_objects.columns,index=np.arange(len(filtered_objects),n,1))
#     print(nandframe)
    filled_filtered = pd.concat([filtered_objects,nandframe])

    return filled_filtered, relabelled_objects, xr.DataArray(filled_filtered.columns)

def map_coordinates_ugrid3D(objects,props):
    objects_added = objects.copy()
#     print('centers')
    objects_added['czt'] = props.obj['Map_Depth'](objects['centroid-0'].values)
    objects_added['clat'] = props.obj['Map_Lats'](objects['centroid-2'].values,\
                                                   objects['centroid-1'].values)
    objects_added['clon'] = (props.obj['Map_Lons'](objects['centroid-2'].values,\
                                                objects['centroid-1'].values))%360
#     print('maxs')
    max_cols = [col for col in objects.columns if 'max_loc' in col]
    for m in range(0,len(max_cols),3):
        objects_added['mx_zt'+str(m//3)] = props.obj['Map_Depth'](\
                                            objects['max_loc-'+str(m)]\
                                            .values+objects['bbox-0'].values)
        objects_added['mx_lat'+str(m//3)] = props.obj['Map_Lats'](objects['max_loc-'+str(m+2)]\
                                            .values+objects['bbox-1'].values,\
                                            objects['max_loc-'+str(m+1)]\
                                            .values+objects['bbox-0'].values)
        
        objects_added['mx_lon'+str(m//3)] = (props.obj['Map_Lons'](objects['max_loc-'+str(m+2)]\
                                            .values+objects['bbox-1'].values,\
                                            objects['max_loc-'+str(m+1)]\
                                            .values+objects['bbox-0'].values))%360.
#     print('corners')
    
    objects_added['z-0'] = props.obj['Map_Depth'](objects['bbox-0'].values)
    objects_added['z-1'] = props.obj['Map_Depth'](objects['bbox-3'].values-1)
       
    objects_added['lat-0'] = props.obj['Map_Lats'](objects['bbox-2'].values,\
                                                   objects['bbox-1'].values)
    objects_added['lat-1'] = props.obj['Map_Lats'](objects['bbox-4'].values-1,\
                                                   objects['bbox-5'].values-1)
    
    objects_added['lon-0'] = (props.obj['Map_Lons'](objects['bbox-2'].values,\
                                                objects['bbox-1'].values))%360.
    objects_added['lon-1'] = (props.obj['Map_Lons'](objects['bbox-4'].values-1,\
                                                objects['bbox-5'].values-1))%360.
        
    objects_added['Obj_Volume'] = objects['area']*objects['mean_intensity-0']
    return objects_added