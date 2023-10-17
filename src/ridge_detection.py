import numpy as np
import xarray as xr
import pandas as pd
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import interp1d
import metpy.calc as mpcalc
import pop_tools
import sys
sys.path.append('./scafet/')
import object_properties as obp
import ridge_detection as rd
import object_filtering as obf
import object_tracking as obt

class ridgeDetector:    
    def apply_smoother(self,data,properties):
        if self.ndims == 2:
            return smoother2D(data,properties)
        elif self.ndims == 3:
            return smoother3D(data,properties)
    def apply_smoother_tgrid(self,data,properties):
        if self.ndims == 2:
            return smoother_tgrid2D(data,properties)
        elif self.ndims == 3:
            return smoother_tgrid3D(data,properties)
        
    def apply_smoother_ugrid(self,data,properties):
        if self.ndims == 2:
            return smoother_ugrid2D(data,properties)
        elif self.ndims == 3:
            return smoother_ugrid3D(data,properties)
        
    def apply_ridge_detection_tgrid(self,data,properties):
        if self.ndims == 2:
            return ridge_detection_tgrid2D(data,properties)
        elif self.ndims == 3:
            return ridge_detection_tgrid3D(data,properties)
      
    def apply_ridge_detection_ugrid(self,data,properties):
        if self.ndims == 2:
            return ridge_detection_ugrid2D(data,properties)
        elif self.ndims == 3:
            return ridge_detection_ugrid3D(data,properties)
        
    def apply_ridge_detection(self,data,properties):
        if (self.ndims == 2):
            if self.vector:
                return ridgeDetection2D_vector(data,properties)
            else:
                return ridgeDetection2D_scalar(data,properties)
        if (self.ndims == 3):
                return ridgeDetection3D_scalar(data,properties)
            
    def __init__(self,data,ndims=None):
        self.ndims = len(list(data.dims))-1 if ndims==None else ndims
        self.vector = True if len(data.data_vars)==2 else False

def gfilter_lons(data,sigma=5):
#     print(np.shape(data))
    sys.path.append('./scafet/')
    import ridge_detection as rd
    return gaussian_filter1d(data,sigma[0],mode='wrap')

def gfilter_lats(data,sigma=3):
#     print(np.shape(data))
    sys.path.append('./scafet/')
    import ridge_detection as rd
    return gaussian_filter1d(data,sigma,mode='nearest')

def non_homogenous_filter(var,slat,slon):
    lonfilter = xr.apply_ufunc(gfilter_lons,var,slon,\
                input_core_dims=[['lon'],['lon']],output_core_dims=[['lon']],\
                vectorize=True, dask='parallelized')

    filtered = xr.apply_ufunc(gfilter_lats,lonfilter,np.mean(slat),\
                input_core_dims=[['lat'],[]],output_core_dims=[['lat']],\
                vectorize=True,dask='parallelized')

#     filtered = xr.DataArray(np.swapaxes(filtered,1,2),dims=['time','lat','lon'],\
#                             coords={'time':var.time,'lat':var.lat,'lon':var.lon})
    return filtered

def calc_magnitude(a, b):
    func = lambda x, y: np.sqrt(x ** 2 + y ** 2)
    with xr.set_options(keep_attrs=True):
        mag = xr.apply_ufunc(func, a, b, dask='allowed')
    return mag

def angleBw(x1,y1,x2,y2):
    ang = np.arccos(((x1*x2)+(y1*y2))/(calc_magnitude(x1,y1)*calc_magnitude(x2,y2)))
    ang = ang*180/np.pi
    return ang

def smoother2D(var,properties):
    #Filter IVTX and IVTY using variying sigma
    for v,i in zip(list(var.data_vars),range(len(list(var.data_vars)))):

        with xr.set_options(keep_attrs=True):
            vn = non_homogenous_filter(var[v],\
                    properties.smooth['sigma_lat'],properties.smooth['sigma_lon'])
            if i==0:
                smoothed = vn.to_dataset(name=v)
                smoothed[v] = smoothed[v].transpose(*var[v].dims)
            else:
                smoothed[v] = vn.transpose(*var[v].dims)
    return smoothed

def smoother3D(var,properties):
    #Filter IVTX and IVTY using variying sigma
    for v,i in zip(list(var.data_vars),range(len(list(var.data_vars)))):

        with xr.set_options(keep_attrs=True):
            vn = non_homogenous_filter(var[v],\
                    properties.smooth['sigma_lat'],properties.smooth['sigma_lon'])
            if i==0:
                smoothed = vn.to_dataset(name=v)
                smoothed[v] = smoothed[v].transpose(*var[v].dims)
            else:
                smoothed[v] = vn.transpose(*var[v].dims)
    return smoothed

def smoother_tgrid2D(data,props):
    max_size = np.max([props.grid['DYT'].max(),\
                       props.grid['DXT'].max()])
    sigma = props.obj['Smooth_Scale']*1e2/max_size

    data_sm =  xr.apply_ufunc(gaussian_filter,data.fillna(0),sigma,\
                input_core_dims=[['nlat','nlon'],[]],\
                output_core_dims=[['nlat','nlon']],\
                dask='parallelized')
    data_aux =  xr.apply_ufunc(gaussian_filter,data.fillna(0)*0+1,sigma,\
                input_core_dims=[['nlat','nlon'],[]],\
                output_core_dims=[['nlat','nlon']],\
                dask='parallelized')
    data_sm = data_sm.where(~np.isnan(data))/data_aux
    att = data.attrs
    data_sm = data_sm.assign_attrs(att)
#     data_sm = data_sm.to_dataset(name='mag')
    return data_sm

def smoother_ugrid3D(data,props):
    max_size = np.max([props.grid['DYU'].max(),\
                       props.grid['DXU'].max()])
    sigma = props.obj['Smooth_Scale']*1e2/max_size

    data_sm =  xr.apply_ufunc(gaussian_filter,data.fillna(0),sigma,\
                input_core_dims=[['nlat','nlon'],[]],\
                output_core_dims=[['nlat','nlon']],\
                dask='parallelized')
    data_aux =  xr.apply_ufunc(gaussian_filter,data.fillna(0)*0+1,sigma,\
                input_core_dims=[['nlat','nlon'],[]],\
                output_core_dims=[['nlat','nlon']],\
                dask='parallelized')
    data_sm = data_sm.where(~np.isnan(data))/data_aux
    att = data.attrs
    data_sm = data_sm.assign_attrs(att)
#     data_sm = data_sm.to_dataset(name='mag')
    return data_sm

def ridge_detection_tgrid2D(data, props):
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
    # Construc Area Objects    
    Ax = gridxgcm.interp(dsxgcm['TAREA'],'X')
    Ay = gridxgcm.interp(dsxgcm['TAREA'],'Y')
    Axx = gridxgcm.interp(Ax,'X')
    Axy = gridxgcm.interp(Ax,'Y')
    Ayy = gridxgcm.interp(Ay,'Y')
    # Construct gradients
    d_dx = gridxgcm.diff(dsxgcm['mag'], 'X')/Ax
    d_dy = gridxgcm.diff(dsxgcm['mag'], 'Y')/Ay
    d2_dx2 = gridxgcm.diff(d_dx,'X')/Axx
    d2_dy2 = gridxgcm.diff(d_dy,'Y')/Ayy
    d2_dxy = gridxgcm.diff(d_dx,'Y')/Axy
    d2_dyx = gridxgcm.diff(d_dy,'X')/Axy
    
    #Arranging it for Matric Calculation
    r1= xr.concat([d2_dx2,gridxgcm.interp(d2_dxy,axis=('X','Y'))],\
                  dim='C1').expand_dims(dim='C2') 
    r2= xr.concat([gridxgcm.interp(d2_dyx,axis=('X','Y')),d2_dy2],\
                  dim='C1').expand_dims(dim='C2')                  
    H_elems = xr.concat([r1,r2],dim='C2')
    ##------------------------------------------------------------------------   
    Hessian = H_elems.transpose('time','nlon_t','nlat_t','C1','C2')
    
#     #Calcualtion of eigen vectors and eigen values for the symmetric array
#     eigvals = xr.apply_ufunc(np.linalg.eigh,Hessian,\
#             input_core_dims=[['nlat_t','nlon_t','C1','C2']],\
#             output_core_dims=[['nlat_t','nlon_t','e'],['nlat_t','nlon_t','n','e']],\
#             dask='parallelized',vectorize=True)
#     eigval = eigvals[0].assign_coords(e=['min','max'])
#     eigvec = eigvals[1].assign_coords(n=['x','y'])
#     eigvec = eigvec.assign_coords(e=['min','max'])
    
    #Calcualtion of eigen values for the symmetric array
    eigvals = xr.apply_ufunc(np.linalg.eigvalsh,Hessian,\
            input_core_dims=[['nlat_t','nlon_t','C1','C2']],\
            output_core_dims=[['nlat_t','nlon_t','e']],\
            dask='parallelized',vectorize=True)
    eigval = eigvals.assign_coords(e=['min','max'])
#     eigvec = eigvals[1].assign_coords(n=['x','y'])
#     eigvec = eigvec.assign_coords(e=['min','max'])
    shapei = (2/np.pi)*np.arctan((eigval.sel(e='min')+eigval.sel(e='max'))/\
                                 (eigval.sel(e='min')-eigval.sel(e='max')))

    
#     shapei['TLAT'] = data['TLAT']
#     shapei['TLONG'] = data["TLONG"]
    
    eigs = shapei.to_dataset(name='sindex')
#     eigs['Ar'] = Ar/magnitude.mag
#     eigs['gAr'] = gAr/magnitude.mag
#     ridges = magnitude.mag.where((eigs['sindex']>=props.obj['Shape_Index'][0])&\
#                              (eigs['sindex']<=props.obj['Shape_Index'][1]) & \
#                                  (abs(eigs['Ar'])>0.))
    ridges = data['mag'].where((eigs['sindex']>=props.obj['Shape_Index'][0])&\
                             (eigs['sindex']<=props.obj['Shape_Index'][1]))
    eigs['ridges'] = ridges*0+1
#     eigs['ridges'] = eigs['sindex']>*0+1
#     print(d_dx)
#     print(d_dy)
    eigs['gradient'] = gridxgcm.interp(d_dx,axis='X')+gridxgcm.interp(d_dy,axis='Y')
    cores = gridxgcm.interp(gridxgcm.diff(np.sign(eigs['gradient']),axis='X'),axis='X')+\
                gridxgcm.interp(gridxgcm.diff(np.sign(eigs['gradient']),axis='Y'),axis='Y')
    eigs['cores'] = (abs(cores).where(abs(cores)>=1))*0+1
    
    return eigs

def ridgeDetection2D_scalar(magnitude,props):
    d_dlon = mpcalc.first_derivative(magnitude.metpy.parse_cf()['mag'], axis='lon')
    d_dlat = mpcalc.first_derivative(magnitude.metpy.parse_cf()['mag'], axis='lat')
    
    d2_d2lon = mpcalc.second_derivative(magnitude.metpy.parse_cf()['mag'], axis='lon')
    d2_d2lat = mpcalc.second_derivative(magnitude.metpy.parse_cf()['mag'], axis='lat')

    d2_dlon_dlat = mpcalc.first_derivative(d_dlon, axis='lat')
    d2_dlat_dlon = mpcalc.first_derivative(d_dlat, axis='lon')
    
    #Arranging it for Matric Calculation
    r1= xr.concat([d2_d2lon,d2_dlon_dlat],dim='C1').expand_dims(dim='C2')                  
    r2= xr.concat([d2_dlat_dlon,d2_d2lat],dim='C1').expand_dims(dim='C2')                  
    H_elems = xr.concat([r1,r2],dim='C2')
    ##------------------------------------------------------------------------   
    Hessian = H_elems.transpose('time','lon','lat','C1','C2')
    #Calcualtion of eigen vectors and eigen values for the symmetric array
    eigvals = xr.apply_ufunc(np.linalg.eigh,Hessian,\
            input_core_dims=[['lat','lon','C1','C2']],\
            output_core_dims=[['lat','lon','e'],['lat','lon','n','e']],\
            dask='parallelized',vectorize=True)
    
    eigval = eigvals[0].assign_coords(e=['min','max'])
    eigvec = eigvals[1].assign_coords(n=['x','y'])
    eigvec = eigvec.assign_coords(e=['min','max'])
    
    #Transport along the ridge direction
    Ar = (magnitude.mag*eigvec.sel(e='max',n='x')*-1 +\
        magnitude.mag*eigvec.sel(e='max',n='y')*-1)/np.sqrt(eigvec.sel(e='max',n='x')**2+\
                            eigvec.sel(e='max',n='y')**2)
    
    #Transport along the ridge direction
    gAr = d_dlon*eigvec.sel(e='max',n='x')*-1 +\
        d_dlat*eigvec.sel(e='max',n='y')*-1
    
    gradient = d_dlon+d_dlat
    shapei = (2/np.pi)*np.arctan((eigval.sel(e='min')+eigval.sel(e='max'))/\
                                 (eigval.sel(e='min')-eigval.sel(e='max')))
    eigs = shapei.to_dataset(name='sindex')
    eigs['Ar'] = Ar/magnitude.mag
    eigs['gAr'] = gAr/magnitude.mag
    ridges = magnitude.mag.where((eigs['sindex']>props.obj['Shape_Index'][0])&\
                             (eigs['sindex']<=props.obj['Shape_Index'][1]) & \
                                 (abs(eigs['Ar'])>0.))
    eigs['ridges'] = ridges*0+1
    eigs['gradient'] = gradient
    
    
    cores = (abs(np.sign(gradient).differentiate('lat'))+\
                     abs(np.sign(gradient).differentiate('lon')))
    eigs['core'] = cores.where((cores>0) & (shapei>props.obj['Shape_Index'][0]) &\
                              (shapei>props.obj['Shape_Index'][0]))*0+1

    return eigs


def ridgeDetection2D_vector(vector,props):
    magnitude = calc_magnitude(vector.u,vector.v).to_dataset(name='mag')
#     print(magnitude.mag)
#     magnitude.mag = magnitude.mag.transpose(*vector.u.dims)
#     print(magnitude.mag)
    
    d_dlon = mpcalc.first_derivative(magnitude.metpy.parse_cf()['mag'], axis='lon')
    d_dlat = mpcalc.first_derivative(magnitude.metpy.parse_cf()['mag'], axis='lat')
    
    d2_d2lon = mpcalc.second_derivative(magnitude.metpy.parse_cf()['mag'], axis='lon')
    d2_d2lat = mpcalc.second_derivative(magnitude.metpy.parse_cf()['mag'], axis='lat')

    d2_dlon_dlat = mpcalc.first_derivative(d_dlon, axis='lat')
    d2_dlat_dlon = mpcalc.first_derivative(d_dlat, axis='lon')

    #Arranging it for Matric Calculation
    r1= xr.concat([d2_d2lon,d2_dlon_dlat],dim='C1').expand_dims(dim='C2')                  
    r2= xr.concat([d2_dlat_dlon,d2_d2lat],dim='C1').expand_dims(dim='C2')                  
    H_elems = xr.concat([r1,r2],dim='C2')
    ##------------------------------------------------------------------------   
    Hessian = H_elems.transpose('time','lon','lat','C1','C2')
    #Calcualtion of eigen vectors and eigen values for the symmetric array
    eigvals = xr.apply_ufunc(np.linalg.eigh,Hessian,\
            input_core_dims=[['lat','lon','C1','C2']],\
            output_core_dims=[['lat','lon','e'],['lat','lon','n','e']],\
            dask='parallelized',vectorize=True)
    
    eigval = eigvals[0].assign_coords(e=['min','max'])
    eigvec = eigvals[1].assign_coords(n=['x','y'])
    eigvec = eigvec.assign_coords(e=['min','max'])
    
    #Transport along the ridge direction
    Ar = (magnitude.mag*eigvec.sel(e='max',n='x')*-1 +\
        magnitude.mag*eigvec.sel(e='max',n='y')*-1)/np.sqrt(eigvec.sel(e='max',n='x')**2+\
        eigvec.sel(e='max',n='y')**2)
    #Transport along the ridge direction
    gAr = d_dlon*eigvec.sel(e='max',n='x')*-1 +\
        d_dlat*eigvec.sel(e='max',n='y')*-1
    
    #Angle between the transport direction and ridge direction
    theta = angleBw(eigvec.sel(e='min',n='x')*-1,eigvec.sel(e='min',n='y')*-1,\
                  vector.u,vector.v)
        
    shapei = (2/np.pi)*np.arctan((eigval.sel(e='min')+eigval.sel(e='max'))/\
                                 (eigval.sel(e='min')-eigval.sel(e='max')))
    eigs = shapei.to_dataset(name='sindex')
#     eigs['Ar'] = Ar/magnitude.mag
#     eigs['gAr'] = gAr/magnitude.mag
#     eigs['theta'] = theta-90
    
    theta = theta-90
    ridges = magnitude.mag.where((eigs['sindex']>props.obj['Shape_Index'][0])&\
                             (eigs['sindex']<=props.obj['Shape_Index'][1]) & \
                             (theta<props.obj['Angle_Coherence']))

    eigs['ridges'] = ridges*0+1
#     eigs['mag'] = magnitude.mag
    
    simple_gradient = d_dlon+d_dlat
    simple_gradient['lat'] = np.arange(len(simple_gradient['lat']))
    simple_gradient['lon'] = np.arange(len(simple_gradient['lon']))

    zeroline = abs(xr.ufuncs.sign(simple_gradient).differentiate('lat'))+\
                         abs(xr.ufuncs.sign(simple_gradient).differentiate('lon'))

    zeroline['lat'] = eigs['lat']
    zeroline['lon'] = eigs['lon']
    
    cores = zeroline.where(zeroline>0)
    eigs['core'] = cores.where((eigs['sindex']>props.obj['Shape_Index'][0])&\
                    (eigs['sindex']<=props.obj['Shape_Index'][1]) & \
                    (theta<props.obj['Angle_Coherence'])).fillna(0)

#     eigs['dx'] = d_dlon
#     eigs['dy'] = d_dlat
#     eigs['ddx'] = d2_d2lon
#     eigs['ddy'] = d2_d2lat
    return eigs


# def ridgeDetection2D_scalar_grid(magnitude,props):
#     d_dlon = magnitude.mag.differentiate('lon')
#     d_dlat = magnitude.mag.differentiate('lat')
    
#     d2_d2lon = d_dlon.differentiate('lon')
#     d2_d2lat = d_dlat.differentiate('lat')


#     d2_dlon_dlat = d_dlon.differentiate('lat')
#     d2_dlat_dlon = d_dlat.differentiate('lon')

#     #Arranging it for Matric Calculation
#     r1= xr.concat([d2_d2lon,d2_dlon_dlat],dim='C1').expand_dims(dim='C2')                  
#     r2= xr.concat([d2_dlat_dlon,d2_d2lat],dim='C1').expand_dims(dim='C2')                  
#     H_elems = xr.concat([r1,r2],dim='C2')
#     ##------------------------------------------------------------------------   
#     Hessian = H_elems.transpose('time','lon','lat','C1','C2')
#     #Calcualtion of eigen vectors and eigen values for the symmetric array
#     eigvals = xr.apply_ufunc(np.linalg.eigh,Hessian,\
#             input_core_dims=[['lat','lon','C1','C2']],\
#             output_core_dims=[['lat','lon','e'],['lat','lon','n','e']],\
#             dask='parallelized',vectorize=True)
    
#     eigval = eigvals[0].assign_coords(e=['min','max'])
#     eigvec = eigvals[1].assign_coords(n=['x','y'])
#     eigvec = eigvec.assign_coords(e=['min','max'])
    
#     #Transport along the ridge direction
#     Ar = (magnitude.mag*eigvec.sel(e='max',n='x')*-1 +\
#         magnitude.mag*eigvec.sel(e='max',n='y')*-1)/np.sqrt(eigvec.sel(e='max',n='x')**2+\
#                             eigvec.sel(e='max',n='y')**2)
    
#     #Transport along the ridge direction
#     gAr = d_dlon*eigvec.sel(e='max',n='x')*-1 +\
#         d_dlat*eigvec.sel(e='max',n='y')*-1
    
#     gradient = d_dlon+d_dlat
#     shapei = (2/np.pi)*np.arctan((eigval.sel(e='min')+eigval.sel(e='max'))/\
#                                  (eigval.sel(e='min')-eigval.sel(e='max')))
#     eigs = shapei.to_dataset(name='sindex')
#     eigs['Ar'] = Ar/magnitude.mag
#     eigs['gAr'] = gAr/magnitude.mag
#     ridges = magnitude.mag.where((eigs['sindex']>props.obj['Shape_Index'][0])&\
#                              (eigs['sindex']<=props.obj['Shape_Index'][1]) & \
#                                  (abs(eigs['Ar'])>0.))
#     eigs['ridges'] = ridges*0+1
#     eigs['gradient'] = gradient
#     cores = (abs(xr.ufuncs.sign(gradient).differentiate('lat'))+\
#                      abs(xr.ufuncs.sign(gradient).differentiate('lon')))
#     eigs['core'] = cores.where((cores>0) & (shapei>props.obj['Shape_Index'][0]) &\
#                               (shapei>props.obj['Shape_Index'][0]))*0+1

#     return eigs

def convert_plev_to_height(dset):
    dset_levs = dset.lev
    height = mpcalc.pressure_to_height_std(dset_levs)
    dset  = dset.assign_coords({'lev':height*1000}).rename({'lev':'height'})
    dset.height.attrs['units'] = 'm'
    dset.height.attrs['standard_name'] = 'height'
    dset.height.attrs['long_name'] = 'vertical_distance_above_the_surface.'
    dset.height.attrs['axis'] = 'Z'
    return dset, dset_levs

def convert_height_to_plev(dset,plev):
    dset = dset.rename({'height':plev.name})
    dset = dset.assign_coords({plev.name:plev})
    return dset


def ridgeDetection3D_scalar(magnitude,props):
    magnitude, plevs = convert_plev_to_height(magnitude)
    
    magnitude = magnitude.transpose('time', 'height', 'lat', 'lon')
    d_dlon = mpcalc.first_derivative(magnitude.metpy.parse_cf()['mag'],\
                                     axis='lon')
    d_dlat = mpcalc.first_derivative(magnitude.metpy.parse_cf()['mag'],\
                                     axis='lat')
    d_dlev = mpcalc.first_derivative(magnitude.metpy.parse_cf()['mag'],\
                                     axis='height')
    
    d2_d2lon = mpcalc.second_derivative(magnitude.metpy.parse_cf()['mag'],\
                                        axis='lon')
    d2_d2lat = mpcalc.second_derivative(magnitude.metpy.parse_cf()['mag'],\
                                        axis='lat')
    d2_d2lev = mpcalc.second_derivative(magnitude.metpy.parse_cf()['mag'],\
                                        axis='height')

    d2_dlon_dlat = mpcalc.first_derivative(d_dlon, axis='lat')
    d2_dlon_dlev = mpcalc.first_derivative(d_dlon, axis='height')

    d2_dlat_dlon = mpcalc.first_derivative(d_dlat, axis='lon')
    d2_dlat_dlev = mpcalc.first_derivative(d_dlat, axis='height')

    d2_dlev_dlon = mpcalc.first_derivative(d_dlev, axis='lon')
    d2_dlev_dlat = mpcalc.first_derivative(d_dlev, axis='lat')

    #Arranging it for Matric Calculation
    r1= xr.concat([d2_d2lon,d2_dlon_dlat,d2_dlon_dlev],\
                  dim='C1').expand_dims(dim='C2')                  
    r2= xr.concat([d2_dlat_dlon,d2_d2lat,d2_dlat_dlev],\
                  dim='C1').expand_dims(dim='C2')
    r3= xr.concat([d2_dlev_dlon,d2_dlev_dlat,d2_d2lev],\
                  dim='C1').expand_dims(dim='C2') 
    H_elems = xr.concat([r1,r2,r3],dim='C2')
    ##------------------------------------------------------------------------   
    Hessian = H_elems.transpose('time','height','lon','lat','C2','C1').fillna(0)
    #Calcualtion of eigen vectors and eigen values for the symmetric array
    eigvals = xr.apply_ufunc(np.linalg.eigh,Hessian,\
            input_core_dims=[['lat','lon','height','C1','C2']],\
            output_core_dims=[['lat','lon','height','e'],['lat','lon','height','n','e']],\
            dask='parallelized',vectorize=True)
            
    eigval = eigvals[0].assign_coords(e=np.arange(3))
    eigvec = eigvals[1].assign_coords(n=['x','y','z'])
    eigvec = eigvec.assign_coords(e=np.arange(3))
       
    gradient = d_dlon+d_dlat+d_dlev
    shapei0 = (2/np.pi)*np.arctan((eigval.isel(e=1)+eigval.isel(e=2))/\
                                 (eigval.sel(e=1)-eigval.sel(e=2)))
#     shapei1 = (2/np.pi)*np.arctan((eigval.sel(e=0)+eigval.sel(e=2))/\
#                                  (eigval.sel(e=0)-eigval.sel(e=2)))
#     shapei2 = (2/np.pi)*np.arctan((eigval.sel(e=0)+eigval.sel(e=1))/\
#                                  (eigval.sel(e=0)-eigval.sel(e=1)))

    
    eigs = shapei0.to_dataset(name='sindex0')

    ridges = magnitude.mag.where((eigs['sindex0']>props.obj['Shape_Index'][0])&\
                             (eigs['sindex0']<=props.obj['Shape_Index'][1]))
    eigs['ridges'] = ridges*0+1
    eigs['gradient'] = gradient
    cores = (abs(xr.ufuncs.sign(d_dlon).diff('lat'))+\
                     abs(xr.ufuncs.sign(d_dlat).diff('lon'))+\
                    abs(xr.ufuncs.sign(d_dlev).diff('height')))
    eigs['core'] = cores.where((cores>=2) & (shapei0>props.obj['Shape_Index'][0]) &\
                              (shapei0>props.obj['Shape_Index'][0]))
    eigs = convert_height_to_plev(eigs,plevs)
    return eigs

def ridge_detection_ugrid3D(data, props):
    # Creating the compatible grid and data
    for v in ['TLAT','TLONG','ULAT','ULONG','DXT', 'DYT',\
               'DXU', 'DYU', 'TAREA', 'UAREA', 'KMT',\
               'REGION_MASK', 'dz']:
        data[v] = props.grid[v] 
    
    for v in props.grid.coords:
        data[v] = props.grid[v] 
    
    data['DZU']= data['dz']
    data['DZT']= data['dz']
    data['DXT']= data['DXT'].broadcast_like(data['DZT'])
    data['DYT']= data['DYT'].broadcast_like(data['DZT'])
    data['DXU']= data['DXU'].broadcast_like(data['DZU'])
    data['DYU']= data['DYU'].broadcast_like(data['DZU'])
#     data['mag']= data['DYU']*0+data['mag'].values
    
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
        
    # Construc Area Objects    
    Dx = gridxgcm.interp(dsxgcm['DXU'],'X')
    Dy = gridxgcm.interp(dsxgcm['DYU'],'Y')
    Dz = gridxgcm.interp(dsxgcm['DZU'],'Z')
    
    Dxy = gridxgcm.interp(Dx,'Y')
    Dxz = gridxgcm.interp(Dx,'Z')
    Dyz = gridxgcm.interp(Dy,'Z')

#     Construct gradients
    d_dx = gridxgcm.diff(dsxgcm['mag'], 'X')/Dx
    d_dy = gridxgcm.diff(dsxgcm['mag'], 'Y')/Dy
    d_dz = gridxgcm.diff(dsxgcm['mag'], 'Z')/Dz
    
    d2_dxx = gridxgcm.diff(d_dx,'X')/dsxgcm['DXU']
    d2_dyy = gridxgcm.diff(d_dy,'Y')/dsxgcm['DYU']
    d2_dzz = gridxgcm.diff(d_dz,'Z')/dsxgcm['DZU']

    d2_dxy = gridxgcm.diff(d_dx,'Y')/Dxy
    d2_dyx = gridxgcm.diff(d_dy,'X')/Dxy

    d2_dxz = gridxgcm.diff(d_dx,'Z')/Dxz  
    d2_dzx = gridxgcm.diff(d_dz,'X')/Dxz
    
    d2_dyz = gridxgcm.diff(d_dy,'Z')/Dyz  
    d2_dzy = gridxgcm.diff(d_dz,'Y')/Dyz

    #Arranging it for Matric Calculation
    r1= xr.concat([d2_dxx,gridxgcm.interp(d2_dxy,axis=('X','Y')),\
                  gridxgcm.interp(d2_dxz,axis=('X','Z'))],\
                  dim='C1').expand_dims(dim='C2') 
    
    r2= xr.concat([gridxgcm.interp(d2_dyx,axis=('X','Y')),d2_dyy,\
                  gridxgcm.interp(d2_dyz,axis=('Y','Z'))],\
                  dim='C1').expand_dims(dim='C2') 
    r3= xr.concat([gridxgcm.interp(d2_dzx,axis=('Z','X')),\
                  gridxgcm.interp(d2_dzy,axis=('Z','Y')),d2_dzz],\
                  dim='C1').expand_dims(dim='C2')

    H_elems = xr.concat([r1,r2,r3],dim='C2')
    ##------------------------------------------------------------------------   
    Hessian = H_elems.transpose('time','z_t','nlon_u','nlat_u','C1','C2').fillna(0)
    
    #Calcualtion of eigen values for the symmetric array
    eigvals = xr.apply_ufunc(np.linalg.eigvalsh,Hessian,\
            input_core_dims=[['z_t','nlat_u','nlon_u','C1','C2']],\
            output_core_dims=[['z_t','nlat_u','nlon_u','e']],\
            dask='parallelized',vectorize=True)
    
    eigval = eigvals.assign_coords(e=np.arange(3))
    shapei0 = (2/np.pi)*np.arctan((eigval.isel(e=1)+eigval.isel(e=2))/\
                                 (eigval.sel(e=1)-eigval.sel(e=2)))
    shapei1 = (2/np.pi)*np.arctan((eigval.sel(e=0)+eigval.sel(e=2))/\
                                 (eigval.sel(e=0)-eigval.sel(e=2)))
    shapei2 = (2/np.pi)*np.arctan((eigval.sel(e=0)+eigval.sel(e=1))/\
                                 (eigval.sel(e=0)-eigval.sel(e=1)))
    
    eigs = shapei0.to_dataset(name='s0')
    eigs['s1'] = shapei1
    eigs['s2'] = shapei2
    
    ridges = eigs['s0'].where((eigs['s0']>props.obj['Shape_Index'][0])&\
                             (eigs['s0']<=props.obj['Shape_Index'][1]))
    eigs['ridges'] = ridges*0+1
    eigs['gradient'] = gridxgcm.interp(d_dx,axis='X')+\
                    gridxgcm.interp(d_dy,axis='Y')+\
                    gridxgcm.interp(d_dz,axis='Z')
    
    eigs['gradient'] = gridxgcm.interp(d_dx,axis='X')+gridxgcm.interp(d_dy,axis='Y')
    cores = gridxgcm.interp(gridxgcm.diff(np.sign(eigs['gradient']),axis='X'),axis='X')+\
                gridxgcm.interp(gridxgcm.diff(np.sign(eigs['gradient']),axis='Y'),axis='Y')
    eigs['cores'] = (abs(cores).where(abs(cores)>=2))*0+1
    
    return eigs