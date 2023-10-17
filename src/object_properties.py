import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import interp1d
import metpy as metpy
from skimage.morphology import remove_small_holes
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
from scipy.interpolate import NearestNDInterpolator as nn_interp
import metpy.calc as mpcalc
import sys
sys.path.append('./scafet/')
import object_properties as obp
import ridge_detection as rd
import object_filtering as obf
import object_tracking as obt

class object_properties_ugrid3D:
    def check_coverage(self,data):
        if data.ULONG.min()+data.ULONG.max() >= 360:
            return True
        else:
            return False  
        
    def __init__(self,gridinfo,length,volume,smooth,
                 min_duration,max_distance,shapei,lon_mask,lat_mask):
        min_length = 2e6 if length==None else length
        min_volume = 1e10 if volume==None else volume
#         cell_area = self.wrapped(grid_area.cell_area)
        
#         min_area_pixel = int(smooth*1e3/np.max([gridinfo['DXT'].max(),\
#                                 gridinfo['DYT'].max()]))
        min_volume_pixel = int(volume*1e4/gridinfo['UAREA'].max().values)

        x = gridinfo.nlon.values
        y = gridinfo.nlat.values
        z = np.arange(len(gridinfo.z_t))
        
        X, Y = np.meshgrid(x, y)
        points = (X.flatten(),Y.flatten())
        map_ulons = nn_interp(points, gridinfo['ULONG'].values.flatten())
        map_ulats = nn_interp(points, gridinfo['ULAT'].values.flatten())
        map_zt = interp1d(z, gridinfo.z_t)
        smooth = self.min_length//3 if smooth==None else smooth
#         theta = 45 if theta_t==None else theta_t
        shapei = 0.375 if shapei==None else shapei
#         ecc = [0.5,1.] if eccentricity==None else eccentricity
        
        #Check if the given data is global
        isglobal = self.check_coverage(gridinfo)
        
        
        obj = {'Smooth_Scale':smooth,'Shape_Index':shapei,\
               'Min_Length':min_length,'Min_Volume':min_volume,\
               'Min_Volume_Points':min_volume_pixel,\
               'Min_Duration':min_duration,'Max_Distance':max_distance,\
               'Map_Lons':map_ulons,'Map_Lats':map_ulats,\
               'Map_Depth':map_zt,\
               'isglobal':isglobal,\
               'Lon_Mask':lon_mask,'Lat_Mask':lat_mask}
        
        self.obj = obj
        self.grid = gridinfo
        

class object_properties_tgrid:
    def check_coverage(self,data):
        if data.TLONG.min()+data.TLONG.max() == 360:
            return True
        else:
            return False  
        
    def __init__(self,gridinfo,length,area,smooth,theta_t,
                 min_duration,max_distance,shapei,eccentricity,lon_mask,lat_mask):
        min_length = 2e6 if length==None else length
        min_area = 1e10 if area==None else area
#         cell_area = self.wrapped(grid_area.cell_area)
        
#         min_area_pixel = int(smooth*1e3/np.max([gridinfo['DXT'].max(),\
#                                 gridinfo['DYT'].max()]))
        min_area_pixel = int(area*1e4/gridinfo['TAREA'].max().values)

        x = gridinfo.nlon.values
        y = gridinfo.nlat.values
        X, Y = np.meshgrid(x, y)
        points = (X.flatten(),Y.flatten())
        map_tlons = nn_interp(points, gridinfo['TLONG'].values.flatten())
        map_tlats = nn_interp(points, gridinfo['TLAT'].values.flatten())

        smooth = self.min_length//3 if smooth==None else smooth
        theta = 45 if theta_t==None else theta_t
        shapei = 0.375 if shapei==None else shapei
        ecc = [0.5,1.] if eccentricity==None else eccentricity
        
        #Check if the given data is global
        isglobal = self.check_coverage(gridinfo)
        
#         obj = {'Smooth_Scale':smooth,'Angle_Coherence':theta,'Shape_Index':shapei,\
#               'Min_Length':min_length,'Min_Area':min_area,'Min_Duration':min_duration,\
#               'Max_Distance':max_distance,'Min_Lon_Points':min_area_lon,'Min_Lat_Points':min_area_lat,\
#               'Map_Lons':map_lons,'Map_Lats':map_lats,\
#               'Eccentricity':ecc,'isglobal':isglobal,'Lon_Mask':lon_mask,'Lat_Mask':lat_mask}
        
        obj = {'Smooth_Scale':smooth,'Angle_Coherence':theta,'Shape_Index':shapei,\
               'Min_Length':min_length,'Min_Area':min_area,'Min_Area_Points':min_area_pixel,\
               'Min_Duration':min_duration,'Max_Distance':max_distance,\
               'Map_Lons':map_tlons,'Map_Lats':map_tlats,\
               'Eccentricity':ecc,'isglobal':isglobal,'Lon_Mask':lon_mask,'Lat_Mask':lat_mask}
        
        self.obj = obj
        self.grid = gridinfo
#         grid_deltas = self.mpcalc_gird_delta(grid_area.metpy.parse_cf())
#         xdistance = grid_deltas.xdistance
#         ydistance = grid_deltas.ydistance
        
#         grid = {'xdistance':xdistance,'ydistance':ydistance,'grid_area':grid_area}
#         self.grid = grid
        
#         sigmas = self.calc_sigma(grid_area)         
#         smooth = {'sigma_lon':sigmas.sigma_lon,'sigma_lat':sigmas.sigma_lat}
#         self.smooth = smooth
        
#         self.land = self.get_coastline_info(grid_land)

    
class object_properties2D:    
    def wrapped(self,data):
        nlons = len(data.lon)//1 #No.of longitudes to be added to right side
        lf_nlons = 0
        #Wrap around in longitudes
        wraped = np.pad(data.values,pad_width=[(0,0),(lf_nlons,nlons)],mode='wrap')
        wrap_lons = np.concatenate([data.lon.values,abs(data.lon.values[:nlons])+360])
#         pad_lats = np.concatenate([[-91],data.lat.values,[91]])

        xwrap = xr.DataArray(wraped,coords={'lon':wrap_lons,'lat':data.lat.values},\
                             dims=['lat','lon'])

        return xwrap

    def check_coverage(self,data):
        if data.lon[1]+data.lon[-1] == 360:
            return True
        else:
            return False
    
    def remove_hole_lats(self,data,lat):
        return remove_small_holes(data,lat)

    def remove_hole_lons(self,data,lon):
        return remove_small_holes(data,lon[0])

    def remove_holes(self,data):
        min_lons= xr.apply_ufunc(self.obj['Min_Lon_Points'],data.lon)
        min_lat = xr.apply_ufunc(self.obj['Min_Lat_Points'],data.lat)
        min_lats= np.tile(min_lat.values,(len(min_lons.lon.values),1))
        min_lats= xr.DataArray(min_lats, coords=[min_lons.lon,min_lat.lat])

        removed_lons = xr.apply_ufunc(self.remove_hole_lons,data,\
                             min_lats,input_core_dims=[['lon'],['lon']],\
                            output_core_dims=[['lon']],vectorize=True)
        removed_lats = xr.apply_ufunc(self.remove_hole_lats,data,\
                             np.mean(min_lons),input_core_dims=[['lat'],[]],\
                                 output_core_dims=[['lat']],vectorize=True)
        removed = xr.ufuncs.logical_and(removed_lons,removed_lats)

        return removed

    #Function to get coastline information from landfraction
    def get_coastline_info(self,land):  
        #Removing small water bodies
        island = land.islnd.fillna(0)
#         land_filled = self.remove_holes(island.astype(bool))
        #Get Coastlines
        coastlines = find_boundaries(island)

        coast_info = land.copy()
        coast_info['coastlines'] = xr.DataArray(coastlines,dims=land.dims,coords=land.coords)
        coast_info['coastlines'] = coast_info['coastlines'].where(coast_info['coastlines']>0)

        xgrad = island.differentiate('lon')
        ygrad = island.differentiate('lat')

        xgrad = xgrad.where(xgrad<=0,1)
        ygrad = ygrad.where(ygrad<=0,1)
        xgrad = xgrad.where(xgrad>=0,-1)
        ygrad = ygrad.where(ygrad>=0,-1)
        xgrad = xgrad.where(coast_info['coastlines']==1)
        ygrad = ygrad.where(coast_info['coastlines']==1)

        coast_info['orientation'] = xr.apply_ufunc(np.arctan2,xgrad,ygrad,dask='parallelized')*180/np.pi
        return coast_info
    
    def mpcalc_grid_delta(self,grid_area):
        #Calculate the grid deltas
        grid_delta = metpy.xarray.grid_deltas_from_dataarray(grid_area.metpy.parse_cf()['cell_area'])
        #Calcualte the sigma of the gaussian filter along each longitude; Constant value is used
        grid_lat  = np.concatenate((np.array(grid_delta[1]),np.array(grid_delta[1][-1:,:])),axis=0)

        #Calculate the sigma of the gaussinan filter along each latitude; Values change with lats
        grid_lon =np.concatenate((np.array(grid_delta[0]),np.array(grid_delta[0][:,0:1])),axis=1)

        distance = xr.Dataset({'xdistance':(['lat','lon'],grid_lon),\
                    'ydistance':(['lat','lon'],grid_lat)},\
                        coords={'lat':(['lat'],grid_area.lat.values),
                         'lon':(['lon'],grid_area.lon.values)})

        distance.xdistance.attrs['units'] = 'm'
        distance.ydistance.attrs['units'] = 'm'
        return distance

    def calc_sigma(self,grid_area):
        #Calculate the sigma of the gaussinan filter along each longitude; Values constant
        sigma_lat = np.rint(self.obj['Smooth_Scale']/np.squeeze(self.grid['ydistance']))
        sigma_lat = np.where(sigma_lat>len(grid_area.lat)//1,len(grid_area.lat)//1,sigma_lat)
        sigma_lat = np.where(sigma_lat<2,2,sigma_lat)/(2*np.pi)

        #Calculate the sigma of the gaussinan filter along each latitude; Values change with lats
        sigma_lon = np.rint(self.obj['Smooth_Scale']/np.squeeze(self.grid['xdistance']))
        sigma_lon = np.where(sigma_lon>len(grid_area.lon)//1,len(grid_area.lon)//1,sigma_lon)
        sigma_lon = np.where(sigma_lon<2,2,sigma_lon)/(2*np.pi)

        sigmas = xr.Dataset({'sigma_lat':(['lat','lon'],sigma_lat),\
                    'sigma_lon':(['lat','lon'],sigma_lon)},\
                        coords={'lat':(['lat'],grid_area.lat.values),
                         'lon':(['lon'],grid_area.lon.values)})
        return sigmas

    def plot_properties(self,option):
        if option == 'grid':
            fig,ax = plt.subplots(1,3,figsize=(12,5),dpi=100,subplot_kw={'projection': \
                                        ccrs.EckertIII(central_longitude=0.0)})
            properties=['xdistance','ydistance','grid_area']

            for a,p in zip(ax,properties):
                if p == 'grid_area':
                    self.grid[p]['cell_area'].plot(ax=a,transform=ccrs.PlateCarree(),\
                                    cbar_kwargs={'shrink':.25,'aspect':15})
                else:
                    self.grid[p].plot(ax=a,transform=ccrs.PlateCarree(),\
                                          cbar_kwargs={'shrink':.25,'aspect':15,'format':'%.1e'})

                a.coastlines()
                a.gridlines(linestyle='--',alpha=0.4)
                a.set_title(p)
        elif option == 'smooth':
            fig,ax = plt.subplots(1,2,figsize=(12,5),dpi=100,subplot_kw={'projection': \
                                    ccrs.EckertIII(central_longitude=0.0)})
            properties=['sigma_lon','sigma_lat']
            for a,p in zip(ax,properties):
                self.smooth[p].plot(ax=a,transform=ccrs.PlateCarree(),\
                    cbar_kwargs={'shrink':.25,'aspect':15,'label':r'$\sigma$'})
                a.coastlines()
                a.gridlines(linestyle='--',alpha=0.4)
                a.set_title(p)
                
        elif option == 'land':
            fig,ax = plt.subplots(1,3,figsize=(12,5),dpi=100,subplot_kw={'projection': \
                                        ccrs.EckertIII(central_longitude=0.0)})
            properties=['islnd','coastlines','orientation']
            cmaps = ['binary','binary','hsv']
            for a,p,c in zip(ax,properties,cmaps):
                c=self.land[p].plot(ax=a,transform=ccrs.PlateCarree(),cmap=c,\
                    add_colorbar=False)
                a.gridlines(linestyle='--',alpha=0.4)
                a.set_title(p)
            cb_pos = ax[-1].get_position()
            pos_cax= fig.add_axes([cb_pos.x0+cb_pos.width+0.01,cb_pos.y0,\
                                   cb_pos.width/30,cb_pos.height])
            cb=plt.colorbar(c, cax=pos_cax, orientation='vertical')
            cb.set_label(r'$degree$')
                
    def print_properties(self):        
        obj_value_unit = ['m','degree',\
                          'unitless','m',r'm$^2$', 'counts','m',\
                          'counts','counts','degrees','degrees',\
                          'unitless','0-1','degrees','degrees']
        
        obj = {'Object Properties':list(self.obj.keys()),'Values':list(self.obj.values()),\
              'Units':obj_value_unit}
        
        obj = pd.DataFrame(obj)
        return obj
    
    def __init__(self,grid_area,grid_land,length,area,smooth,theta_t,
                 min_duration,max_distance,shapei,eccentricity,lon_mask,lat_mask):
        min_length = 2e6 if length==None else length
        min_area = 1e10 if area==None else area
        cell_area = self.wrapped(grid_area.cell_area)
        
#         min_area_pixel = interp1d(cell_area.lat.values,\
#                (min_area/cell_area.isel(lon=0)))
        min_area_lat = interp1d(cell_area.lat.values,\
               ((min_area/4)**.5/cell_area.isel(lon=0)**.5))
        min_area_lon = interp1d(cell_area.lon.values,\
               ((min_area/4)**.5/cell_area.isel(lat=0)**.5))
        
        map_lats = interp1d(range(len(cell_area.lat.values)),\
               cell_area.lat.values,fill_value=(-90.,90.),bounds_error=False)
        map_lons = interp1d(range(len(cell_area.lon.values)),\
               cell_area.lon.values)

        smooth = self.min_length//3 if smooth==None else smooth
        theta = 45 if theta_t==None else theta_t
        shapei = 0.375 if shapei==None else shapei
        ecc = [0.5,1.] if eccentricity==None else eccentricity
        
        #Check if the given data is global
        isglobal = self.check_coverage(grid_area)
        
        obj = {'Smooth_Scale':smooth,'Angle_Coherence':theta,'Shape_Index':shapei,\
              'Min_Length':min_length,'Min_Area':min_area,'Min_Duration':min_duration,\
              'Max_Distance':max_distance,'Min_Lon_Points':min_area_lon,'Min_Lat_Points':min_area_lat,\
              'Map_Lons':map_lons,'Map_Lats':map_lats,\
              'Eccentricity':ecc,'isglobal':isglobal,'Lon_Mask':lon_mask,'Lat_Mask':lat_mask}
        self.obj = obj
        
        grid_deltas = self.mpcalc_grid_delta(grid_area.metpy.parse_cf())
        xdistance = grid_deltas.xdistance
        ydistance = grid_deltas.ydistance
        
        grid = {'xdistance':xdistance,'ydistance':ydistance,'grid_area':grid_area}
        self.grid = grid
        
        sigmas = self.calc_sigma(grid_area)         
        smooth = {'sigma_lon':sigmas.sigma_lon,'sigma_lat':sigmas.sigma_lat}
        self.smooth = smooth
        
        self.land = self.get_coastline_info(grid_land)

        
class object_properties3D:
    def wrapped(self,data):
        nlons = len(data.lon)//1 #No.of longitudes to be added to right side
        lf_nlons = 0
        #Wrap around in longitudes
        wraped = np.pad(data.values,pad_width=[(0,0),(lf_nlons,nlons)],mode='wrap')
        wrap_lons = np.concatenate([data.lon.values,abs(data.lon.values[:nlons])+360])
#         pad_lats = np.concatenate([[-91],data.lat.values,[91]])

        xwrap = xr.DataArray(wraped,coords={'lon':wrap_lons,'lat':data.lat.values},\
                             dims=['lat','lon'])

        return xwrap
    
    def remove_hole_lats(self,data,lat):
        return remove_small_holes(data,lat)

    def remove_hole_lons(self,data,lon):
        return remove_small_holes(data,lon[0])

    def remove_holes(self,data):
#         min_lons= xr.apply_ufunc(self.obj['Min_Lon_Points'],data.lon)
#         min_lat = xr.apply_ufunc(self.obj['Min_Lat_Points'],data.lat)
#         min_lats= np.tile(min_lat.values,(len(min_lons.lon.values),1))
#         min_lats= xr.DataArray(min_lats, coords=[min_lons.lon,min_lat.lat])

#         removed_lons = xr.apply_ufunc(self.remove_hole_lons,data,\
#                              min_lats,input_core_dims=[['lon'],['lon']],\
#                             output_core_dims=[['lon']],vectorize=True)
#         removed_lats = xr.apply_ufunc(self.remove_hole_lats,data,\
#                              np.mean(min_lons),input_core_dims=[['lat'],[]],\
#                                  output_core_dims=[['lat']],vectorize=True)
#         removed = xr.ufuncs.logical_and(removed_lons,removed_lats)
        removed = data
        return removed

    
    def check_coverage(self,data):
        if data.lon[1]+data.lon[-1] == 360:
            return True
        else:
            return False
    
    #Function to get coastline information from landfraction
    def get_coastline_info(self,land):  
        #Removing small water bodies
        island = land.islnd.fillna(0)
        land_filled = self.remove_holes(island.astype(bool))
        #Get Coastlines
        coastlines = find_boundaries(land_filled)

        coast_info = land.copy()
        coast_info['coastlines'] = xr.DataArray(coastlines,dims=land.dims,coords=land.coords)
        coast_info['coastlines'] = coast_info['coastlines'].where(coast_info['coastlines']>0)

        xgrad = island.differentiate('lon')
        ygrad = island.differentiate('lat')

        xgrad = xgrad.where(xgrad<=0,1)
        ygrad = ygrad.where(ygrad<=0,1)
        xgrad = xgrad.where(xgrad>=0,-1)
        ygrad = ygrad.where(ygrad>=0,-1)
        xgrad = xgrad.where(coast_info['coastlines']==1)
        ygrad = ygrad.where(coast_info['coastlines']==1)

        coast_info['orientation'] = xr.ufuncs.arctan2(xgrad,ygrad)*180/np.pi
        return coast_info
    
    def mpcalc_grid_delta2D(self,dset):
        v = list(dset.data_vars)[0]
        #Calculate the grid deltas
        grid_delta = metpy.xarray.grid_deltas_from_dataarray(dset.metpy.parse_cf()[v].isel(time=0,lev=0))
        
        #Calcualte the sigma of the gaussian filter along each longitude; Constant value is used
        grid_lat  = np.concatenate((np.array(grid_delta[1]),np.array(grid_delta[1][-1:,:])),axis=0)
        #Calculate the sigma of the gaussinan filter along each latitude; Values change with lats
        grid_lon =np.concatenate((np.array(grid_delta[0]),np.array(grid_delta[0][:,0:1])),axis=1)
        
        distance = xr.Dataset({'xdistance':(['lat','lon'],grid_lon),\
                    'ydistance':(['lat','lon'],grid_lat)},\
                        coords={'lat':(['lat'],dset.lat.values),
                         'lon':(['lon'],dset.lon.values)})

        distance.xdistance.attrs['units'] = 'm'
        distance.ydistance.attrs['units'] = 'm'
        return distance
    
    def plev_to_height(self,dset):
        height = mpcalc.pressure_to_height_std(dset.lev)
        height = xr.Dataset({'height':(['lev'],height.data*1000)},coords={'lev':(['lev'],\
                                        dset.lev.values)})['height']
        height.attrs['units'] = 'm'
        return height

    def calc_sigma(self,grid_area):
        #Calculate the sigma of the gaussinan filter along each longitude; Values constant
        sigma_lat = np.rint(self.obj['Smooth_Scale']/np.squeeze(self.grid['ydistance']))
        sigma_lat = np.where(sigma_lat>len(grid_area.lat)//1,len(grid_area.lat)//1,sigma_lat)
        sigma_lat = np.where(sigma_lat<2,2,sigma_lat)/(2*np.pi)

        #Calculate the sigma of the gaussinan filter along each latitude; Values change with lats
        sigma_lon = np.rint(self.obj['Smooth_Scale']/np.squeeze(self.grid['xdistance']))
        sigma_lon = np.where(sigma_lon>len(grid_area.lon)//1,len(grid_area.lon)//1,sigma_lon)
        sigma_lon = np.where(sigma_lon<2,2,sigma_lon)/(2*np.pi)

        sigmas = xr.Dataset({'sigma_lat':(['lat','lon'],sigma_lat),\
                    'sigma_lon':(['lat','lon'],sigma_lon)},\
                        coords={'lat':(['lat'],grid_area.lat.values),
                         'lon':(['lon'],grid_area.lon.values)})
        return sigmas

    def plot_properties(self,option):
        if option == 'grid':
            fig   = plt.figure(figsize=(12,4),dpi=300)
            spec  = gridspec.GridSpec(ncols=3, nrows=2,\
                                    figure=fig)
            ax=[]
            c=0.0
            ax.append(fig.add_subplot(spec[0, 0],projection=ccrs.PlateCarree(central_longitude=c)))
            ax.append(fig.add_subplot(spec[0, 1],projection=ccrs.PlateCarree(central_longitude=c)))
            ax.append(fig.add_subplot(spec[0, 2],projection=ccrs.PlateCarree(central_longitude=c)))
            ax.append(fig.add_subplot(spec[1, :1]))

#             fig,ax = plt.subplots(1,3,figsize=(12,5),dpi=100,subplot_kw={'projection': \
#                                         ccrs.EckertIII(central_longitude=0.0)})
            properties=['xdistance','ydistance','grid_area','zdistance']

            for a,p in zip(ax,properties):
                if p == 'zdistance':
                    self.grid[p].plot(ax=a)
                    a.grid(linestyle='--',alpha=0.4)
                    a.set_title(p)
                else:
                    self.grid[p].plot(ax=a,transform=ccrs.PlateCarree(),\
                                          cbar_kwargs={'shrink':.5,'aspect':15,'format':'%.1e'})

                    a.coastlines()
                    a.gridlines(linestyle='--',alpha=0.4)
                    a.set_title(p)
        elif option == 'smooth':
            fig,ax = plt.subplots(1,2,figsize=(12,5),dpi=100,subplot_kw={'projection': \
                                    ccrs.PlateCarree(central_longitude=0.0)})
            properties=['sigma_lon','sigma_lat']
            for a,p in zip(ax,properties):
                self.smooth[p].plot(ax=a,transform=ccrs.PlateCarree(),\
                    cbar_kwargs={'shrink':.25,'aspect':15,'label':r'$\sigma$'})
                a.coastlines()
                a.gridlines(linestyle='--',alpha=0.4)
                a.set_title(p)
                
        elif option == 'land':
            fig,ax = plt.subplots(1,3,figsize=(12,5),dpi=100,subplot_kw={'projection': \
                                        ccrs.PlateCarree(central_longitude=0.0)})
            properties=['islnd','coastlines','orientation']
            cmaps = ['binary','binary','hsv']
            for a,p,c in zip(ax,properties,cmaps):
                c=self.land[p].plot(ax=a,transform=ccrs.PlateCarree(),cmap=c,\
                    add_colorbar=False)
                a.gridlines(linestyle='--',alpha=0.4)
                a.set_title(p)
            cb_pos = ax[-1].get_position()
            pos_cax= fig.add_axes([cb_pos.x0+cb_pos.width+0.01,cb_pos.y0,\
                                   cb_pos.width/30,cb_pos.height])
            cb=plt.colorbar(c, cax=pos_cax, orientation='vertical')
            cb.set_label(r'$degree$')
                
    def print_properties(self):      
        
#         obj = {'Smooth_Scale':smooth,'Shape_Index':shapei,'Min_Length':min_length,\
#                'Min_Projected_Area':min_area,'Map_Lons':map_lons,'Map_Lats':map_lats,\
#                'isglobal':isglobal,'Lon_Mask':lon_mask,'Lat_Mask':lat_mask}
        obj_value_unit = ['m','unitless','m','m',r'm$^3$','timestep','m',\
                          'degrees','degrees','hPa',\
                          '0-1','degrees','degrees']
    
        obj = {'Object Properties':list(self.obj.keys()),'Values':list(self.obj.values()),\
              'Units':obj_value_unit}
        
        obj = pd.DataFrame(obj)
        return obj
    
    def __init__(self,data,grid_area,xdistance,ydistance,zdistance,grid_land,length,height,volume,\
                 smooth,min_duration,max_distance,shapei,lon_mask,lat_mask):
        
        min_length = 2e6 if length==[] else length
        min_height = 5e3 if height==[] else height
        min_volume = 1e10 if volume==[] else volume
        smooth = self.min_length//3 if smooth==[] else smooth
        shapei = [0,1] if shapei==[] else shapei
        #Check if the given data is global
        isglobal = self.check_coverage(grid_area)
        
        if ((len(xdistance)==0)|(len(ydistance)==0)):
            grid_deltas = self.mpcalc_grid_delta2D(data.metpy.parse_cf())
            
            if (xdistance==[]):
                Xdistance = grid_deltas.xdistance
            else:
                Xdistance = xdistance
            
            if ydistance == []:
                Ydistance = grid_deltas.ydistance
            else:
                Ydistance = ydistance
        
        if (zdistance==[]):
            Zdistance = self.plev_to_height(data)
            
        if (len(grid_area) == 0):
            grid_area = Xdistance*Ydistance
        else:
            grid_area = grid_area
        
        wrapped_data = self.wrapped(Xdistance)
        map_lats = interp1d(range(len(wrapped_data.lat.values)),\
                wrapped_data.lat.values,fill_value=(-90.,90.),bounds_error=False)
        map_lons = interp1d(range(len(wrapped_data.lon.values)),\
                wrapped_data.lon.values)
        map_plev = interp1d(range(len(data.lev.values)),\
                data.lev.values)
        
        obj = {'Smooth_Scale':smooth,'Shape_Index':shapei,'Min_Length':min_length,\
               'Min_Height':min_height,'Min_Volume':min_volume,\
               'Min_Duration':min_duration,'Max_Distance':max_distance,\
               'Map_Lons':map_lons,'Map_Lats':map_lats,'Map_Plev':map_plev,\
               'isglobal':isglobal,'Lon_Mask':lon_mask,'Lat_Mask':lat_mask}
        self.obj = obj
        
        grid = {'xdistance':Xdistance,'ydistance':Ydistance,'zdistance':Zdistance.metpy.dequantify(),\
                'grid_area':grid_area['cell_area'].metpy.dequantify()}
        self.grid = grid
        
        sigmas = self.calc_sigma(grid_area)         
        smooth = {'sigma_lon':sigmas.sigma_lon,'sigma_lat':sigmas.sigma_lat}
        self.smooth = smooth
        
        self.land = self.get_coastline_info(grid_land)
