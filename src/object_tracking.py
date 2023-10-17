from tqdm import tqdm
import numpy as np
import xarray as xr
import pandas as pd
from geopy.distance import geodesic
import itertools
import time
import sys
sys.path.append('./scafet/')
import object_properties as obp
import ridge_detection as rd
import object_filtering as obf
import object_tracking as obt


class Tracker:    
    def apply_tracking(self,object_properties,object_mask):
        tracked_objects = track_along_time(object_properties,self.basedon,\
                                          self.max_distance)
        
        tlength = len(tracked_objects['time'].values)
        tstep = np.arange(1,tlength+1)
        tracks = tracked_objects.assign_coords({'times':('time',tstep)})
        tracks = tracks.rename({'label':'old_label'})
        tracks['tstep'] = tracks['times'].broadcast_like(tracks['old_label'])
        tracks['label'] = tracks['index'].broadcast_like(tracks['old_label'])
        tracks['timestamp'] = tracks['time'].broadcast_like(tracks['old_label'])

        sel_elements = list(tracks.data_vars)[7:]
        sel_elements.append('old_label')
        sel_arrays = []
        for se in sel_elements:
            sel_arrays.append(tracks[se].values.flatten())
        tracks_dframe = pd.DataFrame(np.transpose(sel_arrays),\
                                     columns=sel_elements)


        tracked_ids = tracks_dframe.groupby(['track_ids'])
        tsteps_objs = tracked_ids.size()
        obj_counts = pd.DataFrame(tracked_ids,columns=['track_ids','Objects'])
        obj_counts['nsteps'] = tsteps_objs.values

        robjects = obj_counts.where(obj_counts['nsteps']>self.min_duration).dropna()
        robjs = robjects.drop(['track_ids', 'nsteps'], axis = 1)
        nos = len(robjs.index)

        if nos==0:
            xtracks = pd.DataFrame(np.expand_dims(np.repeat(np.nan,\
                    len(tracks_dframe.columns)),axis=0),\
                    columns=tracks_dframe.columns).to_xarray()
            tracked_mask = object_mask*np.nan
        else:
            if nos==1:
                duration_tracks = robjs.values[0][0]
            else:
                duration_tracks = pd.concat(np.squeeze(robjs.values).tolist(),axis=0)

            xtracks = duration_tracks.to_xarray().assign_coords({'index':\
                        duration_tracks['track_ids'].values}).rename({'index':'id'})

            tracks_dframe['xlabel'] = duration_tracks['label']
            mindex=pd.MultiIndex.from_frame(tracks_dframe[['timestamp','label']])

            before_track = pd.Series(tracks_dframe['label'].astype(int).values,\
                                index=mindex).to_xarray()
            before_track = before_track.rename({'timestamp':'time'})
            before_track = before_track.rename({'label':'index'})

            after_track = pd.Series((tracks_dframe['xlabel']*0+tracks_dframe['track_ids'])\
                                .fillna(0).astype(int).values,\
                                index=mindex).to_xarray().fillna(0)
            after_track = after_track.rename({'timestamp':'time'})
            after_track = after_track.rename({'label':'index'})
            
            after_track = after_track.assign_coords(time=object_mask.time)
            before_track = before_track.assign_coords(time=object_mask.time)
            
            tracked_mask = xr.apply_ufunc(remove_and_replace_labels, object_mask.object, \
                        before_track, after_track, vectorize=True,\
                        output_dtypes=[object_mask.object.dtype],\
                        input_core_dims=[['lat','lon'],['index'],['index']],\
                        output_core_dims=[['lat','lon']],dask='parallelized')
        
        return xtracks,tracked_mask

    def apply_tracking_tgrid(self,object_properties,object_mask):
        tracked_objects = track_along_time(object_properties,self.basedon,\
                                          self.max_distance)
#         tracked_objects = tracked_objects.dropna(subset = ['travelled_distance'])
#         tracked_objects = tracked_objects.where(\
#                     ~np.isnan(tracked_objects['travelled_distance']))
        
        tlength = len(tracked_objects['time'].values)
        tstep = np.arange(1,tlength+1)
        tracks = tracked_objects.assign_coords({'times':('time',tstep)})
        tracks = tracks.rename({'label':'old_label'})
        tracks['tstep'] = tracks['times'].broadcast_like(tracks['old_label'])
        tracks['label'] = tracks['index'].broadcast_like(tracks['old_label'])
        tracks['timestamp'] = tracks['time'].broadcast_like(tracks['old_label'])

        sel_elements = list(tracks.data_vars)[7:]
        sel_elements.append('old_label')
        sel_arrays = []
        for se in sel_elements:
            sel_arrays.append(tracks[se].values.flatten())
        tracks_dframe = pd.DataFrame(np.transpose(sel_arrays),\
                                     columns=sel_elements)
#         tracks_dframe = tracks_dframe.dropna(subset = ['travelled_distance'])
#         print(tracks_dframe[['track_ids','travelled_distance']])
#         print(tracks_dframe)

        tracked_ids = tracks_dframe.groupby(['track_ids'])
        tsteps_objs = tracked_ids.size()
#         print(tsteps_objs)
    
        obj_counts = pd.DataFrame(tracked_ids,columns=['track_ids','Objects'])
        obj_counts['nsteps'] = tsteps_objs.values
        
        robjects = obj_counts.where(obj_counts['nsteps']>self.min_duration).dropna()
        robjs = robjects.drop(['track_ids', 'nsteps'], axis = 1)
#         print(robjs)
        nos = len(robjs.index)

        if nos==0:
            xtracks = pd.DataFrame(np.expand_dims(np.repeat(np.nan,\
                    len(tracks_dframe.columns)),axis=0),\
                    columns=tracks_dframe.columns).to_xarray()
            tracked_mask = object_mask*np.nan
        else:
            if nos==1:
                duration_tracks = robjs.values[0][0]
            else:
                duration_tracks = pd.concat(np.squeeze(robjs.values).tolist(),axis=0)
            
#             print(duration_tracks.dropna(subset = ['travelled_distance']))
#             dropna(subset = ['column1_name',
#             tracks = duration_tracks.dropna()
            tracks = duration_tracks.copy()
            xtracks = tracks.to_xarray().assign_coords({'index':\
                        tracks['track_ids'].values}).rename({'index':'id'})

            tracks_dframe['xlabel'] = duration_tracks['label']

            mindex=pd.MultiIndex.from_frame(tracks_dframe[['timestamp','label']])

            before_track = pd.Series(tracks_dframe['label'].astype(int).values,\
                                index=mindex).to_xarray()
            before_track = before_track.rename({'timestamp':'time'})
            before_track = before_track.rename({'label':'index'})

            after_track = pd.Series((tracks_dframe['xlabel']*0+tracks_dframe['track_ids'])\
                                .fillna(0).astype(int).values,\
                                index=mindex).to_xarray().fillna(0)
            after_track = after_track.rename({'timestamp':'time'})
            after_track = after_track.rename({'label':'index'})
#             print(after_track)
#             print(before_track)
#             print(object_mask)
            tracked_mask = xr.apply_ufunc(remove_and_replace_labels, object_mask.object, \
                        before_track, after_track, vectorize=True,\
                        output_dtypes=[object_mask.object.dtype],\
                        input_core_dims=[['nlat_t','nlon_t'],['index'],['index']],\
                        output_core_dims=[['nlat_t','nlon_t']],dask='parallelized')
            
#             tracked_mask = xr.apply_ufunc(remove_and_replace_labels, object_mask.object, \
#                         before_track, after_track, vectorize=True,\
#                         output_dtypes=[object_mask.object.dtype],\
#                         dask='parallelized')
#             print(tracks_dframe)

        return xtracks,tracked_mask

    def apply_tracking3D(self,object_properties,object_mask):
        tracked_objects = track_along_time(object_properties,self.basedon,\
                                          self.max_distance)
#         tracked_objects = tracked_objects.dropna(subset = ['travelled_distance'])
#         tracked_objects = tracked_objects.where(\
#                     ~np.isnan(tracked_objects['travelled_distance']))
        
        tlength = len(tracked_objects['time'].values)
        tstep = np.arange(1,tlength+1)
        tracks = tracked_objects.assign_coords({'times':('time',tstep)})
        tracks = tracks.rename({'label':'old_label'})
        tracks['tstep'] = tracks['times'].broadcast_like(tracks['old_label'])
        tracks['label'] = tracks['index'].broadcast_like(tracks['old_label'])
        tracks['timestamp'] = tracks['time'].broadcast_like(tracks['old_label'])

        sel_elements = list(tracks.data_vars)[10:]
#         print(sel_elements)
        sel_elements.append('old_label')
        sel_arrays = []
        for se in sel_elements:
            sel_arrays.append(tracks[se].values.flatten())
        tracks_dframe = pd.DataFrame(np.transpose(sel_arrays),\
                                     columns=sel_elements)

        tracked_ids = tracks_dframe.groupby(['track_ids'])
        tsteps_objs = tracked_ids.size()
#         print(tsteps_objs)
    
        obj_counts = pd.DataFrame(tracked_ids,columns=['track_ids','Objects'])
        obj_counts['nsteps'] = tsteps_objs.values
        
        robjects = obj_counts.where(obj_counts['nsteps']>self.min_duration).dropna()
        robjs = robjects.drop(['track_ids', 'nsteps'], axis = 1)
#         print(robjs)
        nos = len(robjs.index)

        if nos==0:
            xtracks = pd.DataFrame(np.expand_dims(np.repeat(np.nan,\
                    len(tracks_dframe.columns)),axis=0),\
                    columns=tracks_dframe.columns).to_xarray()
            tracked_mask = object_mask*np.nan
        else:
            if nos==1:
                duration_tracks = robjs.values[0][0]
            else:
                duration_tracks = pd.concat(np.squeeze(robjs.values).tolist(),axis=0)
            
            tracks = duration_tracks.copy()
            xtracks = tracks.to_xarray().assign_coords({'index':\
                        tracks['track_ids'].values}).rename({'index':'id'})

            tracks_dframe['xlabel'] = duration_tracks['label']

            mindex=pd.MultiIndex.from_frame(tracks_dframe[['timestamp','label']])

            before_track = pd.Series(tracks_dframe['label'].astype(int).values,\
                                index=mindex).to_xarray()
            before_track = before_track.rename({'timestamp':'time'})
            before_track = before_track.rename({'label':'index'})

            after_track = pd.Series((tracks_dframe['xlabel']*0+tracks_dframe['track_ids'])\
                                .fillna(0).astype(int).values,\
                                index=mindex).to_xarray().fillna(0)
            after_track = after_track.rename({'timestamp':'time'})
            after_track = after_track.rename({'label':'index'})

            tracked_mask = xr.apply_ufunc(remove_and_replace_labels, object_mask.object, \
                        before_track, after_track, vectorize=True,\
                        output_dtypes=[object_mask.object.dtype],\
                        input_core_dims=[['lat','lon'],['index'],['index']],\
                        output_core_dims=[['lat','lon']],dask='parallelized')

        return xtracks,tracked_mask
        
    def __init__(self,based,props):
        self.max_distance = props.obj['Max_Distance']
        self.min_duration = props.obj['Min_Duration']+1
        self.basedon = based
        
    
def remove_and_replace_labels(all_objects,old,new):
#     print(all_objects.shape)
#     print(old,new)
    objects=all_objects.copy()
    for o,n in zip(old,new):
        objects[np.where(all_objects==o)]=n
    return objects


def closest_obj_distance(lat0,lon0,lat1,lon1,id0,id1,max_d):
    t0 = [(y,x) for y,x in zip(lat0[~np.isnan(lat0)],\
                               lon0[~np.isnan(lon0)])]
    t1 = [(y,x) for y,x in zip(lat1[~np.isnan(lat1)],\
                               lon1[~np.isnan(lon1)])]

    clist = [t0,t1]
    
    if ((t1==[]) & (t0==[])):
        # print('Both Empty')
        track_ids = np.ones(len(lat0))*np.nan
        closest_object = np.ones(len(lat0))*np.nan
#         id0[:]=id1
    elif (t0 == []):
        # print('T1 Empty')
        track_ids = np.ones(len(lat0))*np.nan
        closest_object = np.ones(len(lat0))*np.nan
    elif (t1 == []):
        # print('T0 Empty')
        track_ids = id0
        closest_object = np.ones(len(lat0))*0
    else:
        # print(t1)
        # print(t0)
        # print(id1)
        # print(id0)
        idx = [id0[~np.isnan(id0)],id1[~np.isnan(id1)]]
        # print(idx)
        
        idc = np.reshape([(p[0],p[1]) for p in itertools.product(*idx)],\
                     (len(t0),len(t1),2))
    
        cords_comb = np.reshape([(p[0],p[1]) for p in itertools.product(*clist)],\
                       (len(t0),len(t1),4))
        distances = [geodesic(p[0],p[1]).km for p in itertools.product(*clist)]
        distances = np.reshape(distances,(len(t0),len(t1)))
        min_loc = np.argmin(distances,axis=1)
        min_comb_id = np.take_along_axis(idc,\
            np.expand_dims(np.expand_dims(min_loc,1),1),axis=1)
        min_comb_cords = np.take_along_axis(cords_comb,\
            np.expand_dims(np.expand_dims(min_loc,1),1),axis=1)
        min_distances = np.array(np.nanmin(distances,axis=1))

        # print(min_distances, min_comb_id)
        if len(min_distances)>1:
            result = np.concatenate([np.expand_dims(min_distances,1),\
                                 np.squeeze(min_comb_id)],axis=1)
        elif len(min_distances)==1:
            result = np.concatenate([min_distances,\
                                 np.squeeze(min_comb_id)],axis=0)
            result = np.expand_dims(result,0)
        else:
            print(t1,t0)
            print(min_distances, min_comb_id)
            
        df = pd.DataFrame(result, columns=['Min_Distance','ID2','ID1'])        
        df['nodup']= ~df.sort_values(by=['Min_Distance']).duplicated(subset=['ID1'])
        df['trackID'] = df['ID1'].where((df.nodup)&(df.Min_Distance<max_d),df['ID2'])
        # print(df)
        nandframe = pd.DataFrame(np.empty((len(lat0)-len(df),len(df.columns)))*np.nan,\
                        columns=df.columns,index=np.arange(len(df),len(lat0),1))
        df = pd.concat([df,nandframe])
        closest_object = df['Min_Distance'].values
#         print(closest_object)
        track_ids = df['trackID'].values
    
    return closest_object,track_ids


def track_along_time(objects,trackby,max_distance):
    detected_objects = objects.transpose('time','index').dropna(dim='index',how='all')
    s = np.shape(detected_objects.label)
    track_ids = detected_objects.label*0+np.reshape(np.arange(1,s[0]*s[1]+1),s)
    detected_objects['track_ids'] = track_ids
    shifted_objects = detected_objects.shift(time=1)

    s = np.shape(detected_objects[trackby[0]])
    lat0 = detected_objects[trackby[0]].values
    lon0 = detected_objects[trackby[1]].values
    id0 = detected_objects['track_ids'].values
    distance = np.empty(s)*np.nan
    ids = np.empty(s)*np.nan
    
    # print(lat0,lon0,id0)
    # print(s)
    # print(h)
    
    for i in tqdm(range(1,s[0])):
        # print(i)
        distance[i],id0[i] = closest_obj_distance(lat0[i],lon0[i],lat0[i-1],lon0[i-1],\
                                                  id0[i],id0[i-1],max_distance)

    detected_objects['track_ids'] = detected_objects['track_ids']*0+id0
    detected_objects['travelled_distance'] = detected_objects['track_ids']*0+distance
#     print(detected_objects['track_ids']*0+id0,detected_objects['track_ids']*0+distance)
    return detected_objects

