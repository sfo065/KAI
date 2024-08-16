"""
Functions for transforming data between 3D UTM and 2D image coordinates.
"""
import json
import numpy as np
from shapely import geometry
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import PIL
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import random
import cv2
PIL.Image.MAX_IMAGE_PIXELS = None

class Building:
    """
    Class for storing building information for a single building.
    n_corners : number of corners for the building
    n_connections: number of connections to the corner node 
    utm_corners: (n_corners, 3),  UTM corner coordinates
    edges: list of length n_corners with entries of shape (n_connections, 3), containing indices of connected nodes. 
    id: Building ID from SFKB database
    image_ids: list of image ids in which the building appears
    image_corners: (n_corners, 2),  Building corners in image coordinates
    """
    def __init__(self, utm_corners,edges, id):
        self.utm_corners = utm_corners
        self.edges = edges
        self.utm_mean = np.mean(utm_corners, axis=0)
        self.image_corners = dict() #  {image id: image coordinate corners}
        self.cutout_corners = dict()
        self.id = id

    def plot_SFKB(self, ax):
        '''
        Plots a top-down view of the building graph in UTM coordinates
        '''
        ax.set_aspect('equal')
        for i, edge in enumerate(self.edges):
            for ind in edge:
                ax.plot([self.utm_corners[i, 0], self.utm_corners[ind, 0]],
                         [self.utm_corners[i, 1], self.utm_corners[ind, 1]],
                          zorder=0, color='darkorange')
        ax.scatter(*self.utm_corners.T[:2], color='red', s=15)
    
    def plot_image(self, axs):
        '''
        Plots transformed building graphs
        '''
        for i, corners in enumerate(self.image_corners.values()):
            for j, edge in enumerate(self.edges):
                for ind in edge:
                    axs[i].plot([corners[j, 0], corners[ind, 0]],
                                [corners[j, 1], corners[ind, 1]],
                                zorder=0, color='darkorange')
            axs[i].scatter(*corners.T, color='red', s=15)
    
    def plot_cutout(self, plot_graph=False):
        '''
        Plots transformed building graphs in the cutout frame
        '''
        n = len(self.cutout_corners)
        if n < 5:
            fig, axs = plt.subplots(1, n, figsize=(n*3, 3))    
        else:
            fig, axs = plt.subplots((n-1)//5 + 1, 5, figsize = (15, ((n-1)//5 + 1)*3))
        fig.suptitle(f'Building {self.id}')
        for ax in axs.flatten():
            ax.axis('off')

        for i, key in enumerate(self.cutout_corners):
            corners = self.cutout_corners[key]
            im =  plt.imread(f'../results/cutouts/{self.id}_{key}.png')
            axs.flatten()[i].imshow(im)
            axs.flatten()[i].set_title(int(key.split('_')[-1]))

            if plot_graph:
                for j, edge in enumerate(self.edges):
                    for ind in edge:
                        axs[i].plot([corners[j, 0], corners[ind, 0]],
                                    [corners[j, 1], corners[ind, 1]],
                                    zorder=0, color='darkorange')
                        
                axs[i].scatter(*corners.T, color='red', s=12)
        plt.show()
        
def shrink_polygon(image_polygon, factor=0.10):
    """
    Helper function for Extract_buildings 
    Shrinks polygon for building extraction to avoid inculding buildigs
    that are intersected by the image border.
    polygon: image coverage rectangular polygon - shapely shapefile.
    """
    xs = list(image_polygon.exterior.coords.xy[0])
    ys = list(image_polygon.exterior.coords.xy[1])
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    min_corner = geometry.Point(min(xs), min(ys))
    center = geometry.Point(x_center, y_center)
    shrink_distance = center.distance(min_corner)*factor
    return image_polygon.buffer(-shrink_distance)

def load_sfkb(json_path='../data/SFKB/geojson.json', max_buildings=None):
    '''
    Loads buildings from SFKB json file and stores them as instances of the Building class.
    Outputs a list of Building instances
    '''
    SFKB_buildings = json.load(open(json_path))['features']
    if max_buildings:
        SFKB_buildings = SFKB_buildings[:max_buildings]
    buildings = list()
    spent_ids = list()
    for building in SFKB_buildings:
        if building['building_id'] not in spent_ids: # Ignore duplicate IDs
            spent_ids.append(building['building_id'])
            keys = list(building['nodes'].keys()) 
            utm_corners = np.empty((len(keys), 3))
            edges = list()

            for i, item in enumerate(keys):
                utm_corners[i] = np.array(item.split(','), dtype='float') # collect corner coords

            for item in keys:
                edge_value = np.array(building['nodes'][item]) # collect edge coords
                edge = np.empty(edge_value.shape[0])
                
                for i, edg in enumerate(edge_value):
                    edge[i] = np.flatnonzero(np.prod(edg == utm_corners, axis=1)) # convert coords to indices
                edges.append(edge.astype(int))
            
            buildings.append(Building(utm_corners, edges, building['building_id'])) # Intialize building objects with utm corners, edges indices and building ids

    return buildings

def get_image_ids(buildings, dict_dump_path='../results', sosi_path = '../data/sosi.txt'):
    '''
    Gets the image coverage areas from the txt file converted from sosi, 
    and determines whether buildings are in the coverage area.
    Image ids in which a building appears are added to building.building_ids before the instance is returned.
    A dictionary representing which houses are present in each image is dumped at the dir given by "dict_dump_path".
    this has format {Image_id: list of buildings}
    '''
    copy = False
    rect = list()
    rects = list()
    ids = list()

    with open(sosi_path, 'r') as infile: # Parse coverage metadata file
        for line in infile:
            if line.startswith('..OBJTYPE Bildegrense'):
                copy = True
                rect = list()

            if line.startswith('.FLATE'):
                rects.append(rect)
                copy = False
            
            if copy==True:
                rect.append(line.split(','))
            
            if line.startswith('...BILDEFILRGB'):
                ids.append(line.split('"')[1].split('.')[0])
        rects.append(rect[:-1])

    rects = rects[1:]
    
    coverage = list()
    for i, rect in enumerate(rects): # Cleanup and conversion to int
        l_arr = np.empty((len(rect[2:]), 2)) # array to hold points delineating the coverage area border
        
        for j, item in enumerate(rect[2:]):
            line = item[0].replace(r'\n', ' ').split(' ')
            line[1] = line[1].split('\n')[0]
            line = np.array(line, dtype='int64')
            l_arr[j] = line

        inds = (np.argmin(l_arr[:,0]), np.argmin(l_arr[:,1]),np.argmax(l_arr[:,0]), np.argmax(l_arr[:,1])) # indices of rectangle corners
        poly = shrink_polygon(Polygon(l_arr[inds, ::-1]), factor=0.005) # coverage polygon in UTM coords for an aerial image

        for building in buildings:
            counter = 0
            for corner in building.utm_corners:
                if not poly.contains(Point(corner)):
                    counter +=1
            if counter == 0:
                if ids[i] not in building.image_corners.keys():
                    building.image_corners[ids[i]] = list()

        coverage.append(poly)

    _buildings = list()
    for i, building in enumerate(buildings): # deletes building objects if the building is not in an image
        if len(building.image_corners) > 1 :
            _buildings.append(building)  
    buildings = _buildings

    if not os.path.exists(dict_dump_path):
        os.makedirs(dict_dump_path)

    return buildings



def rotation_matrix(rx, ry, rz):
    '''
    Rturns the rotation matrix for the rotation vector(rx, ry, rz)
    '''
    R1 = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),np.cos(rx)]])
    
    R2 = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    
    R3 = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    
    return R1@R2@R3 


def get_camera_properites(image_id, metadata_path='../data/Aerial_Photos/GNSSINS/EO_V355_TT-14525V_20210727_1.txt'):
    '''
    Fetches position and rotation of camera from the imaging project metadata file.
    Also returns the focal length and principal point coordinates 
    Output: (x, y, z, rx, ry, rz, focal_length, ppa) 
    '''
    header = ['ID', 'x', 'y', 'z', 'rx', 'ry', 'rz']
    metadata = pd.read_table(metadata_path, comment='#', delim_whitespace=True, names=header, usecols=[0, 1, 2, 3, 5, 6, 7])
    metadata[['rx', 'ry', 'rz']] = metadata[['rx', 'ry', 'rz']].apply(np.deg2rad)
    focal_length = int(100.5*1e-3/4e-6) # image coordinates
    ny, nx = 26460, 17004
    ppa = np.array((nx/2 + int(0.08*1e-3/4e-6), ny/2)) # image coordinates
    image_id_int = int(image_id[-3:])-1
    return [metadata.loc[image_id_int][i] for i in ['x', 'y', 'z', 'rx', 'ry', 'rz']] + [focal_length, ppa]




def camera_matrix(image_id):
    '''
    Calculates the camera matrix for the camera used in the image 
    given by the image id. 
    '''
    cx, cy, cz, rx, ry, rz,focal_length, ppa  =  get_camera_properites(image_id)
    R = rotation_matrix(rx, ry, rz) #camera rotation in  UTM
    C = np.array((cx, cy, cz)).reshape(-1, 1) #camera postion in UTM
    extrinsic_matrix = np.vstack([np.hstack([R.T, -R.T@C]),np.array((0, 0, 0, 1))])
    intrinsic_matrix = np.array(((focal_length, 0, ppa[0], 0),(0, focal_length, ppa[1], 0),(0, 0, 1, 0)))

    return intrinsic_matrix@extrinsic_matrix


def utm_to_image(utm_coords, image_id):
    '''
    converts coordinates of a single building to the image coordinates 
    of a single aerial image.

    '''
    nx = 17004
    CM = camera_matrix(image_id)
    utm_coords_hom = np.vstack([utm_coords.T, np.ones((1, utm_coords.shape[0]))])
    im_coords = CM@utm_coords_hom
    im_coords = im_coords[:-1, :]/im_coords[-1]
    im_coords[0] = -im_coords[0] + nx 
    return im_coords.T


def transform_utm_buildings(buildings):
    '''
    Transforms utm coordinates of building corners from utm to image coordinates
    in frames of every aerial photo in which the building appears
    '''
    for building in buildings:
        for image_id in building.image_corners.keys():
            building.image_corners[image_id] = utm_to_image(building.utm_corners, image_id)
    return buildings


def dump_cutouts_and_objects(buildings, image_ids,cutout_path='../results/cutouts', object_path='../results/objects', nudge=np.array((13,20)), pad=15, image_size=256):
    '''
    Extracts building cutouts from an list of aerial images referenced by the name of the image file.
    Assumes pointer dictionary at ../results/image_to_building_pointers.pickle, 
    aerial photos in ../data/aerial_photos/RGB/ and building objects in ../results/building_objects.
    dumps images and objects at the paths given in the keyword arguments

    '''

    if not os.path.exists(cutout_path):
        os.makedirs(cutout_path)

    if not os.path.exists(object_path):
        os.makedirs(object_path)

    for image_id in image_ids:
        aerial_photo = plt.imread(f'../data/aerial_photos/RGB/{image_id}.tif')
        for i, building in enumerate(buildings):
            if image_id in building.image_corners.keys():
                corners = building.image_corners[image_id]
                    

                loc_min = np.min(corners, axis=0).astype(int) 
                loc_max = np.max(corners, axis=0).astype(int)
                
                x_ext, y_ext = loc_max[0]-loc_min[0], loc_max[1]-loc_min[1]

                if x_ext > y_ext: # Pad shortest axis
                    sq_pad = np.array((0, (x_ext-y_ext)/2)).astype(int)
                else:
                    sq_pad = np.array(((y_ext-x_ext)/2, 0)).astype(int)

                

                im =  aerial_photo[loc_min[1]-pad-sq_pad[1]:loc_max[1]+2*pad +sq_pad[1]+ nudge[1], loc_min[0]-pad - sq_pad[0]:loc_max[0]+2*pad + nudge[0]+sq_pad[0]] # crop cutout
                graph_cutout = ((corners.T-loc_min.reshape(-1, 1)+pad+nudge.reshape(-1, 1) + sq_pad.reshape(-1, 1))*image_size/np.array(im.shape[:2]).reshape(2, 1)).T # transform graph to cutout frame 
                buildings[i].cutout_corners[image_id] = graph_cutout
            
                d_min = np.min(im.shape[:2])
                try:    
                    im = cv2.resize(im[:d_min, :d_min], (image_size, image_size))
                    plt.imsave(f'{cutout_path}/{building.id}_{image_id}.png', im)
                    with open(f'{object_path}/{building.id}.pickle', 'wb') as handle:
                        pickle.dump(building, handle)
                except cv2.error:
                    del buildings[i]
                    pass
                    

        print(f'image {image_id} done')

def triangulate_Npts(pt2d_CxPx2, P_Cx3x4):
    """
    Triangulate multiple 3D points from two or more views by DLT.
    """

    assert pt2d_CxPx2.ndim == 3
    assert P_Cx3x4.ndim == 3
    Nc, Np, _ = pt2d_CxPx2.shape
    assert P_Cx3x4.shape == (Nc, 3, 4)

    # P0 - xP2
    x = P_Cx3x4[:,0,:][:,None,:] - np.einsum('ij,ik->ijk', pt2d_CxPx2[:,:,0], P_Cx3x4[:,2,:])
    # P1 - yP2
    y = P_Cx3x4[:,1,:][:,None,:] - np.einsum('ij,ik->ijk', pt2d_CxPx2[:,:,1], P_Cx3x4[:,2,:])

    Ab = np.concatenate([x, y])
    Ab = np.swapaxes(Ab, 0, 1)
    assert Ab.shape == (Np, Nc*2, 4)

    A = Ab[:,:,:3]
    b = - Ab[:,:,3]
    AtA = np.linalg.pinv(A)

    X = np.einsum('ijk,ik->ij', AtA, b)
    return X