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
print(os.getcwd())
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
        self.image_ids = list()
        self.image_corners = list()
        self.id = id

    def plot_SFKB(self, ax):
        '''
        Plots a top-down view of the building graph in UTM coordinates
        '''
        ax.set_aspect('equal')
        for i, edge in enumerate(self.edges):
            for ind in edge:
                plt.plot([self.utm_corners[i, 0], self.utm_corners[ind, 0]],
                         [self.utm_corners[i, 1], self.utm_corners[ind, 1]],
                          zorder=0, color='darkorange')
        ax.scatter(*self.utm_corners.T[:2], color='red', s=15)

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
                utm_corners[i] = np.array(item.split(','), dtype='float')

            for i, item in enumerate(keys):
                edge_value = np.array(building['nodes'][item])
                edge = np.empty(edge_value.shape[0])
                
                for i, edg in enumerate(edge_value):
                    edge[i] = np.flatnonzero(np.prod(edg == utm_corners, axis=1))
                edges.append(edge.astype(int))
            
            buildings.append(Building(utm_corners, edges, building['building_id']))
    return buildings

def get_image_ids(buildings, sosi_path = '../data/sosi.txt'):
    '''
    Gets the image coverage areas from the sosi file, 
    and determins wether buildings are in the coverage area.
    Image ids in which a building appears are added to building.building_ids before the instance is returned.
    '''
    copy = False
    rect = list()
    rects = list()

    with open(sosi_path, 'r') as infile: 
        for line in infile:
            if line.find('..OBJTYPE Bildegrense') != -1:
                copy = True
                rect = list()

            if line.find('.FLATE') != -1:
                rects.append(rect)
                copy = False
            
            if copy==True:
                rect.append(line.split(','))
            

    rects = rects[1:]  
    coverage = list()

    for i, rect in enumerate(rects):
        l_arr = np.empty((len(rect[2:]), 2))
        for j, item in enumerate(rect[2:]):
            line = item[0].replace(r'\n', ' ').split(' ')
            line[1] = line[1].split('\n')[0]
            line = np.array(line, dtype='int64')
            l_arr[j] = line

        inds = (np.argmin(l_arr[:,0]), np.argmin(l_arr[:,1]),np.argmax(l_arr[:,0]), np.argmax(l_arr[:,1]))
        poly = Polygon(l_arr[inds, ::-1])

        for building in buildings:
            if shrink_polygon(poly, factor=0.01).contains(Point(building.utm_mean[:-1])):
                building.image_ids.append(i)
        coverage.append(poly)

    for i, building in enumerate(buildings):
        if len(building.image_ids)<1:
            del building[i]
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
    '''
    header = ['ID', 'x', 'y', 'z', 'rx', 'ry', 'rz']
    metadata = pd.read_table(metadata_path, comment='#', delim_whitespace=True, names=header, usecols=[0, 1, 2, 3, 5, 6, 7])
    metadata[['rx', 'ry', 'rz']] = metadata[['rx', 'ry', 'rz']].apply(np.deg2rad)

    focal_length = int(100.5*1e-3/4e-6) # image coordinates
    ny, nx = 26460, 17004
    ppa = np.array((nx/2 + int(0.08*1e-3/4e-6), ny/2)) # image coordinates

    return [metadata.loc[image_id][i] for i in ['x', 'y', 'z', 'rx', 'ry', 'rz']].append([focal_length, ppa])




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

def UTM_to_image(utm_coords, image_id):
    '''
    converts coordinates of a single building to the image coordinates 
    of a single aerial image.
    '''
    nx = 17004
    CM = camera_matrix(image_id)
    utm_coords_hom = np.vstack([utm_coords, np.ones((1, utm_coords.shape[-1]))])
    im_coords = CM@utm_coords_hom
    im_coords = im_coords[:-1, :]/im_coords[-1]
    im_coords[0] = -im_coords[0] + nx 
    return im_coords


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