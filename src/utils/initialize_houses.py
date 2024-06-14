import json
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely import geometry
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from src.house import House
import os
import PIL
import sys

PIL.Image.MAX_IMAGE_PIXELS = None
header = ['ID', 'x', 'y', 'z', 'height2', 'rx', 'ry', 'rz',
          'pos1', 'pos2', 'pos3', 'att1', 'att2', 'att3', 'week', 'ToW', 'n_sat',
          'PDOP', 'lat', 'long', 'height']
print(os.getcwd())
aerial_photo = plt.imread(r'testing/ims/aerial_photos/RGB/001.tif')
ny, nx = aerial_photo.shape[:-1]
metadata = pd.read_table(r'testing/ims/aerial_photos/GNSSINS/EO_V355_TT-14525V_20210727_1.txt', comment='#', delim_whitespace=True, names=header)
metadata[['rx', 'ry', 'rz']] = metadata[['rx', 'ry', 'rz']].apply(np.deg2rad)
focal_length = int(100.5*1e-3/4e-6) # image coordinates
ppa = np.array((nx/2 + int(0.08*1e-3/4e-6), ny/2)) # image coordinates
'''
class House:
    def __init__(self, utm_coords, id):
        self.utm_coords = utm_coords
        self.utm_mean = np.mean(utm_coords, axis=1)
        self.image_ids = list()
        self.image_coords = None
        self.id = i
'''

def shrink_polygon(my_polygon, factor=0.10):
    xs = list(my_polygon.exterior.coords.xy[0])
    ys = list(my_polygon.exterior.coords.xy[1])
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    min_corner = geometry.Point(min(xs), min(ys))
    center = geometry.Point(x_center, y_center)
    shrink_distance = center.distance(min_corner)*factor
    return my_polygon.buffer(-shrink_distance)

infile = open(r'testing/ims/geojson2.json')
labels = json.load(infile)['features']

houses = list()
for label in labels[:20]:
    keys = list(label['nodes'].keys()) 
    house = np.empty((3, len(keys)))
    edges = list()
    for i, item in enumerate(keys):
        edges.append(np.swapaxes(np.array(label['nodes'][item]), 0, 1))
        house[:, i] = np.array(item.split(','), dtype='float')
    houses.append(House(house, edges, label['building_id']))

copy = False
rect = list()
rects = list()
with open(r'testing/ims/sosi.txt', 'r') as infile: 
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
l_arrs = list()
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
    for house in houses:
        if shrink_polygon(poly, factor=0.01).contains(Point(house.utm_mean[:-1])):
            house.image_ids.append(i)
    coverage.append(poly)

for i, house in enumerate(houses):
    if len(house.image_ids)<1:
        del houses[i]

def RotationMatrix(rx, ry, rz):
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

def UTM_to_image(X, C, x0, R, f):
    '''
    Calculates image coordinates using collinearity equations.

    X: Object positions in external coordinates. expects RGB dim as first shape dim, i.e. for N nodes, (3, N)
    C: Camera position in external coordinates
    x0: Focal point in image coordinates
    R: Perspective projection matrix from RotationMatrix()
    f: Focal length in image coordinates
    '''
    d = X-C.reshape(3, 1)
    denom = (R[0, 2]*d[0] + R[1, 2]*d[1] + R[2, 2]*d[2])
    img_coord = np.empty((2, X.shape[1]))
    img_coord[0] = x0[0] - f*(R[0, 0]*d[0] + R[1, 0]*d[1] + R[2, 0]*d[2])/denom
    img_coord[1] = ny - x0[1] + f*(R[0, 1]*d[0] + R[1, 1]*d[1] + R[2, 1]*d[2])/denom
    return img_coord

def Camera_matrix(image_id):
    cx, cy, cz, rx, ry, rz =  [metadata.loc[image_id][i] for i in ['x', 'y', 'z', 'rx', 'ry', 'rz']]
    R = RotationMatrix(rx, ry, rz) #camera rotation in w.r.t UTM
    C = np.array((cx, cy, cz)).reshape(-1, 1) #camera postion in UTM
    extrinsic_matrix = np.vstack([np.hstack([R.T, -R.T@C]),np.array((0, 0, 0, 1))])
    intrinsic_matrix = np.array(((focal_length, 0, ppa[0], 0),(0, focal_length, ppa[1], 0),(0, 0, 1, 0)))

    return intrinsic_matrix@extrinsic_matrix

def UTM_to_image2(utm_coords, image_id):
    CM = Camera_matrix(image_id)
    utm_coords_hom = np.vstack([utm_coords, np.ones((1, utm_coords.shape[-1]))])
    im_coords = CM@utm_coords_hom
    im_coords = im_coords[:-1, :]/im_coords[-1]
    im_coords[0] = -im_coords[0] + nx 
    return im_coords

def transform_house(house):
    image_corners = list()
    image_edges = list()
    for id in house.image_ids:
        image_corners.append(UTM_to_image2(house.utm_coords, id))
        edge_ = list()
        for edge in house.utm_edges:
            edge_.append(UTM_to_image2(edge, id))
        image_edges.append(edge_)
    
    return image_corners, image_edges 

image_ids_in_use = list()

for house in houses:
    for id in house.image_ids:
        if id not in image_ids_in_use:
            image_ids_in_use.append(id)
    
    corners, edges = transform_house(house)
    house.image_coords.append(corners)
    house.image_edges.append(edges)

with open(r'testing/objects/houses.pickle', 'wb') as file:
    pickle.dump(houses, file)
with open(r'testing/objects/coverage.pickle', 'wb') as file:
    pickle.dump(coverage, file)
with open(r'testing/objects/image_ids.pickle', 'wb') as file:
    pickle.dump(image_ids_in_use, file)
            