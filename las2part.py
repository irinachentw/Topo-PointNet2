import numpy as np
#import laspy
from copy import deepcopy
#import pcl
#import open3d as o3d
#import geopandas as gp
#import matplotlib.pyplot as plt
from osgeo import ogr
import datetime
import multiprocessing as mp

'''
# read las and write as pcd to keep the cloud format and rgb bands
def las2pcd(las_path,save_path):
    las = laspy.read(las_path)
    inFile = np.vstack((las.x, las.y, las.z, las.intensity)).transpose()
    rgb =  np.vstack((las.red, las.green, las.blue)).transpose()
    cloud = pcl.PointCloud.PointXYZRGBA().from_array(np.array(inFile, dtype=np.float32),rgb)
    address=save_path+now+".pcd"
    pcl.io.savePCDFileASCII(address,cloud)
    return address
'''

'''
# read the path of pcd
def read_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points
'''

# read the txt file and write as array
def read_txt(file_path):
    file_txt = open(file_path,'rb')
    lidar = np.loadtxt(file_txt)
    #print('lidar:',lidar)
    print('liadr.shape',lidar.shape)
    points1 = lidar[:,:3]
    points2 = lidar[:,4:7]
    points = np.hstack((points1,points2))
    print('points shape:', points.shape)
    return points


# Determine if a point is within a polygon
def is_in_poly(p,poly):
    """
    param p:[x,y,z]
    param poly:[[x,y],[x,y]....]
    return 
    """
    px,py,pz,pa,pb,pc = p
    is_in = False
    for i, corner in enumerate(poly):
        if i + 1< len(poly):
            next_i = i + 1
        else:
            0
        x1,y1 = corner
        x2,y2 = poly[next_i]
        # if the point is on vertex:
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):
            is_in = True
            break
        if min(y1,y2) < py <= max(y1,y2):
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:
                is_in = True
                break
            elif x > px:
                is_in = not is_in
    return is_in





# extract_targetpoints
def extract_targetpoints(save_path,points,feas):
    print("the length of feas:",len(feas))
    with open(save_path,'w', encoding='utf-8') as f:
        for i, point in enumerate(points):
        #print("processing point No.%f"%(i+1))
            for ID,poly in feas.items():
                #print(ID,poly)
                #print("the ID of %f"%ID)
                is_in = is_in_poly(point,poly)
                if is_in == True:
                    x = point[0]
                    y = point[1]
                    z = point[2]
                    a = point[3]
                    b = point[4]
                    c = point[5]
                    ex_points = str(x) + " " +str(y) + " " + str(z) + " " + str(a) +\
                        " "+ str(b) + " " +str(c) + " " + str(ID) + " "  '\n'
                    f.write(ex_points)
                    print("the No.%f point has been saved to No.%f group"%(i+1,ID))
                else:
                    pass
    f.close()

# read geojson file:
def read_geojson(geojson_path):
    ds = ogr.Open(geojson_path)
    #print("ds.type",type(ds))
    lyr = ds.GetLayer(0)
    print(lyr.GetFeatureCount())
    features = dict()
    for fea in lyr:
        ID = fea.GetField('Id')
        #print("ID:",ID)
        # get the geomnetry attribute
        geom = fea.geometry()
        # get the outer border of polygon
        ring = geom.GetGeometryRef(0)
        # get the coordinates
        coords = ring.GetPoints()
        # read the x & y coordinate individually
        x,y = zip(*coords)
        features[ID] = coords
        #print(coords)
    #print(features)
    return features

    '''
    with open(geojson_path) as f:
        features = geojson.load(f,cls = geojson.GeoJSONDecoder)
    # The export result is a dictionary 
    return features
    '''

# extract target points and write them to txt


if __name__ == '__main__':
    #las_path = 'D:/Capstone_Pointnet2/ProcessedData/ProcessedData/AZ_3699Cloud.las'
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    save_path = 'D:/Capstone_Pointnet2/ProcessedData/Newpoints_412n_3699'+('%s.txt'%timestr)
    file_path = 'D:/Capstone_Pointnet2/ProcessedData/BXY_412n_3699.txt'
    points = read_txt(file_path)
    points = points.tolist()
    feas = read_geojson('D:/Capstone_Pointnet2/ProcessedData/global_geojson.json')
    # using multiple cores to run
    start_t = datetime.datetime.now()
    num_cores = int(mp.cpu_count())
    print("The local PC has %d cores"%num_cores)
    #create process pool
    pool = mp.Pool(num_cores)
    #result = pool.apply_async(extract_targetpoints,args=(save_path,points,feas))
    proc = mp.Process(target=extract_targetpoints,args=(save_path,points,feas))
    proc.start()
    proc.join()
    end_t = datetime.datetime.now()
    elapsed_min = float((end_t - start_t).total_seconds()/60)
    print("multi-progress consumes a total %f mins"%elapsed_min)

