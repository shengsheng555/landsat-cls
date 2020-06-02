# -*- coding: utf-8 -*-

# !pip install rasterio
# !pip install shapely

import os, shutil
import sys
import time
import numpy as np
import pandas as pd
from glob import glob
import rasterio as rio
from rasterio import warp
import zipfile
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

"""Download data"""

NLCD_PATH = './NLCD/'
LANDSAT_PATH = './Landsat8data/'
L8_NLCD_PATH = './L8_NLCD/'
for PATH in [NLCD_PATH, LANDSAT_PATH, L8_NLCD_PATH, gdrive_dir]:
    os.makedirs(PATH, exist_ok=True)

L8_NLCD_file = os.path.join(L8_NLCD_PATH,'L8_NLCD_Site_ID_{}_LARGE.TIF')

# unit: m
image_width = 3840
image_height = 3840
res = 30    

# get input sites, coordinates in LatLong format 
sites = pd.read_csv('./sites.csv', header=0)

# Download and unzip NLCD dataset
# This may take more than 5 minutes
t_start = time.time()

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

print('Downloading NLCD...')
download_url('https://s3-us-west-2.amazonaws.com/mrlc/NLCD_2016_Land_Cover_L48_20190424.zip',
             os.path.join(NLCD_PATH, 'NLCD.zip'))     

print('Unzipping NLCD...')
with zipfile.ZipFile(os.path.join(NLCD_PATH, 'NLCD.zip'), 'r') as zip_ref:
    zip_ref.extractall(NLCD_PATH)       


# Reproject NLCD. Takes ~15min

NLCD_FILE = 'NLCD_2016_Land_Cover_L48_20190424.img'
NLCD_reprojected = 'NLCD_reprojected.tif'

from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproject_ras(inpath, outpath, new_crs):
    dst_crs = new_crs # CRS for web meractor 

    with rio.open(inpath) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rio.open(outpath, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

inpath = os.path.join(NLCD_PATH, NLCD_FILE)
outpath = os.path.join(NLCD_PATH, NLCD_reprojected)


print('Reprojecting...')
reproject_ras(inpath = inpath, 
             outpath = outpath, 
             new_crs = 'EPSG:4326') # match Lon/Lat CRS

t_end = time.time()

print ("Time elapsed: {} s".format(t_end - t_start))

# Form downloading bulk frame

t_start = time.time()

# Read scenes frame from Amazon s3
s3_scenes = pd.read_csv('http://landsat-pds.s3.amazonaws.com/c1/L8/scene_list.gz', compression='gzip')

# conversion function: LatLon to PathRow 
from get_wrs import ConvertToWRS
conv = ConvertToWRS(shapefile='./WRS2_descending/WRS2_descending.shp')
# usage
# conv.get_wrs(25.411914, -80.496381)  # conv.get_wrs(lat, lon)  


def cal_path_row(site):
    '''Calculate Path/Row for each site'''
    path_row = conv.get_wrs(site['Latitude'], site['Longitude'])[0]  #conv.get_wrs(lat, lon)  
    site['path'] = path_row['path']
    site['row'] = path_row['row']
    return site

sites = sites.apply(lambda r : cal_path_row(r), axis=1)


# Form a bulk of L8 products to download

def form_bulk(bulk_list, sites, s3_scenes):  

    # Iterate over sites to select a bulk of scenes
    for index, site in sites.iterrows():

        # check if the site is covered in previous scenes
        covered = False
        for scene in bulk_list:
            if (scene.path == site.path and scene.row == site.row):              
                sites.loc[index, 'productId'] = scene.productId            
                covered = True  
                
        if not covered:

            # Filter the Landsat S3 table for images matching path/row, cloudcover and processing state.
            scenes = s3_scenes[(s3_scenes.path == site.path) & (s3_scenes.row == site.row) & 
                              (s3_scenes.cloudCover <= 5) & 
                              (~s3_scenes.productId.str.contains('_T2')) &
                              (~s3_scenes.productId.str.contains('_RT')) &
                              (s3_scenes.acquisitionDate.str.contains('2016-'))]
            # print(' Found {} images\n'.format(len(scenes)))

            # If any scene exists, select the one that have the minimum cloudCover.
            if len(scenes):
                scene = scenes.sort_values('cloudCover').iloc[0]        
                sites.loc[index, 'productId'] = scene.productId 

                # Add the selected scene to the bulk download list.
                bulk_list.append(scene)      
            else:
                print('cannot find a scene for the site ID={}'.format(site.ID))  
    return bulk_list              

bulk_list = []

bulklist = form_bulk(bulk_list, sites, s3_scenes)  

# Check selected images
bulk_frame = pd.concat(bulk_list, 1).T
bulk_frame.head()

t_end = time.time()
print ("Time elapsed: {} s".format(t_end - t_start))

def download_and_stack_product(row_bulk_frame):
    '''   Download and stack Landsat8 bands   '''
    # Print some the product ID
    print('\n', 'Downloading L8 data:', row_bulk_frame.productId)
    # print(' Checking content: ', '\n')

    # Request the html text of the download_url from the amazon server. 
    # download_url example: https://landsat-pds.s3.amazonaws.com/c1/L8/139/045/LC08_L1TP_139045_20170304_20170316_01_T1/index.html
    response = requests.get(row_bulk_frame.download_url)

    # If the response status code is fine (200)
    if response.status_code == 200:

        # Import the html to beautiful soup
        html = BeautifulSoup(response.content, 'html.parser')

        # Create the dir where we will put this image files.
        entity_dir = os.path.join(LANDSAT_PATH, row_bulk_frame.productId)
        os.makedirs(entity_dir, exist_ok=True)

        # Second loop: for each band of this image that we find using the html <li> tag
        for li in html.find_all('li'):

            # Get the href tag
            file = li.find_next('a').get('href')

            # print('  Downloading: {}'.format(file))

            response = requests.get(row_bulk_frame.download_url.replace('index.html', file), stream=True)

            with open(os.path.join(entity_dir, file), 'wb') as output:
                shutil.copyfileobj(response.raw, output)
            del response

    # Stack bands 1-7,9

    # Obtain the list of bands 1-7,9
    entity_dir = os.path.join(LANDSAT_PATH, row_bulk_frame.productId)     
    landsat_bands = glob(os.path.join(entity_dir, '*B[0-7,9].TIF'))       
    landsat_bands.sort()

    # Read metadata of first file
    with rio.open(landsat_bands[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    meta.update(count = len(landsat_bands))

    # Read each layer and write it to stack
    stackfile = os.path.join(entity_dir, row_bulk_frame.productId+'_Stack.TIF')
    # print('Stacking L8 bands: {}'.format(row_bulk_frame.productId))
    with rio.open(stackfile, 'w', **meta) as dst:
        for id, layer in enumerate(landsat_bands, start=1):
            with rio.open(layer) as src1:
                dst.write_band(id, src1.read(1))

    # Reprojecting
    # print('Reprojecting...')
    reproject_ras(inpath = stackfile, 
                  outpath = stackfile, 
                  new_crs = 'EPSG:4326')      
    
          


def crop_rectangle(lat, lon, image_width, image_height, res, in_file, out_file = './out.TIF'):
    '''crop a rectangle around a point in Lat/Lon CRS'''

    with rio.open(in_file) as src:

        # CRS transformation
        src_crs = rio.crs.CRS.from_epsg(4326) # latlon crs
        dst_crs = src.crs # current crs
        xs = [lon] 
        ys = [lat] 
        coor_transformed = warp.transform(src_crs, dst_crs, xs, ys, zs=None)
        coor = [coor_transformed[0][0], coor_transformed[1][0]]
        # print('coor: ', coor )

        # Returns the (row, col) index of the pixel containing (x, y) given a coordinate reference system
        py, px = src.index(coor[0], coor[1])

        # Build window with right size
        p_width = image_width//res
        p_height = image_height//res        
        window = rio.windows.Window(px - p_width//2, py - p_height//2, p_width, p_height)
        # print('window: ', window)

        # Read the data in the window
        clip = src.read(window=window)
        # print('clip: ', clip)

        # write a new file
        meta = src.meta
        meta['width'], meta['height'] = p_width, p_height
        meta['transform'] = rio.windows.transform(window, src.transform)

        with rio.open(out_file, 'w', **meta) as dst:
            dst.write(clip)                

def stack_rasters(raster_list, outfile):    

    # Read metadata of a certain file
    with rio.open(raster_list[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    num_layers = 0
    for f in raster_list:
        with rio.open(f) as src:
            num_layers += src.count
    meta.update(count = num_layers)

    # Read each layer and write it to stack
    # print('  Stacking site_ID: {}'.format(site.ID))
    with rio.open(outfile.format(site.ID), 'w', **meta) as dst:
        stack_band_offset = 0
        for id, f in enumerate(raster_list, start=1):
            with rio.open(f) as src:
                for band_f in range(1, src.count+1):
                    # Cast dtype, matching L8
                    band_to_write = src.read(band_f).astype(np.uint16)
                    dst.write_band(stack_band_offset+band_f, band_to_write)    
                stack_band_offset += src.count

t_start = time.time()

# For each productID
for i, row_bulk_frame in bulk_frame.iterrows(): 
    download_and_stack_product(row_bulk_frame)
    # For each site with specified productID

    # Crop each site
    # print('Cropping sites in {}...'.format(row_bulk_frame.productId))
    for index, site in sites.iterrows():
        if site.productId == row_bulk_frame.productId:          
              L8_file = os.path.join(LANDSAT_PATH, 'L8_Site_ID_{}_LARGE.TIF'.format(site.ID))
              NLCD_file = os.path.join(NLCD_PATH, 'NLCD_Site_ID_{}_LARGE.TIF'.format(site.ID))

              # Crop L8
              in_dir = os.path.join(LANDSAT_PATH, site.productId)
              in_file = os.path.join(in_dir, site.productId+'_Stack.TIF') 
              crop_rectangle(site.Latitude, site.Longitude, image_width, 
                             image_height, res, in_file, L8_file)
              # Crop NLCD
              in_file = os.path.join(NLCD_PATH, NLCD_reprojected) 
              crop_rectangle(site.Latitude, site.Longitude, image_width, 
                             image_height, res, in_file, NLCD_file) 
              
              # Stack L8 and NLCD
              raster_list = [L8_file, NLCD_file]
              stack_rasters(raster_list, L8_NLCD_file.format(site.ID))

    # Delete Landsat8 product raw data to save space (optional) 
    # shutil.rmtree(os.path.join(LANDSAT_PATH, row_bulk_frame.productId))     

t_end = time.time()
print ("Time elapsed: {} s".format(t_end - t_start))

L8_NLCD_file = os.path.join(L8_NLCD_PATH,'L8_NLCD_Site_ID_{}_LARGE.TIF')

def tif_to_np(tif_file):
    with rio.open(tif_file) as src:
        return src.read() 
        
# dataset is a 4d np-array: (sample_index, band, x-coor, y-coor)
dataset = np.array([tif_to_np(L8_NLCD_file.format(site.ID)) for i, site in sites.iterrows()])

# swap axes to ensure channel_last
arr = np.swapaxes(dataset, 1, 3)

# drop NaN
arr_new = []
for i in range(arr.shape[0]):
    if not 0 in arr[i,8,:,:]:
        arr_new.append(arr[i,:,:,:])
arr_new = np.array(arr_new)
arr = arr_new

# np.save(os.path.join(L8_NLCD_PATH,'L8_NLCD_extracted_dataset_blast.npy'), arr)
np.save('./L8_NLCD_extracted_dataset_blast.npy', arr)

# # Visualization

# arr = np.load('./L8_NLCD/L8_NLCD_extracted_dataset_blast.npy')

# # arr[9][4] represents the 10th site, 5th band
# # Visualize
# plt.imshow(arr[9][4])
# plt.show()

# print('Shape of the image: ', arr[9][4].shape)
# print(arr[9][4])

# # Access the pixel (40, 80)

# print('Pixel value: ', arr[9][4][40][80])
