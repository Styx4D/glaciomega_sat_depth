from shapely.geometry import box, Polygon
import geopandas as gpd
import math
import rasterio
from pyproj.transformer import Transformer
from affine import Affine
import numpy as np
from tqdm import tqdm
import cv2
import pandas as pd
import os

global_metric_crs = 3857
global_deg_crs = 4326

def bbox_to_gdf(north, south, east, west, crs):
    coords = [
        (west, south),  # bottom-left
        (west, north),  # top-left
        (east, north),  # top-right
        (east, south),  # bottom-right
        (west, south)   # close polygon
    ]
    
    polygon = Polygon(coords)
    return gpd.GeoDataFrame(geometry=[polygon], crs=f"EPSG:{crs}")


def expand_geom(geom, multiple=5120):
    """
    Expand a geometry's bounds so width/height are multiples of `multiple`
    """
    minx, miny, maxx, maxy = geom.bounds

    width = maxx - minx
    height = maxy - miny
    # print(width, height)
    width_exp = math.ceil(width / multiple) * multiple
    height_exp = math.ceil(height / multiple) * multiple

    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2

    minx_new = cx - width_exp / 2
    maxx_new = cx + width_exp / 2
    miny_new = cy - height_exp / 2
    maxy_new = cy + height_exp / 2
    # print( maxx_new-minx_new, maxy_new-miny_new)
    return box(minx_new, miny_new, maxx_new, maxy_new)

def expand_geom_bounds_to_multiple(gdf, multiple=5120):
    out = gdf.copy()
    out["geometry"] = gdf.geometry.apply(lambda g: expand_geom(g, multiple))
    return out

def prepare_footprint( north, south, east, west, zone_crs, round_to_512 = True ):

    gdf = bbox_to_gdf( north, south, east, west, zone_crs )

    gdf = gdf.to_crs(global_metric_crs)

    if round_to_512:
        gdf = expand_geom_bounds_to_multiple( gdf, multiple=5120*2 )
    return gdf

def extent_to_box_geometry(extent_wgs84):
    """
    Convert [lon_min, lat_min, lon_max, lat_max] to a Shapely box geometry.
    """
    lon_min, lat_min, lon_max, lat_max = extent_wgs84
    return box(lon_min, lat_min, lon_max, lat_max)

""" Open data def """

def open_glo30_dem( glo30_path, keep_only=None ):

    glo30_dem = { "name":[], "array":[], "geometry":[], "profile":[] }

    for dirpath, dirnames, filenames in os.walk( glo30_path ):
        for filename in filenames:
            if 'xml' in filename:continue
            # print(filename)
            if keep_only is not None:
                if keep_only not in filename:
                    continue

            p = os.path.join(dirpath, filename)

            with rasterio.open(p) as src:
                arr = src.read(1)
                profile = src.profile
                bounds = src.bounds
                extent_wgs84 = extent_to_box_geometry( [bounds.left, bounds.bottom, bounds.right, bounds.top] )

                glo30_dem["name"].append( filename )
                glo30_dem["array"].append( arr )
                glo30_dem["profile"].append( profile )
                glo30_dem["geometry"].append( extent_wgs84 )

            
            print( '---> ',filename, profile['crs'].to_epsg(), arr.shape )

    glo30_dem = gpd.GeoDataFrame(glo30_dem, geometry="geometry", crs=f"EPSG:{4326}")  
    return glo30_dem 


""" Traitement Sentinel """

# création d'un dictionnaire contenant les CRS, chemin, date et emprise AOI

def sentinel_metadata( p ):
    # ouvre une image sentinel et retourne le CRS et l'extents
    with rasterio.open(p) as src:
        # d = src.read(1)
        profile = src.profile
        # print(src.read_crs())
        bounds = src.bounds

        # reprojeté sur le crs de la couche dataset_metadata pour réaliser les couples lidar-sentinels
        transformer = Transformer.from_crs(src.crs, "EPSG:3034", always_xy=True)
        lon_min, lat_min = transformer.transform(bounds.left, bounds.bottom)
        lon_max, lat_max = transformer.transform(bounds.right, bounds.top)
        extent_3034 = [lon_min, lat_min, lon_max, lat_max]

        extent_wgs84 = [bounds.left, bounds.bottom, bounds.right, bounds.top]

    return profile['crs'].to_epsg(), extent_3034, extent_wgs84
    
def get_sentinels_crs( p ):

    sentinels_by_crs = { 3034:{"geometry":[], "origin_crs":[], "path":[]} }

    for dirpath, dirnames, filenames in os.walk(p):
        for filename in filenames:
            if 'B09_60m' in filename:
                sp = os.path.join(dirpath, filename)
                safe_path = dirpath.split('.SAFE')[0] + '.SAFE'
                crs, extent_3034, extent_wgs84 = sentinel_metadata(sp)

                geom_3034 = extent_to_box_geometry( extent_3034 )
                sentinels_by_crs[ 3034 ]["geometry"].append( geom_3034 )
                sentinels_by_crs[ 3034 ]["origin_crs"].append( crs )
                sentinels_by_crs[ 3034 ]["path"].append( safe_path )

                geom_wgs84 = extent_to_box_geometry( extent_wgs84 )

                if crs not in sentinels_by_crs.keys():
                    sentinels_by_crs[crs] = {"geometry":[], "path":[]}
                
                sentinels_by_crs[crs]["geometry"].append( geom_wgs84 )
                sentinels_by_crs[crs]["path"].append( safe_path )

    return sentinels_by_crs

""" crop, resize... """

def get_glo30_data_name( glo30_dem, sentinel_crs, grid_geometry,cut_geometry ):

    # glo30 avec le max de couverture sur la grille
    idx_glo30 = glo30_dem.to_crs( sentinel_crs ).intersection(cut_geometry).area.idxmax()
    # print(glo30_dem.to_crs( sentinel_crs ).intersection(cut_geometry).area)
    glo30_to_use = glo30_dem.iloc[ idx_glo30 ]
    glo30_name = glo30_to_use['name']
    return glo30_name


def cut_glo30( glo30_array, glo30_profile, grid_geometry, s_crs, delta_pixel = 20 ):
    # delta pixel : coupe plus loin pour être sur de recouvrir le MNS
    # car les différents crs entrainenent des rotations

    xmin, ymin, xmax, ymax = grid_geometry
    transformer = Transformer.from_crs(s_crs, "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer.transform(xmin, ymin)
    lon_max, lat_max = transformer.transform(xmax, ymax)

    # print( lon_min, lat_min )
    # print( lon_max, lat_max )
    # print(glo30_profile['transform'])

    tx, ty = ~glo30_profile['transform'] * (lon_min, lat_max)
    tx_c, ty_c = ~glo30_profile['transform'] * (lon_max, lat_min)

    tx,ty = int(tx)-delta_pixel, int(ty)-delta_pixel
    tx_c,ty_c = int(tx_c)+delta_pixel, int(ty_c)+delta_pixel

    # print(tx, ty, tx_c, ty_c)

    glo30_cut_array = glo30_array[max(0,ty):min(ty_c, glo30_array.shape[0]), max(0,tx):min(tx_c,glo30_array.shape[1])]
    # glo30_cut_transform = change_transform_coords( glo30_profile['transform'], lon_min, lat_max, delta_pixel )
    glo30_cut_transform = glo30_profile["transform"] * Affine.translation(max(0,tx), max(0,ty))
    # print(ty,tx)
    # glo30_cut_transform = glo30_profile["transform"] * Affine.translation(tx, ty)

    return glo30_cut_array, glo30_cut_transform

def get_glo30_data( glo30_dem, sentinel_crs, grid_geometry,cut_geometry ):

    # glo30 avec le max de couverture sur la grille
    idx_glo30 = glo30_dem.to_crs( sentinel_crs ).intersection(cut_geometry).area.idxmax()
    # print(glo30_dem.to_crs( sentinel_crs ).intersection(cut_geometry).area)
    glo30_to_use = glo30_dem.iloc[ idx_glo30 ]
    # print(idx_glo30, glo30_to_use)

    glo30_profile = glo30_to_use['profile']
    glo30_array = glo30_to_use['array']
    glo30_name = glo30_to_use['name']

    # print(glo30_profile)
    # coupe le glo30
    glo30_cut_array, glo30_cut_transform = cut_glo30( glo30_array, glo30_profile, cut_geometry.bounds, sentinel_crs )

    return glo30_cut_array, glo30_cut_transform, glo30_name


""" def d'ouverture glo/Sentinel """
def get_bands_and_profile( p ):
    with rasterio.open(p) as src:
        d = src.read()
        profile = src.profile
        # print(src.read_crs())
    return d.transpose(1,2,0).squeeze(), profile

def get_mns_data( p ):
    
    mns, profile = get_bands_and_profile( p )
    crs = profile['crs']
    if crs is not None: 
        epsg_code = crs.to_epsg()
    else:
        epsg_code = None 
        
    return mns.astype(np.float32), profile, epsg_code

sent_to_keep = ['B02_10m.jp2', 'B03_10m.jp2', 'B04_10m.jp2', 'B8A_20m.jp2', 'B11_20m.jp2', 'B09_60m.jp2']
def open_sentinels( path, sent_to_keep = sent_to_keep ):
    # path : .SAFE folder
    bands_data = {}
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            # print(filename)
            check = [ s in filename for s in sent_to_keep ]

            if not any(check): continue
            # print(filename, any(check))
            band = filename.split('_')[2].split('.')[0]
            sent_img, sent_profile = get_bands_and_profile( os.path.join(dirpath, filename) )
            bands_data[ band ] = { "sent_array": sent_img, "sent_profile": sent_profile }
    return bands_data

def harmonize_sentinels( bands_data ):
    shape_x = []
    shape_y = []
    crs = []
    for b_key in bands_data.keys():
        crs.append( bands_data[b_key]['sent_profile']['crs'].to_epsg() )
        shape_x.append( bands_data[b_key]['sent_array'].shape[0])
        shape_y.append( bands_data[b_key]['sent_array'].shape[1])

    assert np.unique(crs).size == 1
    # resize sur le max des shapes (10m)
    shape_x_max = max(shape_x)
    shape_y_max = max(shape_y)

    for b_key in bands_data.keys():
        bands_data[b_key]['sent_array'] = cv2.resize( bands_data[b_key]['sent_array'], (shape_x_max, shape_y_max), interpolation=cv2.INTER_CUBIC)
    return bands_data


def cut_sentinels_bands( bands_data, xmin, ymin, xmax, ymax ):
    sent_profile = bands_data['B02']['sent_profile']
    tx, ty = ~sent_profile['transform'] * (xmin, ymax)
    tx_m, ty_m = ~sent_profile['transform'] * (xmax, ymin)
    tx,ty = int(tx), int(ty)
    tx_m,ty_m = int(tx_m), int(ty_m)
    # print( 'cut Sentinels bands')
    # print(tx,tx_m, ty,ty_m)
    # print(tx-tx_m, ty-ty_m)
    bands_cut = {}
    for bkey in bands_data.keys():
        bands_cut[bkey] = bands_data[bkey]['sent_array'][ty:ty_m,tx:tx_m] 
    return bands_cut

def create_profile_dict( data, dst_crs, transform ):
    count = 1 if data.ndim == 2 else data.shape[-1]
    profile = {
        'driver': 'GTiff',
        'height': data.shape[0],
        'width': data.shape[1],
        'crs': dst_crs,
        'count': count, 
        'dtype': data.dtype,
        'transform': transform,
        'compress': 'LZW'
    }
    return profile


def raster_to_xyz(mns, profile, density = None, replace_nan = False, nan_values = -9999):
    """Convert raster to XYZ point cloud"""
    # mns_density : to discard interpolate areas
    transform = profile['transform']

    if replace_nan:
        mns = np.nan_to_num( mns, nan= nan_values)

    m = ~np.isnan(mns)

    if density is not None:
        m = m & (density > 0)

    rows, cols = np.where(m)  # valid pixels only
    
    zs = mns[rows, cols]
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset = 'center')
    xyz = np.stack([xs, ys, zs], axis=1)
    return xyz

def reproject_xyz(xyz, epsg_from, epsg_to):
    transformer = Transformer.from_crs(epsg_from, epsg_to, always_xy=True)
    xs, ys, zs = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    x_new, y_new = transformer.transform(xs, ys)
    return np.stack([x_new, y_new, zs], axis=1)


def img_at_pos_ji(img, ji_cam, transform):
    # pick RGB values at xy_cam positions by performing a linear interpolation
    ji_cam[:,0] = (ji_cam[:,0] - transform[2] - transform[0]/2 ) / transform[0]
    ji_cam[:,1] = (ji_cam[:,1] - transform[5] - transform[4]/2 ) / transform[4]
    
    ji_cam_lower = np.floor(ji_cam).astype(int)
    ji_cam_upper = np.ceil(ji_cam).astype(int)
    ji_cam_diff = ji_cam - np.floor(ji_cam)

    H, W = img.shape[:2] 

    x_low  = np.clip(ji_cam_lower[..., 0], 0, W - 1)
    x_up   = np.clip(ji_cam_upper[..., 0], 0, W - 1)
    y_low  = np.clip(ji_cam_lower[..., 1], 0, H - 1)
    y_up   = np.clip(ji_cam_upper[..., 1], 0, H - 1)

    img_ll = img[y_low, x_low].astype(np.float64)
    img_ul = img[y_up,  x_low].astype(np.float64)
    img_lu = img[y_low, x_up ].astype(np.float64)
    img_uu = img[y_up,  x_up ].astype(np.float64)

    
    # img_ll = img[ji_cam_lower[..., 1], ji_cam_lower[..., 0]].astype(np.float64)
    # img_ul = img[ji_cam_upper[..., 1], ji_cam_lower[..., 0]].astype(np.float64)
    # img_lu = img[ji_cam_lower[..., 1], ji_cam_upper[..., 0]].astype(np.float64)
    # img_uu = img[ji_cam_upper[..., 1], ji_cam_upper[..., 0]].astype(np.float64)
    # print(img_ll.shape)

    weighted_pixel = ( ((     ji_cam_diff[..., 1]) * (     ji_cam_diff[..., 0])) * img_uu) + \
                     ( ((1. - ji_cam_diff[..., 1]) * (     ji_cam_diff[..., 0])) * img_lu) + \
                     ( ((     ji_cam_diff[..., 1]) * (1. - ji_cam_diff[..., 0])) * img_ul) + \
                     ( ((1. - ji_cam_diff[..., 1]) * (1. - ji_cam_diff[..., 0])) * img_ll)
                
    return weighted_pixel


def glo30_xyz_to_array( xyz, height, width ):
    # xyz : coords of center pixel
    pxl_x_size = (xyz[:,0].max() - xyz[:,0].min()) / ( width - 1)
    pxl_y_size = (xyz[:,1].min() - xyz[:,1].max()) / ( height - 1)

    x_origin = xyz[:,0].min() - (pxl_x_size / 2)
    y_origin = xyz[:,1].max() - (pxl_y_size / 2)

    # print( pxl_x_size, pxl_y_size, x_origin, y_origin )

    raster_array = np.full((height, width), np.nan).astype(np.float32)  # Initialize nodata value
        
    x_indices = ((xyz[:, 1] - y_origin) / pxl_y_size).astype(int)
    y_indices = ((xyz[:, 0] - x_origin) / pxl_x_size).astype(int)
    # print(y_indices)
    raster_array[x_indices, y_indices] = xyz[:, 2]

    return raster_array


def save_tif_rasterio( output_file, data, profile):
    if data.ndim == 2:
        data = data[None, :, :]
    else:
        data = data.transpose(2,0,1) # C,H,W

    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(data)



def open_grids( gdf, dataset_by_crs, min_size = 5120 ):    
    # emprise AOI par image 
    for crs in dataset_by_crs :
        # print(crs)
        dataset_by_crs[crs][ 'aoi' ] = []

        gdf_reproj = gdf.to_crs( crs )

        bounding_boxes = gdf_reproj.bounds # minx	miny	maxx	maxy

        for geom in dataset_by_crs[ crs ]['geometry'] :

            min_x, min_y, max_x, max_y = geom.bounds

            bounding_boxes_shifted = bounding_boxes.copy()

            bounding_boxes_shifted[ ['minx', 'maxx']] -= min_x
            bounding_boxes_shifted[ ['miny', 'maxy']] -= min_y
            
            bounding_boxes_shifted['shiftx'] = (
                (bounding_boxes_shifted['minx'] >= 0) &
                (bounding_boxes_shifted['maxx'] <= (max_x - min_x))
            )

            bounding_boxes_shifted['shifty'] = (
                (bounding_boxes_shifted['miny'] >= 0) &
                (bounding_boxes_shifted['maxy'] <= (max_y - min_y))
            )

            # print( bounding_boxes_shifted )

            bounding_boxes_shifted = bounding_boxes_shifted[bounding_boxes_shifted['shiftx'] & bounding_boxes_shifted['shifty']]

            # round to align on sentinels coords
            bounding_boxes_shifted['minx'] = np.floor( bounding_boxes_shifted['minx']) + min_x
            bounding_boxes_shifted['miny'] = np.floor( bounding_boxes_shifted['miny']) + min_y
            bounding_boxes_shifted['maxx'] = np.ceil( bounding_boxes_shifted['maxx']) + min_x
            bounding_boxes_shifted['maxy'] = np.ceil( bounding_boxes_shifted['maxy']) + min_y

            w = bounding_boxes_shifted['maxx'] - bounding_boxes_shifted['minx']
            mask = w < min_size
            # print('w',w)
            shiftw = min_size - w[mask]
            bounding_boxes_shifted.loc[mask, 'maxx'] += (shiftw//2)
            bounding_boxes_shifted.loc[mask, 'minx'] -= shiftw-(shiftw//2)

            h = bounding_boxes_shifted['maxy'] - bounding_boxes_shifted['miny']
            mask = h < min_size
            # print('h',h)
            shifth = min_size - h[mask]
            bounding_boxes_shifted.loc[mask, 'maxy'] += (shifth//2)
            bounding_boxes_shifted.loc[mask, 'miny'] -= shifth-(shifth//2)
            
            
            dataset_by_crs[crs][ 'aoi' ].append ( bounding_boxes_shifted )
        
    
    return dataset_by_crs


def parse_safe_info( path ):
    """Extract satellite, processing baseline, orbit, and year from a .SAFE file name."""
    file_name = os.path.basename( path )
    parts = file_name.split('_')
    satellite = parts[0]
    date_acquisition = parts[2].split('T')[0] #YYYYMMDD
    baseline = parts[3]
    orbit = parts[4]
    tile = parts[5]
    date_traitement = parts[6].split('T')[0] #YYYYMMDD
    return satellite, baseline, orbit, tile, date_acquisition, date_traitement

def get_sent_path( safe_p ):
    files_path = {}
    files_path['b02_path'] = os.path.join( safe_p, 'B02_10m.tif' )
    files_path['b03_path'] = os.path.join( safe_p, 'B03_10m.tif' )
    files_path['b04_path'] = os.path.join( safe_p, 'B04_10m.tif' )
    files_path['b09_path'] = os.path.join( safe_p, 'B09_10m.tif' )
    files_path['b11_path'] = os.path.join( safe_p, 'B11_10m.tif' )
    files_path['b8a_path'] = os.path.join( safe_p, 'B8A_10m.tif' )
    files_path['glo30_path'] = os.path.join( safe_p, 'GLO30_10m.tif' )
    return files_path

def get_sent_daytime( file_p ):
    data = parse_safe_info( file_p )
    return data[4]

def add_sentinel_daytime_info( dataset_by_crs ):
    for s_crs in dataset_by_crs.keys():
        paths_list = dataset_by_crs[ s_crs ]['path']
        dataset_by_crs[ s_crs ][ 'sent_daytime' ]  = [ get_sent_daytime(p) for p in paths_list]
    return dataset_by_crs

def get_dataset_by_crs( path, gdf ):
    # retourne un dictionnaire contenant les CRS, les chemins, l'AOI, les dates et la géométrie de l'emprise
    dataset_by_crs = get_sentinels_crs( path )
    dataset_by_crs = open_grids( gdf, dataset_by_crs )
    dataset_by_crs = add_sentinel_daytime_info( dataset_by_crs )
    return dataset_by_crs

def make_custom_dataset( dataset_by_crs, glo30_dems, output_folder):

    for crs in dataset_by_crs.keys():

        # ignore 3034
        if crs == 3034: continue
        dataset = dataset_by_crs[crs]

        for i in tqdm(range( len(dataset['path']))):

            sentinels_safe_folder = dataset['path'][i]
            # print( sentinels_safe_folder.split('sentinels')[0] )
            
            safe_basename = os.path.basename(sentinels_safe_folder)
            
            bands_data = None

            for j, row in dataset['aoi'][i].iterrows():
                xmin, ymin, xmax, ymax = row[['minx','miny','maxx','maxy']]
                # print( xmin, ymin, xmax, ymax )
                glo30_name = get_glo30_data_name( glo30_dems, crs, dataset['geometry'][i], box(xmin, ymin, xmax, ymax))

                out_safe = os.path.join( output_folder, glo30_name.split('.tif')[0], safe_basename )

                if os.path.exists( out_safe ):
                    continue

                glo30_cut_array, glo30_cut_transform, glo30_name = get_glo30_data( glo30_dems, crs, dataset['geometry'][i], box(xmin, ymin, xmax, ymax))

                if bands_data is None:
                    bands_data = open_sentinels( sentinels_safe_folder )
                    bands_data = harmonize_sentinels( bands_data )

                bands_data_cut = cut_sentinels_bands( bands_data, xmin, ymin, xmax, ymax)
                
                if  bands_data_cut['B03'].mean() ==0: continue
                # print( glo30_name )
            
                xyz_transform = Affine(10, 0, xmin, 0, -10, ymax)
                xyz_profile = create_profile_dict( bands_data_cut['B02'], crs, xyz_transform )

                xyz_coords = raster_to_xyz( bands_data_cut['B02'], xyz_profile, replace_nan=True ) # replace nan to keep coords same shape
                xyz_coords_4326 = reproject_xyz( xyz_coords.copy(), crs, 4326 )

                xyz_coords[:,2] =img_at_pos_ji( glo30_cut_array.copy(), xyz_coords_4326.copy(), glo30_cut_transform)

                glo30_dem_interp_ji = glo30_xyz_to_array( xyz_coords, bands_data_cut['B02'].shape[0], bands_data_cut['B02'].shape[1])

                # save 
                save_transform = Affine(10, 0, xmin, 0, -10, ymax)
                profile = create_profile_dict( bands_data_cut['B02'], crs, save_transform )

                # out_safe = os.path.join( output_folder, safe_basename )
                
                # os.path.join( output_folder, glo30_name.split('.tif')[0], safe_basename )

                os.makedirs( out_safe, exist_ok = True )

                for k in bands_data_cut.keys():
                    out_p = os.path.join( out_safe, f'{k}_10m.tif')
                    save_tif_rasterio( out_p, bands_data_cut[k], profile)
                
                out_p = os.path.join( out_safe, 'GLO30_10m.tif')
                save_tif_rasterio( out_p, glo30_dem_interp_ji, profile)