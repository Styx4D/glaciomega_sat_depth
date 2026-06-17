import json
import requests
import os
import zipfile
from os import listdir
import pyproj
import numpy as np
from datetime import datetime, timedelta
import shutil
import re
import time
# from tqdm import tqdm

"""
Keycloak : token d'access a l'api
"""
def get_keycloak(username, password) :
        data = {
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
            }
        try:
            r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data=data,
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            )
            r.raise_for_status()
        except Exception as e:
            raise Exception(
                f"Erreur d\'identification: {r.json()}"
                )
        return r.json()["access_token"]

class create_keycloak_manager():
    def __init__(self, username, password):
        # l'API limite 100 keycloak par heure, avec 10 minute d'utilisation max
        # ici simple manager pour éviter de regénérer trop de keycloak

        self.time_init = time.time()
        self.time_counter = 0.
        self.username = username
        self.password = password
        self.keycloak = get_keycloak( username, password )
    
    def check_keycloak(self):
        ellapsed = time.time() - self.time_init 
        self.time_counter += ellapsed

        if self.time_counter > 400:
            self.time_init = time.time()
            self.time_counter = 0.
            self.keycloak = get_keycloak( self.username, self.password )
        
        return self.keycloak

"""
def de construction du query
"""
def convert_bbox_format(bbox):
    # convert bbox to str format for query
    # (x_min, y_min, x_max, y_max) EPSG 4326

    # Extract the individual latitude and longitude values
    min_longitude, min_latitude = bbox[0], bbox[1]
    max_longitude, max_latitude = bbox[2], bbox[3]

    # Construct the converted bbox string
    converted_bbox = "{0} {1},{0} {3},{2} {3},{2} {1}, {0} {1}".format(
        min_longitude, min_latitude, max_longitude, max_latitude
    )

    return converted_bbox


def build_copernicus_url(name, start_date, end_date, bbox, cloud_coverage):
        
    base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter="

    filters = []
    # print('onbuild')
    if name:
        # print('name',name)
        name_filter = 'Name eq \'' + name + '\''
        filters.append( name_filter)

    if cloud_coverage:
        # print('cloud_coverage',cloud_coverage)
        cloud_coverage = int(cloud_coverage)
        cloud_filter = f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {cloud_coverage}.00)"    
        filters.append( cloud_filter)

    if bbox:
        # print('bbox',bbox)
        bbox_str = convert_bbox_format(bbox)
        bbox_filter = "OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({0}))')".format(bbox_str)
        filters.append( bbox_filter )
    
    if start_date:
        # ("%Y-%m-%dT00:00:00.000Z")
        # start_date = "{0}-{1}-{2}T00:00:00.000Z".format(start_date[0],start_date[1],start_date[2])
        # end_date = "{0}-{1}-{2}T00:00:00.000Z".format(end_date[0],end_date[1],end_date[2])
        start_date = "{0:04d}-{1:02d}-{2:02d}T00:00:00.000Z".format(*start_date)
        end_date = "{0:04d}-{1:02d}-{2:02d}T00:00:00.000Z".format(*end_date)
        date_filter = "ContentDate/Start gt {0} and ContentDate/End lt {1}".format(start_date, end_date)
        filters.append( date_filter )

    # TODO add collection to filter
    collection_filter = "contains(Name,'MSIL2A')"
    filters.append( collection_filter )

    filters = " and ".join(filters)
    url = base_url + filters

    return url
 
def get_data(coper_url):
    response = requests.get(coper_url)
    content = response.content
    content = content.decode("utf-8")  # Convert bytes to string
    data = json.loads(content)
    return data

def query_link(url):
    all_data = []
    while url:
        data = get_data(url)
        if 'value' in data.keys():
            all_data.extend(data['value'])
    
        # Check if there is more data to query
        if '@odata.nextLink' in data.keys():
            url = data['@odata.nextLink']
        else:
            url = None
    
    return all_data

def get_data_link( username, password, sent_name, cloud_cover, start_date, end_date, extent, layer_to_DL, keycloak_manager = None ):

    # initiate keycloak
    if keycloak_manager is None:
        keycloak_manager = create_keycloak_manager( username, password )
    

    url_query = build_copernicus_url( sent_name, start_date, end_date, extent, cloud_cover )

    # print(url_query)
    # retrouve tout les liens (id) qui peuvent être télécharger
    all_data = query_link( url_query )

    return keycloak_manager, all_data


def download_archive_with_try(keycloak_manager, data_list, output_folder):

    zip_paths = []
    data_failed = []
    for i, data in enumerate(data_list):
        try:
            keycloak_token = keycloak_manager.check_keycloak()

            id_dl = data['Id']
            name_dl = data['Name'].split('.')[0]

            
            session = requests.Session()
            session.headers.update({'Authorization': f'Bearer {keycloak_token}'})

            url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({id_dl})/$value"

            response = session.get(url, stream=True)

            out_file = os.path.join( output_folder, f"{name_dl}.zip")
            # zip_paths.append( out_file )

            if response.status_code == 200:
                with open(out_file, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # filter out keep-alive new chunks
                            file.write(chunk)
                
                session.close()
            
            zip_paths.append( out_file )
        except:
            data_failed.append(data)

    return zip_paths, data_failed

NODES_BASE = "https://download.dataspace.copernicus.eu/odata/v1"
def find_nodes(uuid, filenames, session):
    urls = {}
    filenames = set(filenames)  

    def recurse(path=""):
        url = f"{NODES_BASE}/Products({uuid})/Nodes{path}"
        r = session.get(url)
        r.raise_for_status()
        nodes = r.json().get("value", r.json().get("result", []))

        for node in nodes:
            name = node["Name"]
            # print(name)
            subpath = f"{path}({name})"
            # print(subpath)

            if any( [f in name for f in filenames ]) :
                urls[name] = f"{NODES_BASE}/Products({uuid})/Nodes{subpath}/$value"

            recurse(subpath + "/Nodes")
    recurse()
    return urls

def dl_node(session, url, outpath):
    response = session.get(url, stream=True)

    if response.status_code == 200:
        with open(outpath, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:  
                    file.write(chunk)
    

def download_archive_with_try_nodes(keycloak_manager, data_list, output_folder, filenames, pass_existing = False):
    data_ok = []
    data_failed = []
    for i, data in enumerate(data_list):

        keycloak_token = keycloak_manager.check_keycloak()

        id_dl = data['Id']
        name_dl = data['Name']

        outpath = os.path.join( output_folder, name_dl)

        if pass_existing and os.path.exists(outpath):
            # print('passed')
            continue 
        # print('not passed')
        # continue
        os.makedirs(outpath, exist_ok=True)
        
        session = requests.Session()
        session.headers.update({'Authorization': f'Bearer {keycloak_token}'})

        try:
            nodes_dict = find_nodes( id_dl, filenames, session )

            for fname, url in nodes_dict.items():
                out_ = os.path.join( outpath, fname )

                if pass_existing and os.path.exists(out_):
                    continue 

                dl_node( session, url, out_)

        except Exception as e:
            print(f'failed to download, error : {e}')
            data_failed.append( name_dl )
            session.close()
            continue
        
        data_ok.append(name_dl)
        session.close()
    return data_ok, data_failed


def download_sentinel_files( gdf, output_folder, name, username_copernicus, passw_copernicus, start_date, end_date, cloud_max = 50 ):
    out_sent = os.path.join( output_folder, 'sentinels' )

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    gdf = gdf.to_crs( 4326 )

    keycloak_manager = None
    s_name = None


    layer_to_DL = [
        'B02_10m.jp2',
        'B03_10m.jp2',
        'B04_10m.jp2',
        'B11_20m.jp2',
        'B8A_20m.jp2',
        'B09_60m.jp2',
        'MTD_MSIL2A.xml',
        'MTD_TL.xml',
    ]


    for i, row in gdf.iterrows():
        # continue

        geom = row["geometry"]
        extent = geom.bounds

        print('---- Downloading Sentinel on ', name, ' from ', start_dt.date(), ' to ', end_dt.date())
        start_year = start_dt.year
        end_year = end_dt.year
        
        if start_year == end_year:
            start_dates = [ [start_dt.year, start_dt.month, start_dt.day] ]
            end_dates = [ [end_dt.year, end_dt.month, end_dt.day] ]
        else:
            start_dates = []
            end_dates = []
            for j, s_year in enumerate(range(start_year, end_year+1 )):
                if j ==0 :
                    start_date = [start_dt.year, start_dt.month, start_dt.day]
                    end_date = [start_dt.year +1, 1, 1]
                elif j == (end_year - start_year):
                    start_date = [end_dt.year, 1, 1]
                    end_date = [end_dt.year, end_dt.month, end_dt.day]
                else:
                    start_date = [ s_year, 1, 1]
                    end_date = [ s_year+1, 1, 1]
                start_dates.append( start_date )
                end_dates.append( end_date )
        
        for start_date, end_date in zip( start_dates, end_dates ):    
            print('> query link', start_date, end_date )
            try:
                keycloak_manager, links = get_data_link( username_copernicus, passw_copernicus, s_name, cloud_max, start_date, end_date, extent, layer_to_DL, keycloak_manager )
            except Exception as e:
                print(f'failed to query links, error : {e}')
                continue
            print('found ', len(links))
            data_ok, data_failed = download_archive_with_try_nodes(
                                keycloak_manager, links, out_sent, layer_to_DL, pass_existing=True)
            print('success : ', len(data_ok), 'failed : ', len(data_failed))

    return out_sent

