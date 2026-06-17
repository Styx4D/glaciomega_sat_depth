import requests
import os

def download_glo30(west, south, east, north, api_key, out_file="glo30.tif"):

    url = "https://portal.opentopography.org/API/globaldem"

    params = {
        "demtype": "COP30",  # Copernicus GLO-30
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": api_key
    }

    r = requests.get(url, params=params, stream=True)

    if r.status_code != 200:
        raise RuntimeError(f"Download failed: {r.text}")

    with open(out_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"GLO-30 DEM saved to {out_file}")

def download_glo30_from_gdf( gdf, API_key_OpenTopo, output_folder, name ):
    gdf = gdf.to_crs( 4326 )

    out_glo = os.path.join( output_folder, 'glo30_dem' )
    os.makedirs( out_glo, exist_ok=True )

    for i, row in gdf.iterrows():
        geom = row["geometry"]

        minx, miny, maxx, maxy = geom.bounds
        out_p = os.path.join( out_glo, name) + '.tif'
        
        if os.path.exists( out_p ): continue

        download_glo30( minx, miny, maxx, maxy, API_key_OpenTopo, out_p )
    return out_glo