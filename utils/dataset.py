import os
import pandas as pd

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

def get_sent_daytime( file_p ):
    data = parse_safe_info( file_p )
    return data[4]

def get_glacioclim_dataset_path( path, check_csv_file ):
    df_keep = pd.read_csv(check_csv_file)
    fold, cham_data_name = os.path.split(path.rstrip(os.sep))
    dataset = get_custom_dataset_path( fold, [cham_data_name] )

    cham_dataset = {cham_data_name:{'safe_folder':[], 'acq_date':[], 'band_path':[]}}

    list_keepers = list(df_keep['path'])

    for i in range(len(dataset[ cham_data_name ]['safe_folder'])):
        safe_folder = dataset[ cham_data_name ]['safe_folder'][i]
        acq_date = dataset[ cham_data_name ]['acq_date'][i]
        band_path = dataset[ cham_data_name ]['band_path'][i]

        if safe_folder in list_keepers:
            cham_dataset[ cham_data_name ]['safe_folder'].append( safe_folder )
            cham_dataset[ cham_data_name ]['acq_date'].append( acq_date )
            cham_dataset[ cham_data_name ]['band_path'].append( band_path )
    print( len(cham_dataset[ cham_data_name ]['safe_folder']), ' left after filter')
    return cham_dataset



def get_custom_dataset_path( path, data_names ):
    # return data path, sorted by time

    dataset = {}

    for data_name in data_names:
        dataset[data_name] = { 'safe_folder':[], 'acq_date':[], 'band_path':[] }

        data_path = os.path.join( path, data_name )
        for safe_file in os.listdir( data_path ):
            sentinel_acq_date = get_sent_daytime(safe_file)

            safe_p = os.path.join( data_path, safe_file )

            dataset[data_name]['safe_folder'].append( safe_file )
            dataset[data_name]['acq_date'].append( sentinel_acq_date )
            dataset[data_name]['band_path'].append( get_sent_path(safe_p) )

    for name, d in dataset.items():
        z = sorted(
            zip(d['acq_date'], d['safe_folder'], d['band_path']),
            key=lambda x: int(x[0])
        )

        d['acq_date'], d['safe_folder'], d['band_path'] = map(list, zip(*z))
        dataset[name] = d
        print(name,', n images : ', len(d['safe_folder']))
    
    return dataset