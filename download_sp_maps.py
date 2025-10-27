import requests
import os

def download_data(sear_dir, save_dir, prefix, lv):
    for fn in os.listdir(sear_dir):
        if fn.startswith(prefix):
            sear_year = fn[0:4]
            sear_month = fn[4:6]
            sear_date = fn[6:8]
            name_record = fn[:-5]
            
            save_path = os.path.join(save_dir, fn)
            
            if os.path.exists(save_path):
                print(f'File {fn} already exists, skipping download')
                continue
            
            print(f'Downloading {fn} with requests')
            
            if lv == 2:
                url = pub + sear_year + '/' + sear_month + '/' + sear_date + '/' \
                    + ins + '/' + name_record + '/' + name_record + '.fits'
            elif lv == 21:
                url = pub + sear_year + '/' + sear_month + '/' + sear_date + '/' \
                    + ins + '/' + name_record + '/' + name_record + '_L2.1.fits'            
            
            try:
                r = requests.get(url)
                r.raise_for_status()
                
                with open(save_path, 'wb') as file:
                    file.write(r.content)
                
                # Checking download    
                print(f'Successfully saved {fn}')
                
            except Exception as error:
                print(f'Failed to download {fn}: {str(error)}')
    
def check_download(sear_dir, save_dir):
    sear_files = [f for f in os.listdir(sear_dir) if os.path.isfile(os.path.join(sear_dir, f))]
    save_files = [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))]
    
    sear_files_sorted = sorted(sear_files)
    save_files_sorted = sorted(save_files)
    
    if sear_files_sorted == save_files_sorted:
        print('All files are downloaded!')
    else:
        print('Not all files are downloaded...')

sear_dir = 'E:/HinodeSOTSPLevel2Update/Main/201401-201404/'
lv = 2
if lv == 2:
    save_dir = 'E:/Research/Data/HINODE/SP/L2/201401-201404/'
    pub = 'https://data.darts.isas.jaxa.jp/pub/hinode/sot/level2hao/'
elif lv == 21:
    save_dir = 'E:/Research/Data/HINODE/SP/L2.1/201401-201404/'
    pub = 'https://data.darts.isas.jaxa.jp/pub/hinode/sot/level2.1hao/'
ins = 'SP3D'

sear_year = '2014'
prefix = sear_year

# download_data(sear_dir, save_dir, prefix, lv)
check_download(sear_dir, save_dir)
