import requests
import os
from pathlib import Path

sear_dir = 'E:/HinodeSOTSPLevel2Update/Main/'
save_dir = 'E:/Research/Data/HINODE/SP/L2.1/'
pub = 'https://data.darts.isas.jaxa.jp/pub/hinode/sot/level2.1hao/'
ins = 'SP3D'

sear_year = '2014'
sear_month = '01'
prefix = sear_year + sear_month

for filename in os.listdir(sear_dir):
    
    if filename.startswith(prefix):
        
        sear_date = filename[6:8]
        name_record = filename[:-5]
        print('Downloading ' + filename + ' with requests')
        
        url = pub + sear_year + '/' + sear_month + '/' + sear_date + '/' \
            + ins + '/' + name_record + '/' + name_record + '_L2.1.fits'
            
        r = requests.get(url)
        save_path = os.path.join(save_dir, name_record + '_L2.1.fits')
        with open(save_path, 'wb') as file:
            file.write(r.content)
        
        # Checking download    
        print('Saved ' + filename)