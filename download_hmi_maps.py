import os

import astropy.units as u
from astropy.time import Time
import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a

sear_dir = 'E:/HinodeSOTSPLevel2Update/Main/201401-201404/'
save_dir = 'E:/Research/Data/SDO/HMI/Magnetogram/720s/201401-201404/'
year_record = '2014'
jsoc_email = 'ruizhuo@pku.edu.cn'

sear_year = '2014'
prefix = sear_year

os.makedirs(save_dir, exist_ok=True)

for fn in os.listdir(sear_dir):
    if fn.startswith(prefix):
        name_record = fn[:-5]
        print('Downloading ' + fn + ' with Fido')        

        sear_time = Time(f"{fn[0:4]}-{fn[4:6]}-{fn[6:8]}T{fn[9:11]}:{fn[11:13]}:{fn[13:15]}", 
                         scale='utc', format='isot')

        query = Fido.search(
            a.Time(sear_time - 1*u.s, sear_time + 1*u.s),
            a.Sample(360*u.s),
            a.jsoc.Series.hmi_m_720s,
            a.jsoc.Notify(jsoc_email)
        )
        print(query)

        # Checking download    
        if len(query) > 0 and len(query[0]) > 0:
            file_info = query[0][0]
            T_REC = file_info['T_REC']
            
            save_fn = f'hmi.m_720s.{T_REC[0:4]}{T_REC[5:7]}{T_REC[8:10]}_{T_REC[11:13]}{T_REC[14:16]}{T_REC[17:19]}_TAI.1.magnetogram.fits'
            save_path = os.path.join(save_dir, save_fn)
            
            if os.path.exists(save_path):
                print(f'File {save_fn} already exists, skipping download')
                continue
        
            files = Fido.fetch(query[0][0], path=save_dir, overwrite=False)
            print(f'Saved to {files}')
        else:
            print(f'No data found for {fn}')