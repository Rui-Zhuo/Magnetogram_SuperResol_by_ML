## Using jsoc_email: 'ruizhuo@pku.edu.cn'
## Using jsoc_email: '2301110628@pku.edu.cn'
## Using jsoc_email: '1900012447@pku.edu.cn'
## Using jsoc_email: 'gjh1024989783@gmail.com'

import os
import time
import urllib.error

import astropy.units as u
from astropy.time import Time
from sunpy.net import Fido
from sunpy.net import attrs as a

# Configuration
sear_dir = 'E:/HinodeSOTSPLevel2Update/Main/2014/'
save_dir = 'E:/Research/Data/SDO/HMI/FullDisk/Magnetogram/4096/2014/'
year_record = '2014'
sear_year = '2014'
jsoc_email = 'ruizhuo@pku.edu.cn'
prefix = sear_year

# Network settings
MAX_RETRIES = 5
RETRY_DELAY = 10

# Failed log path
failed_log_path = os.path.join(save_dir, f'download_failed_{year_record}.txt')
# Match log path (records fn <-> save_fn mapping)
match_log_path = os.path.join(save_dir, f'download_match_{year_record}.txt')

os.makedirs(save_dir, exist_ok=True)

# Initialize match log (append mode, with header if empty)
with open(match_log_path, 'a', encoding='utf-8') as match_log:
    if os.path.getsize(match_log_path) == 0:
        match_log.write(f'===== File Mapping Log ({year_record}) =====\n')
        match_log.write('Log Time Format: UTC\n')
        match_log.write('Format: Source Filename (fn) | Matched Save Filename (save_fn) | Match Time | Status\n')
        match_log.write('Status: FOUND/MISSING/ERROR\n')
        match_log.write('-' * 120 + '\n')

with open(failed_log_path, 'a', encoding='utf-8') as failed_log:
    if os.path.getsize(failed_log_path) == 0:
        failed_log.write(f'===== Download Failure Log ({year_record}) =====\n')
        failed_log.write('Log Time Format: UTC\n')
        failed_log.write('Format: Filename | Failure Time | Reason\n')
        failed_log.write('-' * 80 + '\n')

    for fn in os.listdir(sear_dir):
        if fn.startswith(prefix):
            print(f'\n===== Processing {fn} =====')
            current_utc = Time.now().isot
            save_fn = ""  # Initialize save_fn as empty
            match_status = "ERROR"  # Default status: ERROR

            try:
                sear_time = Time(
                    f'{fn[0:4]}-{fn[4:6]}-{fn[6:8]}T{fn[9:11]}:{fn[11:13]}:{fn[13:15]}',
                    scale='utc', format='isot'
                )
            except ValueError as e:
                error_msg = f'Invalid time format: {str(e)}'
                print(f'Error: {error_msg}, skipping {fn}')
                failed_log.write(f'{fn} | {current_utc} | {error_msg}\n')
                failed_log.flush()
                # Write to match log (save_fn empty, status ERROR)
                with open(match_log_path, 'a', encoding='utf-8') as match_log:
                    match_log.write(f'{fn} | {save_fn} | {current_utc} | {match_status}\n')
                    match_log.flush()
                continue

            query = None
            search_failed = False
            search_error_msg = ''
            for retry in range(MAX_RETRIES):
                try:
                    print(f'Search attempt {retry+1}/{MAX_RETRIES}...')
                    query = Fido.search(
                        a.Time(sear_time - 1*u.s, sear_time + 1*u.s),
                        a.Sample(360*u.s),
                        a.jsoc.Series.hmi_m_720s,
                        a.jsoc.Notify(jsoc_email)
                    )
                    break
                except urllib.error.URLError as e:
                    search_error_msg = f'Network error (search): {str(e)}'
                    print(f'Error: {search_error_msg}')
                    if retry < MAX_RETRIES - 1:
                        print(f'Retrying in {RETRY_DELAY} seconds...')
                        time.sleep(RETRY_DELAY)
                    else:
                        print(f'Max retries reached, skipping {fn}')
                        search_failed = True
                except Exception as e:
                    search_error_msg = f'Unexpected error (search): {str(e)}'
                    print(f'Error: {search_error_msg}, skipping {fn}')
                    search_failed = True
                    break

            if search_failed:
                failed_log.write(f'{fn} | {current_utc} | {search_error_msg}\n')
                failed_log.flush()
                # Write to match log (save_fn empty, status ERROR)
                with open(match_log_path, 'a', encoding='utf-8') as match_log:
                    match_log.write(f'{fn} | {save_fn} | {current_utc} | {match_status}\n')
                    match_log.flush()
                continue

            if query is None:
                no_data_msg = 'Empty search result (no data matched)'
                print(f'Info: {no_data_msg} for {fn}')
                failed_log.write(f'{fn} | {current_utc} | {no_data_msg}\n')
                failed_log.flush()
                match_status = "MISSING"
                # Write to match log (save_fn empty, status MISSING)
                with open(match_log_path, 'a', encoding='utf-8') as match_log:
                    match_log.write(f'{fn} | {save_fn} | {current_utc} | {match_status}\n')
                    match_log.flush()
                continue

            print(f'Search result: {query}')

            if len(query) > 0 and len(query[0]) > 0:
                file_info = query[0][0]
                T_REC = file_info['T_REC']
                
                save_fn = f'hmi.m_720s.{T_REC[0:4]}{T_REC[5:7]}{T_REC[8:10]}_{T_REC[11:13]}{T_REC[14:16]}{T_REC[17:19]}_TAI.1.magnetogram.fits'
                match_status = "FOUND"  # Update status to FOUND
                save_path = os.path.join(save_dir, save_fn)
                
                # Write to match log (save_fn filled, status FOUND)
                with open(match_log_path, 'a', encoding='utf-8') as match_log:
                    match_log.write(f'{fn} | {save_fn} | {current_utc} | {match_status}\n')
                    match_log.flush()
                
                if os.path.exists(save_path):
                    print(f'File {save_fn} already exists, skipping download')
                    continue
            
                download_success = False
                download_error_msg = ''
                for retry in range(MAX_RETRIES):
                    try:
                        print(f'Download attempt {retry+1}/{MAX_RETRIES}...')
                        files = Fido.fetch(query[0][0], path=save_dir, overwrite=False)
                        print(f'Saved to {files}')
                        download_success = True
                        break
                    except urllib.error.URLError as e:
                        download_error_msg = f'Network error (download): {str(e)}'
                        print(f'Error: {download_error_msg}')
                        if retry < MAX_RETRIES - 1:
                            print(f'Retrying in {RETRY_DELAY} seconds...')
                            time.sleep(RETRY_DELAY)
                    except Exception as e:
                        download_error_msg = f'Unexpected error (download): {str(e)}'
                        print(f'Error: {download_error_msg}')
                        break
                
                if not download_success:
                    print(f'Failed to download {save_fn} after {MAX_RETRIES} retries')
                    failed_log.write(f'{fn} | {current_utc} | {download_error_msg}\n')
                    failed_log.flush()
            else:
                no_data_msg = 'No corresponding HMI data found (search returned empty)'
                print(f'Info: {no_data_msg} for {fn}')
                failed_log.write(f'{fn} | {current_utc} | {no_data_msg}\n')
                failed_log.flush()
                match_status = "MISSING"
                # Write to match log (save_fn empty, status MISSING)
                with open(match_log_path, 'a', encoding='utf-8') as match_log:
                    match_log.write(f'{fn} | {save_fn} | {current_utc} | {match_status}\n')
                    match_log.flush()

print(f'\n===== Processing Completed =====')
print(f'Failure log saved to: {failed_log_path}')
print(f'Mapping log saved to: {match_log_path}')