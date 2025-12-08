import requests
import os
import traceback
from astropy.time import Time

def download_data(sear_dir, save_dir, prefix, lv, failed_log_path):
    for fn in os.listdir(sear_dir):
        if fn.startswith(prefix):
            sear_year = fn[0:4]
            sear_month = fn[4:6]
            sear_date = fn[6:8]
            name_record = fn[:-5]
            current_utc = Time.now().isot  # Record UTC time for log
            
            save_path = os.path.join(save_dir, fn)
            
            if os.path.exists(save_path):
                print(f'File {fn} already exists, skipping download')
                continue
            
            print(f'Downloading {fn} with requests')
            
            # Construct URL
            if lv == 2:
                url = pub + sear_year + '/' + sear_month + '/' + sear_date + '/' \
                    + ins + '/' + name_record + '/' + name_record + '.fits'
            elif lv == 21:
                url = pub + sear_year + '/' + sear_month + '/' + sear_date + '/' \
                    + ins + '/' + name_record + '/' + name_record + '_L2.1.fits'            
            else:
                error_msg = f'Invalid level value: {lv}'
                print(f'Error: {error_msg}, skipping {fn}')
                with open(failed_log_path, 'a', encoding='utf-8') as failed_log:
                    failed_log.write(f'{fn} | {current_utc} | {error_msg}\n')
                    failed_log.flush()
                continue
            
            try:
                r = requests.get(url, timeout=30)  # Add timeout to avoid hanging
                r.raise_for_status()  # Raise HTTP errors (4xx, 5xx)
                
                with open(save_path, 'wb') as file:
                    file.write(r.content)
                
                # Verify file exists after download
                if os.path.exists(save_path):
                    print(f'Successfully saved {fn}')
                else:
                    raise Exception('File not created after download')
                
            except Exception as error:
                # Capture detailed error info (truncate long traceback)
                error_msg = f"{type(error).__name__}: {str(error)}\n{traceback.format_exc()[:500]}..."
                print(f'Failed to download {fn}: {str(error)}, skipping to next file')
                
                # Write to failure log
                with open(failed_log_path, 'a', encoding='utf-8') as failed_log:
                    failed_log.write(f'{fn} | {current_utc} | {error_msg.replace("|", "-")}\n')
                    failed_log.flush()

def check_download(sear_dir, save_dir):
    sear_files = [f for f in os.listdir(sear_dir) if os.path.isfile(os.path.join(sear_dir, f)) and f.startswith(prefix)]
    save_files = [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f)) and f.startswith(prefix)]
    
    sear_files_sorted = sorted(sear_files)
    save_files_sorted = sorted(save_files)
    
    if sear_files_sorted == save_files_sorted:
        print('All files are downloaded!')
    else:
        missing_files = set(sear_files_sorted) - set(save_files_sorted)
        print(f'Not all files are downloaded. Missing files count: {len(missing_files)}')

# Configuration
# sear_dir = 'E:/HinodeSOTSPLevel2Update/Main/2014/' # for test
sear_dir = 'E:/HinodeSOTSPLevel2Update/Main/2019/'
lv = 2
if lv == 2:
    # save_dir = 'E:/Research/Data/HINODE/SP/L2/2014/' # for test
    save_dir = 'E:/Research/Data/HINODE/SP/L2/2019/'
    pub = 'https://data.darts.isas.jaxa.jp/pub/hinode/sot/level2hao/'
elif lv == 21:
    # save_dir = 'E:/Research/Data/HINODE/SP/L2.1/2014/' # for test
    save_dir = 'E:/Research/Data/HINODE/SP/L2.1/2019/'
    pub = 'https://data.darts.isas.jaxa.jp/pub/hinode/sot/level2.1hao/'
ins = 'SP3D'

# sear_year = '2014' # for test
sear_year = '2019'
prefix = sear_year

# Create save directory if not exists
os.makedirs(save_dir, exist_ok=True)

# Failure log configuration (consistent with previous scripts)
failed_log_path = os.path.join(save_dir, f'download_failed_{sear_year}.txt')

# Initialize failure log (append mode, no overwriting)
with open(failed_log_path, 'a', encoding='utf-8') as failed_log:
    if os.path.getsize(failed_log_path) == 0:
        failed_log.write(f'===== Download Failure Log ({sear_year}) =====\n')
        failed_log.write('Log Time Format: UTC\n')
        failed_log.write('Format: Filename | Failure Time | Error Message\n')
        failed_log.write('-' * 80 + '\n')
        failed_log.flush()

# Execute download and check
download_data(sear_dir, save_dir, prefix, lv, failed_log_path)
check_download(sear_dir, save_dir)

print(f'\n===== Processing Completed =====\nFailure log saved to: {failed_log_path}')