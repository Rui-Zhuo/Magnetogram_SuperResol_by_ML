import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def epoch_to_date(epoch):
    return datetime.fromtimestamp(epoch).strftime('%Y-%m-%d')

root_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_ML/dataset/L72/'
data_dir = os.path.join(root_dir, 'saveData/')
save_or_not = 0

# list all npz files
file_list = [fn for fn in os.listdir(data_dir) if fn.endswith('.npz')]
print(f'{len(file_list)} npz files found in {data_dir}')    

# Check if have saved csv file, if not, process the data and save it
csv_path = os.path.join(root_dir, 'dataset_distribution.csv')
if os.path.exists(csv_path):
    print(f'CSV file already exists, skip processing and load it directly')
    df = pd.read_csv(csv_path)
    epoch_lst = df['Epoch'].tolist()
    dist_lst = df['Distance'].tolist()
else:
    epoch_lst = []
    dist_lst = []
    for fn in file_list:
        if fn.endswith('.npz'):
            print(f'Processing {fn}...')
            
            fields = np.load(os.path.join(data_dir, fn))
            HMIfield = fields['HMIfield']
            SPfield = fields['SPfield']
            Txy = fields['Txy']
            
            height, width = HMIfield.shape
            
            # Extract date and time
            datetime_str = fn[:-4]
            date_str, time_str = datetime_str.split('_')
            date_formatted = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}'
            time_formatted = f'{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}'
            
            # Convert to epoch time
            dt = datetime.strptime(f'{date_formatted} {time_formatted}', '%Y-%m-%d %H:%M:%S')
            epoch_sub = int(dt.timestamp())
            epoch_lst.append(epoch_sub)
            
            # extract the center Txy coordinate
            dist_sub = Txy[height//2, width//2]
            dist_lst.append(dist_sub)

    # Save the epoch and distance lists to a csv file
    df = pd.DataFrame({'Epoch': epoch_lst, 'Distance': dist_lst})
    df.to_csv(csv_path, index=False)
    print(f'Saved CSV file to: {csv_path}')

# Convert distance to degrees
deg_lst = np.arcsin(np.array(dist_lst) / 960) * 180 / np.pi

# Read sunspot number data
ssn_csv_path = 'E:/Research/Data/ROB/SILSO/SSN/SN_ms_tot_V2.0_202410.csv'

# Extract the datetime and the sunspot number
ssn_df = pd.read_csv(ssn_csv_path, sep=';', header=None, names=['Year', 'Month', 'DecimalYear', 'SunspotNumber', 'StdDev', 'Observations', 'Provisional'])
year_ssn = ssn_df['Year']
month_ssn = ssn_df['Month']
ssn_value = ssn_df['SunspotNumber']

epoch_ssn_lst = []
ssn_lst = []
for year, month, ssn_val in zip(year_ssn, month_ssn, ssn_value):
    if year >= 1970:
        dt = datetime.strptime(f'{year}-{month}-15 00:00:00', '%Y-%m-%d %H:%M:%S')
        epoch_sub = int(dt.timestamp())
        epoch_ssn_lst.append(epoch_sub)
        ssn_lst.append(ssn_val)

# Plot the histogram of the epoch and distance distribution
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.hist(epoch_lst, bins=11*12, color='blue', alpha=0.7, density=False, linewidth=0.5)
plt.xticks(ticks=[epoch for epoch in epoch_lst if datetime.fromtimestamp(epoch).month == 1 and datetime.fromtimestamp(epoch).day == 1],
           labels=[epoch_to_date(epoch) for epoch in epoch_lst if datetime.fromtimestamp(epoch).month == 1 and datetime.fromtimestamp(epoch).day == 1],
           rotation=45)
plt.ylabel('Sample Counts')
plt.twinx()
plt.plot(epoch_ssn_lst, ssn_lst, color='red', label='Sunspot Number')
plt.ylim(0, 150)
plt.ylabel('Sunspot Number')
plt.legend(loc='upper right')
plt.xlim(min(epoch_lst), max(epoch_lst))
plt.grid(which='both', alpha=0.2, linewidth=0.5)
# plt.minorticks_on()

plt.subplot(2, 1, 2)
plt.hist(deg_lst, bins=np.arange(0, 90, 1), color='orange', alpha=0.7, density=False, linewidth=0.5)
plt.xlim(0, 90)
plt.xlabel('Angle from Disk Center (degrees)')
plt.ylabel('Sample Counts')
plt.grid(which='both', alpha=0.2, linewidth=0.5)
# plt.minorticks_on()

plt.suptitle('Dataset Distribution')
plt.tight_layout()
png_path = os.path.join(root_dir, 'dataset_distribution.png')
if save_or_not == 1:
    plt.savefig(png_path, dpi=300)
plt.show()

db