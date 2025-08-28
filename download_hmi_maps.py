import requests
import os

save_dir = 'E:/Research/Data/SDO/HMI/Magnetogram/hr/'
year_record = '2014'

# Iterating for time range and parameters
for datehour in ['0101_09', '0102_03', '0105_00', '0105_14', '0106_21', '0107_03', '0108_22', '0109_04', \
    '0109_16', '0110_03', '0111_07', '0111_19', '0116_07', '0116_10', '0116_13', '0116_17', '0116_22', \
        '0125_09', '0125_12', '0127_09', '0127_10', '0127_13', '0127_22', '0128_01']:
        
    # Getting time record
    time_record = year_record + datehour + '0000'
    name_record = 'hmi.M_720s.' + time_record + '_TAI'
    print('Downloading ' + name_record + ' with requests')
    
    # Setting download url (submitted request on JSOC first)
    month_record = datehour[0:2]
    date_record = datehour[2:4]
    
    url = 'https://jsoc1.stanford.edu/data/hmi/fits/' + year_record + '/' + month_record + '/' + \
        date_record + '/' + name_record + '.fits'
    
    # Downloading
    r = requests.get(url)
    file_name = url.split('/')[-1]
    file_path = os.path.join(save_dir, file_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(file_path, 'wb') as file:
        file.write(r.content)
    
    # Checking download    
    print('Saved ' + name_record)