import os
import pandas as pd

root_dir = 'E:/Research/Program/Magnetogram_SuperResol_by_ML/ltew/apply_20260416_5/'
png_dir = os.path.join(root_dir, 'test_pred/valid/')
csv_path = os.path.join(root_dir, 'matched_pairs.csv')
column_name = 'output_filename'

print('='*60)
print('Checking repeating or not ...')

df = pd.read_csv(csv_path)

check_columns = ['input_filename', 'output_filename']

for col in check_columns:
    print(f'Column: {col}')
    
    value_counts = df[col].value_counts()
    duplicate_values = value_counts[value_counts > 1]
    
    if len(duplicate_values) == 0:
        print(f'No repeating in Column {col} with {len(df)} filename')
    else:
        total_duplicate_rows = duplicate_values.sum()
        print(f'===!!! Repeating in Column {col} !')
        print(f'{len(duplicate_values)} repeating filename in {total_duplicate_rows} lines')
        for filename, count in duplicate_values.items():
            print(f'{filename} repeats for {count} times')

print('='*60)
print('Checking matched or not ...')

png_files = [f[:-4] for f in os.listdir(png_dir) if f.endswith('.png')]
png_set = set(png_files)

df = pd.read_csv(csv_path)
npz_files = df[column_name].astype(str).str.replace('.npz', '', regex=False).tolist()
npz_set = set(npz_files)

print(f'Total PNG filenames: {len(png_set)}')
print(f'Toral NPZ filenames: {len(npz_set)}')
print('='*60)

only_png = sorted(png_set - npz_set)
only_npz = sorted(npz_set - png_set)

if len(only_png) == 0 and len(only_npz) == 0:
    print('All matched! ')
else:
    print(f'Not matched!')
    print('Only PNG filename: ')
    for name in only_png:
        print(f'  {name}.png')

    print('Only NPZ filename: ')
    for name in only_npz:
        print(f'  {name}.npz')