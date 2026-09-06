import os
import json
import random

root_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_ML/dataset/L72/'
data_dir = os.path.join(root_dir, 'saveData/')
num_sample = 10115  # 10115
train_ratio = 0.6
val_ratio = 0.2 
test_ratio = 0.2
seed = 42         

json_fn = f'dataset_split_{num_sample}_{train_ratio}_{val_ratio}_{test_ratio}.json'
json_path = os.path.join(data_dir, json_fn)

all_samples = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
print(f'{len(all_samples)} samples found in {data_dir}')

random.seed(seed)
random.shuffle(all_samples)

slct_sample = all_samples[:num_sample]
print(f'Selected {len(slct_sample)} samples for splitting.')

train_num = int(num_sample * train_ratio)
val_num = int(num_sample * val_ratio)

train_sample = slct_sample[:train_num]
val_sample = slct_sample[train_num : train_num + val_num]
test_sample = slct_sample[train_num + val_num :]

split_dict = {
    'train': train_sample,
    'val': val_sample,
    'test': test_sample
}

os.makedirs(os.path.dirname(json_path), exist_ok=True)
with open(json_path, 'w') as f:
    json.dump(split_dict, f, indent=2)

print('Split done!')
print(f'{len(train_sample)} samples ({train_ratio*100:.1f}%) for training')
print(f'{len(val_sample)} samples ({val_ratio*100:.1f}%) for validation')
print(f'{len(test_sample)} samples ({test_ratio*100:.1f}%) for testing')
print(f'Saved JSON file to: {json_path}')