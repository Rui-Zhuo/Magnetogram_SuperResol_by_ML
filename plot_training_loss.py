import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

run_dir = 'E:/Research/Program/Magnetogram_SuperResol_by_ML/ltew/'
run_id = 'train_20260416/'
tb_dir = 'tensorboard/loss_epoch/'
# loss_fn = 'events.out.tfevents.1776656059.k28g30'
# loss_fn = 'events.out.tfevents.1776737006.k28g32'
loss_fn = 'events.out.tfevents.1776737006.k28g32'
save_or_not = 1

train_loss_dir = 'train_l1_cc/'
train_loss_fn = loss_fn
train_loss_path = os.path.join(run_dir, run_id, tb_dir, train_loss_dir, train_loss_fn)

val_loss_dir = 'val_l1_cc/'
val_loss_fn = loss_fn
val_loss_path = os.path.join(run_dir, run_id, tb_dir, val_loss_dir, val_loss_fn)

def extract_loss(event_file):    
    loss_data = []
    
    for event in tf.compat.v1.train.summary_iterator(event_file):
        if not event.HasField('summary'):
            continue
        
        step = event.step
        wall_time = datetime.fromtimestamp(event.wall_time).strftime('%Y-%m-%d %H:%M:%S')
        
        for value in event.summary.value:
            if value.HasField('simple_value'):
                loss_value = value.simple_value
                loss_data.append({
                    'step': step,
                    'loss': loss_value,
                    'record_time': wall_time
                })
    
    df = pd.DataFrame(loss_data)
    df = df.sort_values(by='step').reset_index(drop=True)
    return df

def extract_training_time(df):
    record_time = df['record_time']

    beg_time_str = record_time.iloc[0]   # 第一个
    end_time_str = record_time.iloc[-1] 

    fmt = '%Y-%m-%d %H:%M:%S'
    beg_time = datetime.strptime(beg_time_str, fmt)
    end_time = datetime.strptime(end_time_str, fmt)

    time_diff = end_time - beg_time

    return time_diff

if __name__ == '__main__':
    train_loss_df = extract_loss(train_loss_path)
    val_loss_df = extract_loss(val_loss_path)
    
    time_diff = extract_training_time(train_loss_df)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    step = train_loss_df['step']
    beg_epoch = step.iloc[0]
    end_epoch = step.iloc[-1]

    ax.plot(train_loss_df['step'], train_loss_df['loss'], linewidth=2, label='Train loss')
    ax.plot(val_loss_df['step'], val_loss_df['loss'], linewidth=2, label='Val loss')
    
    ax.set_xlabel('Training Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.grid()
    ax.legend(fontsize=12)
    ax.set_title(f'{run_id[:-1]} Epoch {beg_epoch}-{end_epoch} ({time_diff.total_seconds() / 3600:.2f}h)', fontsize=14)
    
    plt.ylim(0,1)
    plt.tight_layout()
    
    if save_or_not == 1:
        save_dir = os.path.join(run_dir, run_id)
        save_png_fn = f'loss_epoch.{beg_epoch}-{end_epoch}.png'
        plt.savefig(save_dir + save_png_fn)
        
        loss_data = pd.DataFrame({
            'train_epoch': train_loss_df['step'],
            'train_loss': train_loss_df['loss'],
            'val_epoch': val_loss_df['step'],
            'val_loss': val_loss_df['loss']
        })

        save_csv_fn = f'loss_epoch.{beg_epoch}-{end_epoch}.csv'
        loss_data.to_csv(save_dir + save_csv_fn, index=False, encoding='utf-8-sig')

        print(f'Saved loss during {beg_epoch}-{end_epoch} to: {save_dir}')
    
    plt.show()
    
    db