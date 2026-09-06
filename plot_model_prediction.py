import os
import csv
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from constants import h_shape_lr, w_shape_lr, h_shape_hr, w_shape_hr, gt_norm
from utils_load_data import load_output_sample
from utils_plot_field import plot_field_map, plot_field_distribution
from utils_eval import get_bicubic_map, get_empirical_map_from_dalda2017, get_empirical_map_from_zhang2023
from utils_eval import eval_metrics, eval_flux

if __name__ == '__main__':
    run_dir = 'E:/Research/Program/Magnetogram_SuperResol_by_ML/ltew/'
    # run_id = 'train_20260416/'
    # run_id = 'test_20260416/'
    run_id = 'apply_20260416_5'
    
    dataset_type = 'test' # 'train'; 'val'; 'test'
    match_fn = 'matched_pairs.csv'
    save_or_not = 1
    
    # sample_id_lst = np.arange(1, 31) # for 'train'
    # sample_id_lst = np.arange(1, 11) # for 'val'
    # sample_id_lst = np.arange(1, 2024) # for 'test'
    sample_id_lst = np.arange(1, 401) # for 'apply'
    
    # sample_id_slt = 1576 # 1738, 1576
    # sample_id_lst = np.arange(sample_id_slt, sample_id_slt+1) # for show
        
    if match_fn is not None: 
        print(f'Reading matching-table from file: {match_fn}...')
        with open(os.path.join(run_dir, run_id, match_fn), 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            file_list = [(row[0].strip(), row[1].strip()) for row in reader if len(row) >= 2]
    
    for sample_id in sample_id_lst:
        sample_pattern = os.path.join(run_dir, run_id, f'{dataset_type}_pred/', f'{dataset_type}_sample_{sample_id}_idx_*.npz')
        sample_files = glob.glob(sample_pattern)
        data_fn = sample_files[0]

        data_fn_base = os.path.basename(data_fn)
        matched_original_name = None
        for orig_name, gen_name in file_list:
            if gen_name == data_fn_base:
                matched_original_name = orig_name
                break
        
        matched_name_noext = matched_original_name.replace('.npz', '') if matched_original_name else f'unknown_{sample_id}'
        
        sample_save_path = os.path.join(run_dir, run_id, f'{dataset_type}_pred/', f'{matched_name_noext}.png')
        if save_or_not == 1:
            if os.path.exists(sample_save_path):
                print(f'Existing sample {sample_id} in: {sample_save_path}')
                continue
        
        data_path = os.path.join(run_dir, run_id, f'{dataset_type}_pred/', data_fn)
        inp, gt, pred, sample_idx = load_output_sample(data_path)
        
        inp_field = inp[0]
        gt_field = gt
        
        LR_map = np.squeeze(inp_field)
        HR_map = gt_field.reshape((h_shape_hr, w_shape_hr)) / gt_norm
        SR_map = pred.reshape((h_shape_hr, w_shape_hr))
        bic_map = get_bicubic_map(LR_map, h_shape_hr, w_shape_hr)
        
        mae_SR, rmse_SR, cc_SR, r2_SR, ssim_SR, psnr_SR = eval_metrics(HR_map, SR_map)
        mae_SR, rmse_bic, cc_bic, r2_bic, ssim_bic, psnr_bic = eval_metrics(HR_map, bic_map)
        flux_net_LR, flux_uns_LR = eval_flux(LR_map)
        flux_net_HR, flux_uns_HR = eval_flux(HR_map)
        flux_net_SR, flux_uns_SR = eval_flux(SR_map)
        flux_net_bic, flux_uns_bic = eval_flux(bic_map)
        
        HR_std = np.std(HR_map,  ddof=0)
        
        # plot figures
        # plt.figure(figsize=(14, 10))
        # gs = GridSpec(3, 4, figure=plt.gcf(), hspace=0.2, wspace=0.2, height_ratios=[2, 2, 1])
        plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 4, figure=plt.gcf())
        
        abs_max = max(np.max(np.abs(SR_map)), np.max(np.abs(HR_map)))
        vlim = [-0.8*abs_max, 0.8*abs_max]

        plt.subplot(gs[0, 0])
        text = r'$\Phi_n$/$\Phi_{n0}$=' f'{((flux_net_LR / flux_net_HR)*100):.0f}% \n' \
            r'$\Phi_u$/$\Phi_{u0}$=' f'{((flux_uns_LR / flux_uns_HR)*100):.0f}%'
        plot_field_map(LR_map, 'LR (HMI)', cmap='bwr', vlim=vlim, text=text)

        plt.subplot(gs[0, 1])
        text = r'$\Phi_n$/$\Phi_{n0}$=' f'{((flux_net_bic / flux_net_HR)*100):.0f}% \n' \
            r'$\Phi_u$/$\Phi_{u0}$=' f'{((flux_uns_bic / flux_uns_HR)*100):.0f}%'
        plot_field_map(bic_map, 'Bicubic', cmap='bwr', vlim=vlim, text=text)

        plt.subplot(gs[0, 2])
        text = f'RMSE={rmse_bic:.2f}G \n' r'RMSE/$\sigma_0$='f'{(rmse_bic/HR_std):.2f}'
        plot_field_map(bic_map - HR_map, '(Bicubic - HR)', cmap='PiYG', vlim=[-1000, 1000], text=text)

        plt.subplot(gs[0, 3])
        text = f'CC={cc_bic:.4f} \nSSIM={ssim_bic:.4f} \nR2={r2_bic:.4f} \nPSNR={psnr_bic:.2f}dB'
        plot_field_distribution(HR_map, bic_map, xlabel='HR (G)', ylabel='Bicubic (G)', bins=300, vmax=100, text=text)

        plt.subplot(gs[1, 0])
        text = r'$\sigma_0$=' f'{HR_std:.2f}G \n' r'$\Phi_{n0}$=' f'{flux_net_HR:.2e}Wb \n' r'$\Phi_{u0}$=' f'{flux_uns_HR:.2e}Wb'
        plot_field_map(HR_map, 'HR (SP)', cmap='bwr', vlim=vlim, text=text)

        plt.subplot(gs[1, 1])
        text = r'$\Phi_n$/$\Phi_{n0}$=' f'{((flux_net_SR / flux_net_HR)*100):.0f}% \n' \
            r'$\Phi_u$/$\Phi_{u0}$=' f'{((flux_uns_SR / flux_uns_HR)*100):.0f}%'
        plot_field_map(SR_map, 'SR (output)', cmap='bwr', vlim=vlim, text=text)

        plt.subplot(gs[1, 2])
        text = f'RMSE={rmse_SR:.2f}G \n' r'RMSE/$\sigma_0$='f'{(rmse_SR/HR_std):.2f}'
        plot_field_map(SR_map - HR_map, '(SR - HR)', cmap='PiYG', vlim=[-1000, 1000], text=text)

        plt.subplot(gs[1, 3])
        text = f'CC={cc_SR:.4f} \nSSIM={ssim_SR:.4f} \nR2={r2_SR:.4f} \nPSNR={psnr_SR:.2f}dB'
        plot_field_distribution(HR_map, SR_map, xlabel='HR (G)', ylabel='SR (G)', bins=300, vmax=100, text=text)

        # plt.subplot(gs[2, :])
        # bin_width = 20
        # max_Br = max(np.max(SR_map), np.max(HR_map))
        # min_Br = max(np.min(SR_map), np.min(HR_map))
        # bins = np.arange(np.floor(min_Br / bin_width) * bin_width,
        #                 np.ceil(max_Br / bin_width) * bin_width + bin_width,
        #                 bin_width)
        # plt.hist(HR_map.flatten(), bins=bins, density=True, color='k', alpha=0.6, label=r'HR        ($\sigma_0$='f'{HR_std:.2f}G)')
        # plt.hist(SR_map.flatten(), bins=bins, density=True, color='r', alpha=0.6, label=r'SR        (RMSE/$\sigma_0$='f'{(rmse_SR/HR_std):.2f})')
        # plt.hist(bic_map.flatten(), bins=bins, density=True, color='b', alpha=0.6, label=r'Bicubic (RMSE/$\sigma_0$='f'{(rmse_bic/HR_std):.2f})')
        # plt.xlim(np.floor(min_Br/100) * 100, np.ceil(max_Br/100) * 100)
        # plt.xlabel('Magnetic Field Strength (G)', fontsize=12)
        # plt.ylabel('Probability Density', fontsize=12)
        # # plt.yscale('log')
        # plt.grid(which='both', alpha=0.2, linewidth=0.5)
        # plt.minorticks_on()
        # plt.legend(fontsize=12)
        # plt.title('Magnetic Field Distribution', fontsize=14)
        
        plt.suptitle(f'{dataset_type} sample #{sample_id} @{matched_name_noext}', fontsize=14)

        plt.tight_layout()
        
        if save_or_not == 0:
            plt.show()
        elif save_or_not == 1:
            plt.savefig(sample_save_path)
            plt.close('all') 
            print(f'Saved sample {sample_id} to: {sample_save_path}')
