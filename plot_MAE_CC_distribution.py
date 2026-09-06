import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plot_mode = 1

run_dir = 'E:/Research/Program/Magnetogram_SuperResol_by_ML/ltew/test_20260416/'
eval_fn = 'eval_test_results_all_theta.0-90.num.2023.csv'
eval_path = os.path.join(run_dir, eval_fn)
df = pd.read_csv(eval_path)

mae_sr = df['MAE_SR'].values
mae_bic = df['MAE_bic'].values
mae_emp_dalda = df['MAE_emp_dalda'].values
mae_emp_zhang = df['MAE_emp_zhang'].values

cc_sr = df['CC_SR'].values
cc_bic = df['CC_Bic'].values
cc_emp_dalda = df['CC_emp_dalda'].values
cc_emp_zhang = df['CC_emp_zhang'].values

mae_sr_sorted = np.sort(mae_sr)
mae_bic_sorted = np.sort(mae_bic)
mae_emp_dalda_sorted = np.sort(mae_emp_dalda)
mae_emp_zhang_sorted = np.sort(mae_emp_zhang)

cc_sr_sorted = np.sort(cc_sr)[::-1]
cc_bic_sorted = np.sort(cc_bic)[::-1]
cc_emp_dalda_sorted = np.sort(cc_emp_dalda)[::-1]
cc_emp_zhang_sorted = np.sort(cc_emp_zhang)[::-1]

if plot_mode == 0:
    
    percent = np.linspace(0, 100, len(mae_sr))

    plt.figure(figsize=(10, 5))

    ax1 = plt.gca()
    line1 = ax1.plot(percent, mae_sr_sorted, label='MAE (SR)', color='r', linewidth=2)
    line2 = ax1.plot(percent, mae_bic_sorted, label='MAE (Bicubic)', color='b', linewidth=2)
    line3 = ax1.plot(percent, mae_emp_dalda_sorted, label='MAE (Dalda et al. 2017)', color='orange', linewidth=2)
    line4 = ax1.plot(percent, mae_emp_zhang_sorted, label='MAE (Zhang et al. 2023)', color='g', linewidth=2)
    ax1.set_xlabel('Cumulative Percentage (%)', fontsize=12)
    ax1.set_ylabel('Mean Absolute Error (G)', fontsize=12, color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.set_ylim(0, 500)
    ax1.grid(alpha=0.2)

    ax1.axvline(x=25, color='gray', linestyle='-', linewidth=2, alpha=0.6)
    ax1.axvline(x=50, color='gray', linestyle='-', linewidth=2, alpha=0.6)
    ax1.axvline(x=75, color='gray', linestyle='-', linewidth=2, alpha=0.6)

    ax2 = ax1.twinx()
    line5 = ax2.plot(percent, cc_sr_sorted, label='CC (SR)', color='r', linestyle='--', linewidth=2)
    line6 = ax2.plot(percent, cc_bic_sorted, label='CC (Bicubic)', color='b', linestyle='--', linewidth=2)
    line7 = ax2.plot(percent, cc_emp_dalda_sorted, label='CC (Emp Dalda et al. 2017)', color='orange', linestyle='--', linewidth=2)
    line8 = ax2.plot(percent, cc_emp_zhang_sorted, label='CC (Emp Zhang et al. 2023)', color='g', linestyle='--', linewidth=2)
    ax2.set_ylabel('Correlation Coefficient', fontsize=12, color='k')
    ax2.tick_params(axis='y', labelcolor='k')
    ax2.set_ylim(0, 1)

    ax1.set_xticks(np.arange(0, 101, 5), minor=True)
    ax1.set_xticks(np.arange(0, 101, 10), minor=False)
    ax1.set_yticks(np.arange(0, 501, 25), minor=True)
    ax1.set_yticks(np.arange(0, 501, 50), minor=False) 
    ax2.set_yticks(np.arange(0, 1.01, 0.05), minor=True)
    ax2.set_yticks(np.arange(0, 1.01, 0.1), minor=False)
    ax1.grid(which='both', alpha=0.2, linewidth=0.5)
    ax2.grid(which='both', alpha=0.2, linewidth=0.5)
    ax1.minorticks_on()
    ax2.minorticks_on()

    lines = line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', fontsize=11)
    
    plt.title('Distribution of MAE and CC in test dataset', fontsize=13)

elif plot_mode == 1: 
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    bins_mae_sr    = np.linspace(1, 500, 51)
    bins_mae_bic   = np.linspace(1, 500, 51) + 0.5
    bins_mae_dalda = np.linspace(1, 500, 51) + 1.0
    bins_mae_zhang = np.linspace(1, 500, 51) + 1.5

    # plt.hist(mae_sr_sorted, bins=bins_mae_sr, density=True, color='r', alpha=0.6, label='SR model')
    # plt.hist(mae_bic_sorted, bins=bins_mae_bic, density=True, color='b', alpha=0.6, label='Bicubic')
    # plt.hist(mae_emp_dalda_sorted, bins=bins_mae_dalda, density=True, color='orange', alpha=0.6, label='Empirical (Dalda et al. 2017)')
    # plt.hist(mae_emp_zhang_sorted, bins=bins_mae_zhang, density=True, color='g', alpha=0.6, label='Empirical (Zhang et al. 2023)')


    plt.hist(mae_sr_sorted, bins=bins_mae_sr, density=True, color='r', linewidth=2, histtype='step', label='PM-LTEW')
    plt.hist(mae_bic_sorted, bins=bins_mae_bic, density=True, color='b', linewidth=2, histtype='step', label='Bicubic')
    plt.hist(mae_emp_dalda_sorted, bins=bins_mae_dalda, density=True, color='orange', linewidth=2, histtype='step', label='Empirical-D')
    plt.hist(mae_emp_zhang_sorted, bins=bins_mae_zhang, density=True, color='g', linewidth=2, histtype='step', label='Empirical-Z')

    mean_sr_mae = np.mean(mae_sr_sorted)
    mean_bic_mae = np.mean(mae_bic_sorted)
    mean_dalda_mae = np.mean(mae_emp_dalda_sorted)
    mean_zhang_mae = np.mean(mae_emp_zhang_sorted)

    plt.axvline(mean_sr_mae, color='r', linestyle=':', linewidth=2)
    plt.axvline(mean_bic_mae, color='b', linestyle=':', linewidth=2)
    plt.axvline(mean_dalda_mae, color='orange', linestyle=':', linewidth=2)
    plt.axvline(mean_zhang_mae, color='g', linestyle=':', linewidth=2)

    plt.xlabel('Mean Absolute Error (G)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    # plt.yscale('log')
    plt.xlim(1, 500)
    plt.grid(which='both', alpha=0.2, linewidth=0.5)
    plt.minorticks_on()
    plt.legend(fontsize=10)

    plt.subplot(1, 2, 2)
    bins_cc_sr     = np.linspace(0, 1, 51)
    bins_cc_bic    = np.linspace(0, 1, 51) + 0.001
    bins_cc_dalda  = np.linspace(0, 1, 51) + 0.002
    bins_cc_zhang  = np.linspace(0, 1, 51) + 0.003

    # plt.hist(cc_sr_sorted, bins=bins_cc_sr, density=True, color='r', alpha=0.6, label='SR model')
    # plt.hist(cc_bic_sorted, bins=bins_cc_bic, density=True, color='b', alpha=0.6, label='Bicubic')
    # plt.hist(cc_emp_dalda_sorted, bins=bins_cc_dalda, density=True, color='orange', alpha=0.6, label='Empirical (Dalda et al. 2017)')
    # plt.hist(cc_emp_zhang_sorted, bins=bins_cc_zhang, density=True, color='g', alpha=0.6, label='Empirical (Zhang et al. 2023)')
    

    plt.hist(cc_sr_sorted, bins=bins_cc_sr, density=True, color='r', linewidth=2, histtype='step', label='PM-LTEW')
    plt.hist(cc_bic_sorted, bins=bins_cc_bic, density=True, color='b', linewidth=2, histtype='step', label='Bicubic')
    plt.hist(cc_emp_dalda_sorted, bins=bins_cc_dalda, density=True, color='orange', linewidth=2, histtype='step', label='Empirical-D')
    plt.hist(cc_emp_zhang_sorted, bins=bins_cc_zhang, density=True, color='g', linewidth=2, histtype='step', label='Empirical-Z')
    
    mean_sr_cc = np.mean(cc_sr_sorted)
    mean_bic_cc = np.mean(cc_bic_sorted)
    mean_dalda_cc = np.mean(cc_emp_dalda_sorted)
    mean_zhang_cc = np.mean(cc_emp_zhang_sorted)

    plt.axvline(mean_sr_cc, color='r', linestyle=':', linewidth=2)
    plt.axvline(mean_bic_cc, color='b', linestyle=':', linewidth=2)
    plt.axvline(mean_dalda_cc, color='orange', linestyle=':', linewidth=2)
    plt.axvline(mean_zhang_cc, color='g', linestyle=':', linewidth=2)

    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    # plt.yscale('log')
    plt.xlim(0, 1)
    plt.grid(which='both', alpha=0.2, linewidth=0.5)
    plt.minorticks_on()
    plt.legend(fontsize=10)

plt.tight_layout()
plt.show()