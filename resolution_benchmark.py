import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from skimage.transform import resize
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

def me_synth(B, gamma, chi, vlos, dop, a, eta0, S0, S1, ll, lambda0=6173.5):
    c = 3e5
    dop = np.clip(dop, 1e-4, 1.0)
    B = np.clip(np.abs(B), 1e-3, 2000.0)

    ll_shifted = ll * (1.0 - vlos / c)

    g_eff = 1.5
    dlambda = 4.6686e-13 * lambda0**2 * g_eff * B

    l_plus = ll_shifted + dlambda / 2.0
    l_minus = ll_shifted - dlambda / 2.0

    def absorption_profile(lam):
        u = (lam - lambda0) / dop
        voigt = np.exp(-u**2) / np.sqrt(np.pi) + a / (np.pi * (u**2 + a**2))
        return eta0 * voigt

    eta_plus = absorption_profile(l_plus)
    eta_minus = absorption_profile(l_minus)

    I = S0 + S1 * (1.0 - 0.5 * (np.exp(-eta_plus) + np.exp(-eta_minus)))
    V = S1 * 0.5 * (np.exp(-eta_minus) - np.exp(-eta_plus))

    cosg = np.cos(gamma)
    sing = np.sin(gamma)
    c2chi = np.cos(2 * chi)
    s2chi = np.sin(2 * chi)

    Q = S1 * 0.5 * sing**2 * c2chi * (np.exp(-eta_plus) - np.exp(-eta_minus))
    U = S1 * 0.5 * sing**2 * s2chi * (np.exp(-eta_plus) - np.exp(-eta_minus))

    I = np.clip(I, 0.01, 1.0)
    return np.stack([I, Q, U, V]).T.astype(np.float64)


def me_invert(stokes, ll, fixed_params, initial_guess):
    vlos, dop, a, eta0, S0, S1 = fixed_params

    def residuals(params):
        B, gamma, chi = params
        syn = me_synth(B, gamma, chi, vlos, dop, a, eta0, S0, S1, ll)
        return (syn.ravel() - stokes.ravel()).astype(np.float64)

    bounds = ([1e-3, 0.0, 0.0], [2000.0, np.pi, 2*np.pi])
    lb, ub = bounds

    try:
        res = leastsq(residuals, initial_guess, maxfev=2000, ftol=1e-7, xtol=1e-7)
        params = np.clip(res[0], lb, ub)
    except:
        params = initial_guess

    return params

def invert_grid(stokes):
    ny_, nx_ = stokes.shape[:2]
    Br = np.zeros((ny_, nx_))
    for j in range(ny_):
        for i in range(nx_):
            p = me_invert(stokes[j,i], ll,
                        (vlos_fixed, dop_fixed, a_fixed, eta0_fixed, S0_fixed, S1_fixed),
                        inv_guess)
            Br[j,i] = p[0] * np.cos(p[1])
    return Br

def threshold_stokes(s, thresh):
    s_copy = s.copy()
    mask = np.abs(s_copy) < thresh
    s_copy[mask] = 0.0
    return s_copy
 
if __name__ == '__main__':
    ll = np.linspace(6173.0, 6174.0, 100)

    vlos_fixed = 0.1
    dop_fixed = 0.02
    a_fixed = 0.2
    eta0_fixed = 30.0
    S0_fixed = 0.2
    S1_fixed = 0.8

    inv_guess = [500.0, np.pi/2, 0.0]
    noise_amp = 5e-3
    
    save_dir = 'E:/Research/Work/Magnetogram_SuperResol_by_ML/benchmark/'
    save_or_not = 1

    stokes_threshold = 2e-2 

    lx , ly = 2880, 2880 # [km]
    # lx , ly = 1440, 1440 # [km]
    dr0 = 20
    ny, nx = int(lx / dr0), int(ly / dr0)
    X, Y = np.meshgrid(np.linspace(0, lx, nx), np.linspace(0, ly, ny))
    
    # resolutions
    arc2km = 700000 / 960
    dr1 = 17.6 # [km], MURaM simulation, 17.6 km
    dr2 = 0.03 * arc2km # [km], DKIST/ViSP, 22 km
    dr3 = 0.32 * arc2km # [km], Hinode/SP, 233 km
    dr4 = 0.50 * arc2km # [km], SDO/HMI, 365 km
    
    dr1 = 20 # [km], MURaM simulation, 17.6 km
    dr2 = 25 # [km], DKIST/ViSP, 22 km
    dr3 = 240 # [km], Hinode/SP, 233 km
    dr4 = 360 # [km], SDO/HMI, 365 km
    
    print(f'Resolution: dr0={dr0}, dr1={dr1}, dr2={dr2}, dr3={dr3}, dr4={dr4}')
    
    titles = [
        f'Original ({dr0:.0f}km)',
        f'MURaM Simul. (~{dr1:.0f}km)',
        f'DKIST/ViSP Obs. (~{dr2:.0f}km)',
        f'Hinode/SP Obs. (~{dr3:.0f}km)',
        f'SDO/HMI Obs. (~{dr4:.0f}km)'
    ]

    n_sources = 50
    Br_mean = -500
    np.random.seed(42)
    
    sigma_lst = [30, 40, 50, 80, 100, 150]
    for sigma in sigma_lst:
        print(f'========== Begin processing sigma={sigma} ==========')
        sigma_x = sigma
        sigma_y = sigma
        
        B0 = np.zeros_like(X)
        for _ in range(n_sources):
            x0 = np.random.uniform(0.1, 0.9) * lx
            y0 = np.random.uniform(0.1, 0.9) * ly
            
            amp = np.random.normal(loc=Br_mean, scale=100)
            
            gauss = np.exp(-((X - x0)**2 / (2*sigma_x**2) + (Y - y0)**2 / (2*sigma_y**2)))
            B0 += amp * gauss

        gamma0 = np.where(B0 >= 0, 0.0, np.pi)
        B0 = np.abs(B0)
        chi0 = np.zeros_like(B0)
        Br0 = B0 * np.cos(gamma0)

        stokes0 = np.zeros((ny, nx, len(ll), 4), dtype=np.float64)
        for j in range(ny):
            for i in range(nx):
                stokes0[j,i] = me_synth(B0[j,i], gamma0[j,i], chi0[j,i],
                                        vlos_fixed, dop_fixed, a_fixed, eta0_fixed, S0_fixed, S1_fixed, ll)
        
        stokes0_noise = stokes0 + np.random.normal(0, noise_amp, stokes0.shape)
        stokes0_noise = threshold_stokes(stokes0_noise, stokes_threshold)
        
        stokes1 = resize(stokes0_noise, (int(ly//dr1), int(lx//dr1), len(ll), 4), preserve_range=True)
        stokes2 = resize(stokes0_noise, (int(ly//dr2), int(lx//dr2), len(ll), 4), preserve_range=True)
        stokes3 = resize(stokes0_noise, (int(ly//dr3), int(lx//dr3), len(ll), 4), preserve_range=True)
        stokes4 = resize(stokes0_noise, (int(ly//dr4), int(lx//dr4), len(ll), 4), preserve_range=True)

        stokes2 = threshold_stokes(stokes2, stokes_threshold)
        stokes3 = threshold_stokes(stokes3, stokes_threshold)
        stokes4 = threshold_stokes(stokes4, stokes_threshold)

        Br1 = invert_grid(stokes1)
        Br2 = invert_grid(stokes2)
        Br3 = invert_grid(stokes3)
        Br4 = invert_grid(stokes4)

        
        phi0 = np.sum(np.abs(Br0) * dr0 ** 2) * 1e-5 * 1e6
        phi1 = np.sum(np.abs(Br1) * dr1 ** 2) * 1e-5 * 1e6
        phi2 = np.sum(np.abs(Br2) * dr2 ** 2) * 1e-5 * 1e6
        phi3 = np.sum(np.abs(Br3) * dr3 ** 2) * 1e-5 * 1e6
        phi4 = np.sum(np.abs(Br4) * dr4 ** 2) * 1e-5 * 1e6
        
        pixle_size = [dr0, dr1, dr2,  dr3, dr4]
        phis = [phi0, phi1, phi2, phi3, phi4]
        ratios = np.array(phis) / phi0
        
        plt.rcParams['font.size'] = 12
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()

        Br_lst = [Br0, Br1, Br2, Br3, Br4]

        vmin, vmax = -800, 800
        cmap = 'bwr'

        for i in range(5):
            im = axes[i].imshow(Br_lst[i], cmap=cmap, vmin=vmin, vmax=vmax)
            axes[i].set_title(titles[i])
            axes[i].text(0.1, 0.01, f'$\Phi$ = {phis[i]:.3e} Wb', 
                    transform=axes[i].transAxes, 
                    ha='left', fontsize=12, color='black')
            if i == 2:
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        axes[5].axhline(y=phi0, color='gray', linestyle='--', linewidth=2, label='Original Flux')
        axes[5].plot(pixle_size[1:], phis[1:], 'o-', linewidth=3, markersize=8, color='crimson', label='Observed Flux')
        axes[5].set_xlabel('Pixel Size')
        axes[5].set_ylabel('Total Signed Flux')
        axes[5].grid(True, alpha=0.3)
        axes[5].legend()

        for i, (s, p, r) in enumerate(zip(pixle_size[1:], phis[1:], ratios[1:])):
            axes[5].annotate(f'{r:.2f}', (s, p), textcoords='offset points', xytext=(20,0), ha='center', fontsize=12, color='blue')

        # plt.tight_layout()
        plt.suptitle(f'Br observation based on {n_sources} magnetic elements of $\sigma$={sigma}km', fontsize=15)
        
        if save_or_not == 1:
            save_fn = f'benchmark.br.sigma.{sigma}.png'
            plt.savefig(save_dir + save_fn)
            plt.close()
        
        print(f'Saved Br distribution of sigma={sigma}')
        
        ## Figure 2: Stokes profiles
        plt.rcParams['font.size'] = 12
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        
        plot_x = [720, 1440]
        plot_y = [720, 1440]
        V0 = stokes0[int(plot_x[0]/dr0):int(plot_x[1]/dr0), int(plot_y[0]/dr0):int(plot_y[1]/dr0), :, 3]
        V1 = stokes1[int(plot_x[0]/dr1):int(plot_x[1]/dr1), int(plot_y[0]/dr1):int(plot_y[1]/dr1), :, 3]
        V2 = stokes2[int(plot_x[0]/dr2):int(plot_x[1]/dr2), int(plot_y[0]/dr2):int(plot_y[1]/dr2), :, 3]
        V3 = stokes3[int(plot_x[0]/dr3):int(plot_x[1]/dr3), int(plot_y[0]/dr3):int(plot_y[1]/dr3), :, 3]
        V4 = stokes4[int(plot_x[0]/dr4):int(plot_x[1]/dr4), int(plot_y[0]/dr4):int(plot_y[1]/dr4), :, 3]
        
        V_list = [V0, V1, V2, V3, V4]

        ymin, ymax = -0.25, 0.25

        for i in range(5):
            data_sub = V_list[i]
            total_points = data_sub.shape[0] * data_sub.shape[1]
            count = 0
            for ix in range(data_sub.shape[0]):
                for iy in range(data_sub.shape[1]):
                    color = cm.autumn(count / total_points)
                    axes[i].plot(ll, data_sub[ix, iy, :], color=color, linewidth=0.8)
                    count += 1
            axes[i].set_title(titles[i])
            axes[i].set_ylim([ymin, ymax])
        
        
        im = axes[5].imshow(Br_lst[0], cmap=cmap, vmin=vmin, vmax=vmax)
        rect = Rectangle((plot_x[0]/dr0, plot_y[0]/dr0), (plot_x[1]-plot_x[0])/dr0, (plot_y[1]-plot_y[0])/dr0, 
                        linewidth=2, edgecolor='k', facecolor='none')
        axes[5].add_patch(rect)
        axes[5].set_title(titles[0])
        plt.colorbar(im, ax=axes[5], fraction=0.046, pad=0.04)
                
        # plt.tight_layout()
        plt.suptitle(f'Stokes-V profiles based on {n_sources} magnetic elements of $\sigma$={sigma}km', fontsize=15)
        
        if save_or_not == 1:
            save_fn = f'benchmark.stokes.sigma.{sigma}.png'
            plt.savefig(save_dir + save_fn)
            plt.close()
        else: 
            plt.show()
        
        print(f'Saved Stokes profiles of sigma={sigma}')
    
    
    db