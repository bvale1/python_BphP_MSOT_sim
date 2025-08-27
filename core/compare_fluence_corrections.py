import numpy as np
import h5py
import os
import json
import argparse
import matplotlib.pyplot as plt
import func.plot_func as pf
import func.utility_func as uf
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.interpolate import interp1d
#from mua_extrusion_model_based_reconstruction import TestMetricCalculator

def plot_line_profiles(line_profile_axis, line_profiles, labels, colors, save_dir, ylabel):
    (fig, ax) = plt.subplots(1, 1, figsize=(6, 3))
    for i in range(1, len(line_profiles)):
        ax.plot(line_profile_axis, line_profiles[i], 
                label=labels[i], color=colors[i], alpha=0.7)
    ax.plot(line_profile_axis, line_profiles[0], 
                label=labels[0], color=colors[0], linestyle='dashed')
    ax.set_xlabel('distance (mm)')
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlim(np.min(line_profile_axis), np.max(line_profile_axis))
    ax.legend(bbox_to_anchor=(1.01, 0.5), loc="center left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(save_dir)

# patato filter
def make_filter(n_samples : int, 
                fs : float, # sample rate
                irf : np.ndarray,
                hilbert : bool,
                lp_filter : float,
                hp_filter : float,
                rise : float=0.2,
                n_filter : int=512,
                window=None) -> np.ndarray:
    # filter function from patato
    # https://github.com/BohndiekLab/patato
    
    # at the moment, it looks like it is shifting the data a bit??
    # Impulse Response Correction
    output = np.ones((n_samples,), dtype=np.cdouble)
    if irf is not None:
        irf_shifted = np.zeros_like(irf)
        irf_shifted[:irf.shape[0] // 2] = irf[irf.shape[0] // 2:]
        irf_shifted[-irf.shape[0] // 2:] = irf[:irf.shape[0] // 2]
        output *= np.conj(fft(irf_shifted)) / np.abs(fft(irf_shifted)) ** 2
        from scipy.signal.windows import hann
        output *= fftshift(hann(n_samples))

    # Hilbert Transform
    frequencies = fftfreq(n_samples)
    if hilbert:
        output *= (1 + np.sign(frequencies)) / 2

    frequencies = np.abs(fftfreq(n_filter, 1 / fs))
    fir_filter = np.ones_like(frequencies, dtype=np.cdouble)
    if hp_filter is not None:
        fir_filter[frequencies < hp_filter * (1 - rise)] = 0
        in_rise = np.logical_and(frequencies > hp_filter * (1 - rise), frequencies < hp_filter)
        fir_filter[in_rise] = (frequencies[in_rise] - hp_filter * (1 - rise)) / (hp_filter * rise)
    if lp_filter is not None:
        fir_filter[frequencies > lp_filter * (1 + rise)] = 0
        in_rise = np.logical_and(frequencies < lp_filter * (1 + rise), frequencies > lp_filter)
        fir_filter[in_rise] = 1 - (frequencies[in_rise] - lp_filter) / (lp_filter * rise)

    time_series = ifft(fir_filter)

    if window == "hann":
        from scipy.signal.windows import hann
        time_series *= fftshift(hann(n_filter))

    filter_time = np.zeros_like(output)
    filter_time[:n_filter // 2] = time_series[:n_filter // 2]
    filter_time[-n_filter // 2:] = time_series[-n_filter // 2:]
    fir_filter = fft(filter_time)
    output *= fir_filter
    return output


parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, default='/home/billy/Projects/Scripts/20250826_kwave_200_750_TV0p0/20250826_kwave_200_750_TV0p0/')
args = parser.parse_args()

results_path = os.path.join(args.results_dir, 'results.h5')
save_dir = args.results_dir


print(f'saving plots to: {save_dir}')
'''
data_3d, cfg_3d = uf.load_sim(sim_path_3d, args='all', verbose=False)
data_3d = data_3d[image_name]
data_3d['sensor_data'] = data_3d['sensor_data'].astype(np.float32)

data_extrusion, cfg_extrusion = uf.load_sim(sim_path_extrusion, args='all', verbose=False)
data_extrusion = data_extrusion[image_name]
data_extrusion['sensor_data'] = data_extrusion['sensor_data'].astype(np.float32)

nyquist_freq = 1e-6 * cfg_3d['c_0'] / (cfg_3d['dx'] * 2) # MHz
# define the time and frequency arrays
Nt = cfg_3d['Nt']
dt = cfg_3d['dt'] # seconds
t_array = np.arange(Nt) * dt * 1e6 # convert to microseconds
k = np.arange(1+Nt//2)
f_array = k / (Nt * dt) * 1e-6 # convert to MHz

signals_fft_amp = np.abs(np.fft.fft(data_3d['sensor_data'], axis=-1)[:, :Nt//2+1])
signals_fft_amp[:, 0] = 0.0 # remove DC component
mean_amplitude_spectrum = np.mean(signals_fft_amp, axis=0)

fig, ax = plt.subplots(2, 1, figsize=(5, 5))
for sensor in range(data_3d['sensor_data'].shape[0]):
    ax[0].plot(t_array, data_3d['sensor_data'][sensor, :], alpha=0.2)
    ax[1].plot(f_array, signals_fft_amp[sensor, :], alpha=0.2)
ax[0].plot(t_array, np.mean(data_3d['sensor_data'], axis=0), color='black', lw=2, label='Mean')
ax[0].set_xlabel(r't ($\mu$s)')
ax[0].set_ylabel('Amplitude (Pa)')
ax[0].set_xlim(0, np.max(t_array))
ax[0].grid(True)
ax[0].set_axisbelow(True)
ax[1].plot(f_array, mean_amplitude_spectrum, color='black', lw=2, label='Mean')
ax[1].set_xlabel(r'Frequency (MHz)')
ax[1].set_ylabel('Amplitude (Pa)')
ax[1].vlines(
    nyquist_freq,
    ymin=0,
    ymax=np.max(mean_amplitude_spectrum),
    color='red',
    linestyle='--', 
    label='Nyquist frequency'
)
ax[1].set_xlim(0, 7.5)
ax[1].grid(True)
ax[1].set_axisbelow(True)
ax[1].legend()
fig.tight_layout()
fig.savefig('digimouse_signals.png')

# compute true signal RMS from 18.75 to 37.5 microseconds (samples 750 to 1450)
#RMS = np.sqrt(np.mean(data_3d['sensor_data'][:,750:1450]**2))
# compute true signal RMS from all samples
RMS = np.sqrt(np.mean(data_3d['sensor_data']**2))
assumed_SNR_dB = 20 # dB
noise_std = RMS * (10**(-assumed_SNR_dB / 20)) # RMS = std if mean = 0
print(f'RMS_true: {RMS} Pa, SNR: {assumed_SNR_dB} dB, noise_std: {noise_std} Pa')

irf = np.load('/home/wv00017/python_BphP_MSOT_sim/invision_irf.npy')
irf_fft = np.abs(np.fft.fft(irf))
filter_forward = make_filter(
    n_samples=Nt, fs=1/dt, irf=irf,
    hilbert=True, lp_filter=6.5e6, hp_filter=50e3, rise=0.2,
    n_filter=512, window='hann'
)
irf_fft_backward = interp1d(np.arange(Nt)/(Nt*dt), irf_fft, kind='linear', fill_value=1.0)(np.arange(1500)/(1500*30e-9))
irf_backward = np.fft.ifft(irf_fft_backward)
filter_backward = make_filter(
    n_samples=1500, fs=1/30e-9, irf=irf_backward,
    hilbert=True, lp_filter=6.5e6, hp_filter=50e3, rise=0.2,
    n_filter=512, window='hann'
)
t_array_backward = np.arange(1500) * 30e-9 * 1e6 # convert to microseconds
k_array_backward = np.arange(1+1500//2)
f_array_backward = k_array_backward / (1500 * 30e-9) * 1e-6 # convert to MHz

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(f_array, irf_fft[:Nt//2 + 1], label='forward IRF')
ax.plot(f_array, np.abs(filter_forward[:Nt//2 + 1]), label='forward filter')
ax.plot(f_array_backward, np.abs(irf_fft_backward[:1500//2 + 1]), label='backward IRF', color='black', linestyle='-.')
ax.plot(f_array_backward, np.abs(filter_backward[:1500//2 + 1]), label='backward filter', linestyle='-.')

ax.vlines(
    nyquist_freq,
    ymin=0,
    ymax=2.0,
    color='red',
    linestyle='--', 
    label='Nyquist frequency'
)
ax.set_xlabel(r'Frequency (MHz)')
ax.set_ylabel('Amplitude (a.u.)')
ax.set_xlim(0, 10)
ax.set_ylim(0, 2.0)
ax.grid(True)
ax.set_axisbelow(True)
ax.legend()
fig.tight_layout()
fig.savefig('digimouse_filters.png')
'''
with h5py.File(results_path, 'r') as f:
    gt = {}
    for key in list(f['ground_truth'].keys()):
        gt[key] = f['ground_truth'][key][()]
    mu_a = f['results']['mu_a'][()]
    Phi = f['results']['Phi'][()]
    p0_tr = f['results']['H_recon'][()]
    grad_TV = f['results']['grad_TV'][()]
    grad_MSE = f['results']['grad_MSE'][()]
    
# convert m^-1 to cm^-1
mu_a = mu_a * 1e-2 # [m^-1] -> [cm^-1]
gt['mu_a_true'] = gt['mu_a_true'] * 1e-2 # [m^-1] -> [cm^-1]
gt['mu_s_true'] = gt['mu_s_true'] * 1e-2 # [m^-1] -> [cm^-1]
    
print(f'GT mu_a shape: {gt["mu_a_true"].shape}')
print(f'GT mu_s shape: {gt["mu_s_true"].shape}')
print(f'GT Phi shape: {gt["Phi_true"].shape}')
print(f'GT p0_tr shape: {gt["H_recon_true"].shape}')
print(f'mu_a shape: {mu_a.shape}')
print(f'Phi shape: {Phi.shape}')
print(f'p0_tr shape: {p0_tr.shape}')
print(f'grad_TV shape: {grad_TV.shape}')
print(f'grad_MSE shape: {grad_MSE.shape}')
    
with open (os.path.join(save_dir, 'cfg.json'), 'r') as f:
    cfg = json.load(f)    

mu_a_true = gt['mu_a_true']

mu_a_line_profiles = [np.diag(x) for x in mu_a]
recon_line_profiles = [np.diag(x) for x in p0_tr]
phi_line_profiles = [np.diag(x) for x in Phi]
p0_line_profiles = [np.diag(x*y) for x, y in zip(mu_a, Phi)]
recon_err_line_profiles = [np.diag(gt["H_recon_true"] - x) for x in (p0_tr)]
recon_err_over_phi_line_profiles = [np.diag(gt["H_recon_true"] - x) / (np.diag(y) + 1e-8) for x, y in zip(p0_tr, Phi)]
grad_TV_line_profiles = [np.diag(x) for x in grad_TV]
grad_MSE_line_profiles = [np.diag(x) for x in grad_MSE]

mu_a_plots = uf.square_centre_crop(np.asarray(mu_a), cfg['crop_size'])
labels=['ground truth', 'initial guess n=0']
for n in range(1, 10+1):
    labels.append(f'n={n}')
(fig, ax, frames) = pf.heatmap(
    mu_a_plots, 
    labels=labels,
    title=r'$\mu_{\mathrm{a}}$',
    dx=cfg['dx'],
    sharescale=True,
    cmap='viridis',
    rowmax=4,
    cbar_label=r'cm$^{-1}$'
)
fig.savefig(os.path.join(save_dir, 'mu_a.png'))
residuals = mu_a_plots[2:] - uf.square_centre_crop(mu_a_true.copy(), cfg['crop_size'])
labels = []
for n in range(1, 10+1):
    labels.append(f'n={n}')
(fig, ax, frames) = pf.heatmap(
    residuals, 
    labels=labels,
    title=r'$\mu_{\mathrm{a}}$ residuals',
    dx=cfg['dx'],
    sharescale=True,
    cmap='plasma',
    rowmax=4,
    vmin=np.maximum(-np.max(mu_a_true), np.min(residuals)),
    vmax=np.minimum(np.max(mu_a_true), np.max(residuals))
)
fig.savefig(os.path.join(save_dir, 'mu_a_residuals.png'))
labels=['ground truth']
for n in range(1, 10+1):
    labels.append(f'n={n}')
    
labels = ['ground truth', 'initial guess n=0']
for n in range(1, 10+1):
    labels.append(f'n={n}')
linestyle = ['solid', 'dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5)),
                (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1)),
                (0, (3, 5, 1, 5, 1, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1, 1, 1)),
                (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (10, 3))]
#colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown',
#          'pink', 'gray', 'cyan', 'magenta', 'yellow', 'lime', 'teal']
# create a palette of colors equally spaced between (26, 133, 255) and (212, 17, 89)
colors = ['black'] # ground truth is black
for x in np.linspace(0, 1, len(mu_a_line_profiles)-1):
    colors.append((
        (26/256)*x + (212/256)*(1-x), # r [0.0 to 1.0]
        (133/256)*x + (17/256)*(1-x), # g [0.0 to 1.0]
        (255/256)*x + (89/256)*(1-x)  # b [0.0 to 1.0]
    ))
line_profile_axis = np.arange(
    -cfg['dx']*cfg['crop_size']/2,
    cfg['dx']*cfg['crop_size']/2, 
    cfg['dx']
) * 1e3 # convert to mm

# line profiles for all iterations
plot_line_profiles(line_profile_axis, mu_a_line_profiles, labels, colors,
                   os.path.join(save_dir, 'mu_a_line_profile.png'),
                   ylabel=r'$\mu_{\mathrm{a}}$ (cm$^{-1}$)')

plot_line_profiles(line_profile_axis, recon_line_profiles, labels, colors,
                   os.path.join(save_dir, 'reconstructions_line_profile.png'),
                   ylabel=r'$\hat{p}_{0}$ (Pa)')

plot_line_profiles(line_profile_axis, phi_line_profiles, labels, colors,
                   os.path.join(save_dir, 'phi_line_profile.png'),
                   ylabel=r'$\Phi$ (J m$^{-2}$)')

plot_line_profiles(line_profile_axis, recon_err_line_profiles, labels, colors,
                    os.path.join(save_dir, 'recon_err_line_profile.png'),
                    ylabel=r'$\hat{p}_{0} - p_{0}$ (Pa)')

plot_line_profiles(line_profile_axis, recon_err_over_phi_line_profiles, labels, colors,
                    os.path.join(save_dir, 'recon_err_over_phi_line_profile.png'),
                    ylabel=r'$\frac{\hat{p}_{0} - p_{0}}{\Phi}$ (m$^{-1}$)')

plot_line_profiles(line_profile_axis, p0_line_profiles, labels, colors,
                   os.path.join(save_dir, 'p0_line_profile.png'),
                   ylabel=r'$p_{0}$ (Pa)')

plot_line_profiles(line_profile_axis, grad_TV_line_profiles, labels, colors,
                   os.path.join(save_dir, 'grad_TV_line_profile.png'),
                   ylabel=r'$\nabla_{\mu_{\mathrm{a}}}$ TV$(\mu_{\mathrm{a}})$ (m$^{-1}$)')

plot_line_profiles(line_profile_axis, grad_MSE_line_profiles, labels, colors,
                   os.path.join(save_dir, 'grad_MSE_line_profile.png'),
                   ylabel=r'$\nabla_{\mu_{\mathrm{a}}}$ MSE$(\mu_{\mathrm{a}})$ (m$^{-1}$)')

# line profiles for final iteration
(fig, ax) = plt.subplots(1, 2, figsize=(6, 3))
ax[0].plot(line_profile_axis, mu_a_line_profiles[0], 
        label='ground truth', color='black')
ax[0].plot(line_profile_axis, mu_a_line_profiles[-1],
        label='10th iteration', color='red', linestyle='dashed')
ax[0].set_xlabel('distance (mm)')
ax[0].set_ylabel(r'$\mu_{\mathrm{a}}$ (cm$^{-1}$)')
ax[0].grid(True)
ax[0].set_axisbelow(True)
ax[0].set_xlim(np.min(line_profile_axis), np.max(line_profile_axis))
ax[0].legend(bbox_to_anchor=(0, 1.01, 1.5, 0.2), loc="lower left",
               mode="expand", ncol=3)

ax[1].plot(line_profile_axis, recon_line_profiles[0], color='black')
ax[1].plot(line_profile_axis, recon_line_profiles[-1], color='red',
             linestyle='dashed')
ax[1].set_xlabel('distance (mm)')
ax[1].set_ylabel(r'$\hat{p}_{0}$ (Pa)')
ax[1].grid(True)
ax[1].set_axisbelow(True)
ax[1].set_xlim(np.min(line_profile_axis), np.max(line_profile_axis))
fig.tight_layout()
fig.savefig(os.path.join(save_dir, 'final_reconstructions_line_profile.png'))


(fig, ax, frames) = pf.heatmap(
    np.asarray(p0_tr), 
    labels=labels,
    title=r'$\hat{p}_{0}$',
    dx=cfg['dx'],
    sharescale=True,
    cmap='viridis',
    rowmax=4,
    cbar_label='Pa'
)
fig.savefig(os.path.join(save_dir, 'p0_recon.png'))

(fig, ax, frames) = pf.heatmap(
    np.asarray(Phi), 
    labels=labels,
    title=r'$\Phi$',
    dx=cfg['dx'],
    sharescale=True,
    cmap='viridis',
    rowmax=4,
    cbar_label=r'J m$^{-2}$'
)
fig.savefig(os.path.join(save_dir, 'Phi.png'))

(fig, ax, frames) = pf.heatmap(
    np.asarray(grad_TV), 
    labels=labels,
    title=r'$\nabla_{\mu_{\mathrm{a}}} TV(\mu_{\mathrm{a}})$',
    dx=cfg['dx'],
    sharescale=True,
    cmap='viridis',
    rowmax=4,
    cbar_label=r'm$^{-1}$'
)
fig.savefig(os.path.join(save_dir, 'grad_TV.png'))

(fig, ax, frames) = pf.heatmap(
    np.asarray(grad_MSE),
    labels=labels,
    title=r'$\nabla_{\mu_{\mathrm{a}}} MSE(\mu_{\mathrm{a}})$',
    dx=cfg['dx'],
    sharescale=True,
    cmap='viridis',
    rowmax=4,
    cbar_label=r'm$^{-1}$'
)
fig.savefig(os.path.join(save_dir, 'grad_MSE.png'))

labels = [r'$\mu_{a}$ (cm$^{-1}$)', r'$\mu_{s}$ (cm$^{-1}$)',
            r'$\Phi$ (J m$^{-2}$)', r'$p_{0}$ initial pressure (Pa)',
            r'$\hat{p}_{0}$ reconstructed (Pa)']
images = [gt['mu_a_true'], 
            gt['mu_s_true'], 
            gt['Phi_true'], 
            gt['mu_a_true']*gt['Phi_true'],
            gt['H_recon_true']]
(fig, ax, frames) = pf.heatmap(
    np.asarray(images), dx=cfg['dx'], rowmax=5, labels=labels
)
fig.savefig(os.path.join(save_dir, 'images.png'))
