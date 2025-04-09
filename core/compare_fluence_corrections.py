import numpy as np
import matplotlib.pyplot as plt
import func.plot_func as pf
import func.utility_func as uf
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.interpolate import interp1d

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


sim_path_3d = '/mnt/f/cluster_MSOT_simulations/digimouse_fluence_correction/3d_digimouse/20241018_digimouse_phantom.c193723.p2'
sim_path_extrusion = '/mnt/f/cluster_MSOT_simulations/digimouse_fluence_correction/2d_extrusion_digimouse/20250206_digimouse_extrusion_phantom.Naisurrey26.j742887'
image_name = '500_750'

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
#assumed_SNR_dB = 20 # dB
#noise_std = np.exp(-assumed_SNR_dB / 20) * RMS # RMS = std if mean = 0
#print(f'RMS_true: {RMS} Pa, SNR: {assumed_SNR_dB} dB, noise_std: {noise_std} Pa')

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