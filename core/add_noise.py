import numpy as np
import json, h5py, os, timeit, logging, argparse
import acoustic_inverse_simulation
import func.utility_func as uf
from scipy.ndimage import convolve1d
from scipy.fft import fft, ifft, fftshift, fftfreq
from scipy.linalg import cholesky
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import Sequence
from tqdm.auto import tqdm


# patato backprojection
def reconstruct(time_series: np.ndarray,
                fs: float,
                geometry: np.ndarray, n_pixels: Sequence[int],
                field_of_view: Sequence[float],
                speed_of_sound: float,
                **kwargs) -> np.ndarray:
    """

    Parameters
    ----------
    time_series: array_like
        Photoacoustic time series data in a numpy array. Shape: (..., n_detectors, n_time_samples)
    fs: float
        Time series sampling frequency (Hz).
    geometry: array_like
        The detector geometry. Shape: (n_detectors, 3)
    n_pixels: tuple of int
        Tuple of length 3, (nx, ny, nz)
    field_of_view: tuple of float
        Tuple of length 3, (lx, ly, lz) - the size of the reconstruction volume.
    speed_of_sound: float
        Speed of sound (m/s).
    kwargs
        Extra parameters (optional), useful for advanced algorithms (e.g. multi speed of sound etc.).

    Returns
    -------
    array_like
        The reconstructed image.

    """
    print("Running batch of delay and sum reconstruction code.")

    # Get useful parameters:
    dl = speed_of_sound / fs
    print(f'time_series {time_series.shape}')
    # Reshape frames so that we can loop through to reconstruct
    original_shape = time_series.shape[:-2]
    frames = int(np.product(original_shape))
    print(f"Original shape: {original_shape}, frames: {frames}")
    signal = time_series.reshape((frames,) + time_series.shape[-2:])
    print(f"Reshaped signal: {signal.shape}")

    xs, ys, zs = [
        np.linspace(-field_of_view[i] / 2, field_of_view[i] / 2, n_pixels[i]) if n_pixels[i] != 1 else np.array(
            [0.]) for i in range(3)]
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing='ij')

    # Note that the reconstructions are stored in memory in the order z, y, x (i.e. the x axis is the fastest
    # changing in memory)
    output = np.zeros((frames,) + tuple(n_pixels)[::-1])

    for n_frame in tqdm(range(frames), desc="Looping through frames", position=0):
        for n_detector in tqdm(range(signal.shape[-2]), desc="Looping through detectors", position=1, leave=False):
            detx, dety, detz = geometry[n_detector]
            d = (np.sqrt((detx - X) ** 2 + (dety - Y) ** 2 + (detz - Z) ** 2) / dl).astype(np.int32)
            output[n_frame] += signal[n_frame, n_detector, d]
    return output.reshape(original_shape + tuple(n_pixels)[::-1])

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


def add_noise(data : np.array,
              cfg : dict,
              rng : np.random.Generator, 
              std : float = 2.0,
              frequency_distribution : np.array = None,
              cov0 : np.array = None):
    '''
    Add stochastic noise to the data with a given amplitude spectrum.
    Guassian noise is generated and convolved with the amplitude spectrum.
    A covariance matrix can be provided to correlate the noise between certain
    channels.
    '''
    cfg['noise_std'] = std
    
    noise = rng.normal(0, std, size=data.shape)
    
    #if cov0 is not None:
    #    C = cholesky(cov0)
    #    print(f'C {C}')
    #    print(f'cov {cov0}')
    #    for cycle in range(data.shape[0]):
    #        for wavelength in range(data.shape[1]):
    #            for pulse in range(data.shape[2]):
    #                noise[cycle, wavelength, pulse] = np.dot(
    #                    C, noise[cycle, wavelength, pulse], 
    #                )
    
    # check the noise correlation matrix matches cov0
    #cov = np.zeros((noise.shape[-2], noise.shape[-2]))
    #for cycle in range(data.shape[0]):
    #    for wavelength in range(data.shape[1]):
    #        for pulse in range(data.shape[2]):
    #            for i in range(noise.shape[-2]):
    #                for j in range(noise.shape[-2]):
    #                    x = noise[cycle,wavelength,pulse,i] - np.mean(noise[cycle,wavelength,pulse,i])
    #                    y = noise[cycle,wavelength,pulse,j] - np.mean(noise[cycle,wavelength,pulse,j])
    #                    cov[i, j] += np.sum(x * y) / np.sqrt(np.sum(x**2)*np.sum(y**2))  #cfg['Nt']
    #cov /= (data.shape[0] * data.shape[1] * data.shape[2])
    
    #if cov0 is not None:
    #    fig, ax = plt.subplots(1, 2, figsize=(8, 16))
    #    im1 = ax[0].imshow(cov0, cmap='viridis', origin='lower')
    #    ax[0].set_title('Target covariance matrix')
    #    ax[0].set_xlabel('Channel index')
    #    ax[0].set_ylabel('Channel index')
    #    plt.colorbar(im1, ax=ax[0])
    #    im2 = ax[1].imshow(cov, cmap='viridis', origin='lower')
    #    ax[1].set_title('Noise covariance matrix')
    #    ax[1].set_xlabel('Channel index')
    #    ax[1].set_ylabel('Channel index')
    #    plt.colorbar(im2, ax=ax[1])
    #    fig.tight_layout()
    #    plt.savefig('noise_covariance_matrix.png')
        
    if frequency_distribution is not None:
        noise = np.fft.ifft(np.fft.fft(noise, axis=-1) * frequency_distribution, axis=-1)
    
    noisy_data = data + noise
    
    return (noisy_data, cfg)


if __name__ == '__main__':
    
    '''
    ==================================Workflow==================================

    1. Load simulated signals and the corresponding configuration file
    
    2. add appropriate white noise to the signals
    
    3. apply convolution with the impulse response of the sensor to the signals    
    
    4. (optional) save the new signals
    
    5. run image reconstruction algorithm
    
    6. (optional) save the reconstructed images

    ============================================================================
    '''
    
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_dir', type=str, default='/mnt/f/cluster_MSOT_simulations/digimouse_fluence_correction/20240702_digimouse_phantom.c174176.p1',
        help='path to simulation data directory'        
    )
    parser.add_argument(
        '--irf_path', type=str, default='invision_irf.npy',
    )
    parser.add_argument(
        '--weights_dir', type=str, default=None,
        help='path to the directory containing the weights for the inverse model'
    )
    parser.add_argument(
        '--save_images', type=str, default='noisy_p0_tr', 
        help='key to save noisy reconstructed images to the data.h5 file \
            the signal to noise ratio in practice is approximately 11'
    )
    parser.add_argument(
        '--noise_std', type=float, default=2.0,
        help='standard deviation of the white noise to add to the signals'
    )
    parser.add_argument(
        '--plot_comparison', type=str, default='add_noise_comparison.png',
        help='path to save a comparison of the original and noise \
            added signal and reconstructions'
    )
    parser.add_argument(
        '--bandpass_filter', default=False, action=argparse.BooleanOptionalAction,
        help='apply bandpass filter to sensor data'
    )
    parser.add_argument('-v', type=str, help='verbose level', default='INFO')
    args = parser.parse_args()
    
    # configure logging
    if args.v == 'INFO':
        logging.basicConfig(level=logging.INFO)
    elif args.v == 'DEBUG':
        logging.basicConfig(level=logging.DEBUG)
    else:
        raise ValueError('verbose level must be INFO or DEBUG')
    
    logging.info(f'command line arguments: {args}')
    
    # check if the save directory exists
    if not os.path.exists(args.save_dir):
        raise FileNotFoundError(f'file {args.save_dir} not found')
    logging.debug(f'found directory {args.save_dir}, adding noise...')
    
    # check input arguments
    if args.save_images == 'p0_tr': # TODO: add option to overwrite
        raise ValueError('key p0_tr is reserved for the original images')
        
    # 1. Load simulation configuration file
    if os.path.exists(os.path.join(args.save_dir, 'config.json')):
        with open(os.path.join(args.save_dir, 'config.json'), 'r') as f:
            cfg = json.load(f)
        logging.info(f'loaded configuration file from {args.save_dir}/config.json')
        logging.debug(f'configuration file: {cfg}')
    else:
        raise FileNotFoundError(f'file {args.save_dir}/config.json not found')
    
    # pass weights_dir to config
    if args.weights_dir is not None:
        cfg['weights_dir'] = args.weights_dir
    
    # load the simulated signals [256*2030*4*128/(1024**2) ~ 254 MB] if 128 images are in the dataset
    if os.path.exists(os.path.join(args.save_dir, 'data.h5')):
        with h5py.File(os.path.join(args.save_dir, 'data.h5'), 'r') as f:
            groups = list(f.keys())
            logging.info(f'groups {groups}')
            sensor_data = np.zeros((len(groups), 256, 2030), dtype=np.float32)
            for i in range(len(groups)):
                sensor_data[i] = f[groups[i]]['sensor_data'][()]
        logging.info(f'loaded sensor data from {args.save_dir}/data.h5')
    else:
        raise FileNotFoundError(f'file {args.save_dir}/data.h5 not found')    
    
    # 4. add appropriate noise to the signals
    # same seed as simulation configuration is used to ensure reproducibility
    # instantiate random number generator
    if cfg['seed'] is None:
        cfg['seed'] = np.random.randint(0, 2**32 - 1)
        logging.info(f'no seed provided, random seed: {cfg["seed"]}')
    else:
        logging.info(f'seed provided by config: {cfg["seed"]}')
    rng = np.random.default_rng(cfg['seed'])
    
    #(noisy_sensor_data, cfg) = add_gaussian_noise(deepcopy(sensor_data), cfg, args.noise_std)
    #noise_amplitude_distribution = np.load('mean_noise_amplitude_spectrum.npy')
    #noise_amplitude_distribution = noise_amplitude_distribution / np.max(noise_amplitude_distribution)
    #cov0 = np.load('cov0.npy')
    (noisy_sensor_data, cfg) = add_noise(
        deepcopy(sensor_data), cfg, rng, std=args.noise_std#, cov0=cov0
    )
    logging.info(f'white noise added to sensor data with std: {args.noise_std}')
    
    # 2. apply convolution with the impulse response of the sensor to the signals    
    irf = np.load(args.irf_path)
    noisy_sensor_data = convolve1d(noisy_sensor_data, irf, mode='nearest', axis=-1)
    #sensor_data = np.fft.ifft(np.fft.fft(sensor_data, axis=-1) * np.fft.fft(irf), axis=-1).real.astype(np.float32)
    logging.info('sensor data convolved with impulse response function')
    
    # add the white noise standard deviation to the configuration then save it
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f)
    
    if args.bandpass_filter:
        filter = make_filter(
            n_samples=cfg['Nt'], fs=1/cfg['dt'], irf=irf,
            hilbert=True, lp_filter=6.5e6, hp_filter=50e3, rise=0.2,
            n_filter=512, window='hann'
        )
        logging.info('filter created')
        '''
        # plot the filter
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        k = np.arange(1+cfg['Nt']//2)
        f_array = 1e-6 * k / (cfg['Nt'] * cfg['dt'])
        ax.plot(f_array, np.abs(filter[:1+cfg['Nt']//2]) * 2)
        ax.set_title('Filter amplitude response')
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Amplitude (a.u.)')
        ax.grid(True)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig('filter_amplitude_response.png')    
        '''
        # subtract the mean of each channel
        #noisy_sensor_data -= np.mean(noisy_sensor_data, axis=-1, keepdims=True)
        
        # apply the filter to the noisy sensor data
        noisy_sensor_data = np.fft.ifft(np.fft.fft(noisy_sensor_data, axis=-1) * filter, axis=-1).real.astype(np.float32)
        #noisy_sensor_data = convolve1d(noisy_sensor_data, np.fft.ifft(filter), mode='nearest', axis=-1)
        logging.info('bandpass filter applied to noisy sensor data')
        
    # 6. run image reconstruction algorithm
    start = timeit.default_timer()
    simulation = acoustic_inverse_simulation.kwave_inverse_adapter(
        cfg,
        transducer_model=cfg['inverse_model']
    )
    simulation.configure_simulation()
    geometry = np.array([simulation.source_x, np.zeros(256), simulation.source_z]).T
    logging.info(f'kwave inverse initialised in \
                 {timeit.default_timer() - start} seconds')
    
    
    # iterate over the cycles, wavelengths and pulses
    #cycle = 0; wavelength_index = 1; pulse = 0
    for i in range(sensor_data.shape[0]):
        logging.info(f'time reversal, image: {i+1}/{sensor_data.shape[0]}')
        
        start = timeit.default_timer()
        noisy_tr = simulation.run_time_reversal(
            noisy_sensor_data[i]
        )
        #noisy_tr = reconstruct(
        #    noisy_sensor_data[i],
        #    fs=1/cfg['dt'],
        #    geometry=geometry, 
        #    n_pixels=(256, 1, 256),
        #    field_of_view=cfg['dx']*np.array([256, 0, 256]), 
        #    speed_of_sound=cfg['c_0']
        #)[:, 0, :].T
        noisy_tr[noisy_tr < 0.0] = 0.0
        print(f'noisy_tr {noisy_tr.shape}')
        noisy_tr = uf.square_centre_crop(noisy_tr, cfg['crop_size'])
        logging.info(f'time reversal run in \
                        {timeit.default_timer() - start} seconds')
        
        # 7. (optional) save the reconstructed images
        if args.save_images is not None:
            start = timeit.default_timer()
            with h5py.File(os.path.join(args.save_dir, 'data.h5'), 'r+') as f:
                if args.save_images not in f[groups[i]].keys():
                    f[groups[i]].create_dataset(args.save_images, data=noisy_tr, dtype=np.float32)
                else:
                    f[groups[i]][args.save_images][()] = noisy_tr
            logging.info(f'reconstruction saved in {timeit.default_timer() - start} seconds')


    # for testing purposes save a comparison of the original and noise added 
    # signal and reconstruction for the first cycle, wavelength and pulse only
    if args.plot_comparison:
        
        if not args.save_images:
            raise ValueError('key save_images must be provided to plot comparison')
        
        t_array = np.arange(cfg['Nt']) * cfg['dt'] * 1e6
        # 1 + dx//2 includes the dc component, positive and Nyquist frequencies
        k = np.arange(1+cfg['Nt']//2)
        f_array = k / (cfg['Nt'] * cfg['dt'] ) * 1e-6
        
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Comparison of original and noise added signals and reconstructions')
        
        ax[0, 0].plot(t_array, noisy_sensor_data[0, 0], label='noise added', linestyle='--')
        ax[0, 0].plot(t_array, sensor_data[0, 0], label='original', linestyle=':')        
        ax[0, 0].set_title('Time domain signals')
        ax[0, 0].set_xlabel(r'Time ($\mu$s)')
        ax[0, 0].set_ylabel('Amplitude (Pa)')
        ax[0, 0].grid(True)
        ax[0, 0].set_axisbelow(True)
        ax[0, 0].legend()
        
        f_mag_sensor_data = np.abs(np.fft.fft(sensor_data[0, 0]))
        # normalise the magnitude of the fft and only take below the nyquist frequency
        # the factor of 2 is to account for the negative frequencies
        f_mag_sensor_data = 2 * f_mag_sensor_data[0:int(1+cfg['Nt']/2)] / (cfg['Nt'])
        f_mag_sensor_data[0] /= 2 # DC component does not get doubled
        f_mag_noisy_sensor_data = np.abs(np.fft.fft(noisy_sensor_data[0, 0]))
        # normalise the magnitude of the fft and only take below the nyquist frequency
        f_mag_noisy_sensor_data = 2 * f_mag_noisy_sensor_data[0:int(1+cfg['Nt']/2)] / (cfg['Nt'])
        f_mag_noisy_sensor_data[0] /= 2 # DC component is halved
        spacial_nyquist = cfg['c_0'] / (2 * cfg['dx']) * 1e-6
        
        ax[0, 1].axvline(spacial_nyquist, color='g', linestyle='--', label='spacial nyquist frequency')
        ax[0, 1].plot(f_array, f_mag_noisy_sensor_data, label='noise added', linestyle='--')
        ax[0, 1].plot(f_array, f_mag_sensor_data, label='original', linestyle=':')
        ax[0, 1].set_title('Signals amplitude spectrum')
        ax[0, 1].set_xlabel('Frequency (MHz)')
        ax[0, 1].set_ylabel('Amplitude (Pa)')
        ax[0, 1].set_yscale('log')
        ax[0, 1].grid(True)
        ax[0, 1].set_axisbelow(True)
        ax[0, 1].legend()
        
        # load the original and noise added reconstructions
        with h5py.File(os.path.join(args.save_dir, 'data.h5'), 'r') as f:
            p0_tr = f[groups[0]]['p0_tr'][()]
            noisy_tr = f[groups[0]][args.save_images][()]
        extent = [-1e3*cfg['dx']*cfg['crop_size']/2,
                1e3*cfg['dx']*cfg['crop_size']/2,
                -1e3*cfg['dx']*cfg['crop_size']/2,
                1e3*cfg['dx']*cfg['crop_size']/2]
        #vmin = np.min(np.array([np.min(p0_tr), np.min(noisy_tr)]))
        #vmax = np.max(np.array([np.max(p0_tr), np.max(noisy_tr)]))
        vmin = None
        vmax = None
        
        im1 = ax[1, 0].imshow(
            np.rot90(p0_tr), cmap='gray', extent=extent, vmin=vmin, vmax=vmax
        )
        ax[1, 0].set_title('Original reconstruction')
        ax[1, 0].set_xlabel('x (mm)')
        ax[1, 0].set_ylabel('y (mm)')
        cbar1 = plt.colorbar(im1, ax=ax[1, 0])
        cbar1.set_label('Pressure (Pa)')
        
        im2 = ax[1, 1].imshow(
            np.rot90(noisy_tr), cmap='gray', extent=extent, vmin=vmin, vmax=vmax
        )
        ax[1, 1].set_title('Noise added reconstruction')
        ax[1, 1].set_xlabel('x (mm)')
        ax[1, 1].set_ylabel('y (mm)')
        cbar2 = plt.colorbar(im2, ax=ax[1, 1])
        cbar2.set_label('Pressure (Pa)')
        
        fig.tight_layout()
        if args.plot_comparison.endswith('.png'):
            plt.savefig(args.plot_comparison)
        else:
            plt.savefig(args.plot_comparison+'.png')
            
            
        # (optional) seporate plot for the impulse response amplitude spectrum
        #fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        #f_mag_impulse_response = np.abs(np.fft.fft(irf))
        #f_mag_impulse_response = 2 * f_mag_impulse_response[0:int(1+cfg['Nt']/2)] / (cfg['Nt'])
        #f_mag_impulse_response[0] /= 2
        #ax.plot(f_array, f_mag_impulse_response)
        #ax.set_title('Impulse response function amplitude spectrum')
        #ax.set_xlabel('Frequency (MHz)')
        #ax.set_ylabel('Amplitude (Pa)')
        #ax.grid(True)
        #ax.set_axisbelow(True)
        #plt.tight_layout()
        #plt.savefig('impulse_response_function_amplitude_spectrum.png')
        
    
    