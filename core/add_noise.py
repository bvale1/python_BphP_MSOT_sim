import numpy as np
import json, h5py, os, timeit, logging, argparse
import acoustic_inverse_simulation
import func.utility_func as uf
from scipy.ndimage import convolve1d
from scipy.fft import fft, ifft, fftshift, fftfreq
from copy import deepcopy
import matplotlib.pyplot as plt


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


def add_gaussian_noise(data : np.array, cfg : dict, std : float = 2.0):
    '''
    Add Gaussian noise to the data with a given standard deviation.
    
    Parameters
    ----------
    data : np.ndarray
        The data to add noise to.
    cfg : dict
        The configuration dictionary containing the seed for the random number
        generator.
    std : float
        The standard deviation of the Gaussian noise to add.
    
    Returns
    -------
    noisy_data : np.ndarray
        The data with added Gaussian noise.
    cfg : dict
        The configuration dictionary with the seed and noise std added.
    '''
    seed = cfg['seed']
    cfg['noise_std'] = std
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
        cfg['seed'] = seed
        logging.info(f'no seed provided, random seed: {seed}')
    else:
        logging.info(f'seed provided by config: {seed}')
    rng = np.random.default_rng(seed)
    
    noisy_data = data + rng.normal(0, std, size=data.shape)
    
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
        '--save_dir', type=str, default='/mnt/fast/nobackup/users/wv00017/20231123_BphP_phantom.c139519.p12',
        help='path to simulation data directory'        
    )
    parser.add_argument(
        '--irf_path', type=str, default='invision_irf.npy',
    )
    parser.add_argument(
        '--save_signals', type=str, default=None, 
        help='key to save noise added signals to the data.h5 file'
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
        '--plot_comparison', type=str, default=None, #'add_noise_comparison.png',
        help='path to save a comparison of the original and noise \
            added signal and reconstructions'
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
    
    # check if the save directory exists
    if not os.path.exists(args.save_dir):
        raise FileNotFoundError(f'file {args.save_dir} not found')
    logging.debug(f'found directory {args.save_dir}, adding noise...')
    
    # check input arguments
    if args.save_signals == 'sensor_data': # TODO: add option to overwrite
        raise ValueError('key sensor_data is reserved for the original signals')
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
       
    # load the simulated signals [256*2030*4*32/(1024**2) ~ 64 MB]
    if os.path.exists(os.path.join(args.save_dir, 'data.h5')):
        with h5py.File(os.path.join(args.save_dir, 'data.h5'), 'r') as f:
            sensor_data = f['sensor_data'][()].astype(np.float32)
        logging.info(f'loaded sensor data from {args.save_dir}/data.h5')
    else:
        raise FileNotFoundError(f'file {args.save_dir}/data.h5 not found')
    
    # initialise new dataset for noisy signals (optional) and noisey reconstructions
    if args.save_signals is not None:
        try:
            with h5py.File(os.path.join(args.save_dir, 'data.h5'), 'r+') as f:
                f.create_dataset(
                    args.save_signals,
                    shape=tuple(np.shape(sensor_data)),
                    dtype=np.float16
                )
            logging.info(f'noisy sensor dataset "{args.save_signals}" created')
        except Exception as e:
            # error is most likely because the dataset already exists, which is
            # fine as it will be overwritten
            logging.warning(f'{e} {args.save_signals}')
        
    if args.save_images is not None:
        try:
            with h5py.File(os.path.join(args.save_dir, 'data.h5'), 'r+') as f:
                f.create_dataset(
                    args.save_images,
                    shape=(
                        cfg['ncycles'],
                        len(cfg['wavelengths']),
                        cfg['npulses'],
                        cfg['crop_size'],
                        cfg['crop_size']
                    ),
                    dtype=np.float32
                )
            logging.info(f'noisy reconstruction dataset "{args.save_images}" created')
        except Exception as e: 
            # error is most likely because the dataset already exists, which is
            # fine as it will be overwritten
            logging.warning(f'{e} {args.save_images}')
        
    # 4. add appropriate white noise to the signals
    # same seed as simulation configuration is used to ensure reproducibility
    # instantiate random number generator
    (noisy_sensor_data, cfg) = add_gaussian_noise(deepcopy(sensor_data), cfg, cfg['noise_std'])
    logging.info(f'white noise added to sensor data with std: {cfg["noise_std"]}')
    
    # add the white noise standard deviation to the configuration then save it
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f)
    
    # 2. apply convolution with the impulse response of the sensor to the signals    
    irf = np.load("invision_irf.npy")
    noisy_sensor_data = convolve1d(noisy_sensor_data, irf, mode='nearest', axis=-1)
    logging.info('sensor data convolved with impulse response function')
    
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
    # apply the filter to the noisy sensor data
    noisy_sensor_data = np.fft.ifft(np.fft.fft(noisy_sensor_data) * filter)
    #noisy_sensor_data = convolve1d(noisy_sensor_data, np.fft.ifft(filter), mode='nearest', axis=-1)
    
    # 5. (optional) save the new signals
    if args.save_signals is not None:
        with h5py.File(cfg['save_dir']+'data.h5', 'r+') as f:
            f[args.save_signals] = noisy_sensor_data
        logging.info(f'sensor_data saved to {cfg["save_dir"]+"data.h5"}')
    
    # 6. run image reconstruction algorithm
    start = timeit.default_timer()
    simulation = acoustic_inverse_simulation.kwave_inverse_adapter(
        cfg,
        transducer_model=cfg['inverse_model']
    )
    simulation.configure_simulation()
    logging.info(f'kwave inverse initialised in \
                 {timeit.default_timer() - start} seconds')
    
    # iterate over the cycles, wavelengths and pulses
    
    for cycle in range(sensor_data.shape[0]):
        for wavelength_index in range(sensor_data.shape[1]):
            for pulse in range(sensor_data.shape[2]):
    
                logging.info(f'time reversal, cycle: {cycle+1}, \
                                wavelength_index: {wavelength_index+1}, \
                                pulse: {pulse+1}')
                
                start = timeit.default_timer()
                noisy_tr = simulation.run_time_reversal(
                    noisy_sensor_data[cycle, wavelength_index, pulse])
                noisy_tr = uf.square_centre_crop(noisy_tr, cfg['crop_size'])
                logging.info(f'time reversal run in \
                                {timeit.default_timer() - start} seconds')
                
                # 7. (optional) save the reconstructed images
                if args.save_images is not None:
                    start = timeit.default_timer()
                    with h5py.File(cfg['save_dir']+'data.h5', 'r+') as f:
                        f[args.save_images][cycle, wavelength_index, pulse] = noisy_tr
                    logging.info(f'reconstruction saved in {timeit.default_timer() - start} seconds')


    # for testing purposes save a comparison of the original and noise added 
    # signal and reconstruction for the first cycle, wavelength and pulse only
    if args.plot_comparison:
        
        t_array = np.arange(cfg['Nt']) * cfg['dt'] * 1e6
        # 1 + dx//2 includes the dc component, positive and Nyquist frequencies
        k = np.arange(1+cfg['Nt']//2)
        f_array = k / (cfg['Nt'] * cfg['dt'] ) * 1e-6
        
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Comparison of original and nois added signals and reconstructions')
        
        ax[0, 0].plot(t_array, noisy_sensor_data[0, 1, 0, 0], label='noise added', linestyle='--')
        ax[0, 0].plot(t_array, sensor_data[0, 1, 0, 0], label='original', linestyle=':')        
        ax[0, 0].set_title('Time domain signals')
        ax[0, 0].set_xlabel(r'Time ($\mu$s)')
        ax[0, 0].set_ylabel('Amplitude (Pa)')
        ax[0, 0].grid(True)
        ax[0, 0].set_axisbelow(True)
        ax[0, 0].legend()
        
        f_mag_sensor_data = np.abs(np.fft.fft(sensor_data[0, 1, 0, 0]))
        # normalise the magnitude of the fft and only take below the nyquist frequency
        # the factor of 2 is to account for the negative frequencies
        f_mag_sensor_data = 2 * f_mag_sensor_data[0:int(1+cfg['Nt']/2)] / (cfg['Nt'])
        f_mag_sensor_data[0] /= 2 # DC component does not get doubled
        f_mag_noisy_sensor_data = np.abs(np.fft.fft(noisy_sensor_data[0, 1, 0, 0]))
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
        with h5py.File(cfg['save_dir']+'data.h5', 'r') as f:
            p0_tr = f['p0_tr'][0, 1, 0]
            noisy_tr = f[args.save_images][0, 1, 0]
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
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        f_mag_impulse_response = np.abs(np.fft.fft(irf))
        f_mag_impulse_response = 2 * f_mag_impulse_response[0:int(1+cfg['Nt']/2)] / (cfg['Nt'])
        f_mag_impulse_response[0] /= 2
        ax.plot(f_array, f_mag_impulse_response)
        ax.set_title('Impulse response function amplitude spectrum')
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Amplitude (Pa)')
        ax.grid(True)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig('impulse_response_function_amplitude_spectrum.png')
        
    
    