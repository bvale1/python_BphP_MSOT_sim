import numpy as np
import json, h5py, os, timeit, logging, argparse
import acoustic_inverse_simulation
import utility_func as uf
from scipy.signal import fftconvolve


if __name__ == '__main__':
    
    '''
    ==================================Workflow==================================

    1. Load simulated signals and the corresponding configuration file
    
    2. apply convolution with the impulse response of the sensor to the signals
    
    3. apply transformation to simulated signals (calibrated from real data),
       i.e. sytematic noise
    
    4. add appropriate white noise to the signals
    
    5. (optional) save the new signals
    
    6. run image reconstruction algorithm
    
    7. (optional) save the reconstructed images

    ============================================================================
    '''
    
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_dir', type=str, help='path to simulation data directory',
        required=True
    )
    parser.add_argument(
        '--save_signals', type=str, default=None, 
        help='key to save noise added signals to the data.h5 file'
    )
    parser.add_argument(
        '--save_images', type=str, default='noisy_p0_tr', 
        help='key to save reconstructed images to the data.h5 file'
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
    logging.debug(f'found directory {args.save_dir}')
    
    # check input arguments
    if args.save_signals == 'sensor_data': # TODO: add option to overwrite
        raise ValueError('key sensor_data is reserved for the original signals')
    if args.save_images == 'p0_tr': # TODO: add option to overwrite
        raise ValueError('key p0_tr is reserved for the original images')
        
    # 1. Load simulated signals and the corresponding configuration file
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
            sensor_data = np.array(f.get('sensor_data')).astype(np.float32)
        logging.info(f'loaded sensor data from {args.save_dir}/data.h5')
    else:
        raise FileNotFoundError(f'file {args.save_dir}/data.h5 not found')
    
    # initialise new dataset for noisy signals (optional) save the new signals
    if args.save_signals is not None:
        try:
            with h5py.File(os.path.join(args.save_dir, 'data.h5'), 'r+') as f:
                f.create_dataset(
                    args.save_signals,
                    shape=tuple(np.shape(sensor_data)),
                    dtype=np.float16
                )
        except Exception as e:
            logging.error(f'error creating dataset: {e}')
    noisy_sensor_data = np.zeros_like(sensor_data, dtype=np.float32)
    logging.info(f'noisy sensor dataset "{args.save_signals}" initialised')
    
    # 2. apply convolution with the impulse response of the sensor to the signals
    irf = np.load("invision_irf.npy")
    for cycle in range(sensor_data.shape[0]):
        for wavelength_index in range(sensor_data.shape[1]):
            for pulse in range(sensor_data.shape[2]):
                for sensor in range(sensor_data.shape[3]):
                    noisy_sensor_data[cycle, wavelength_index, pulse, sensor] =  \
                        fftconvolve(
                            noisy_sensor_data[cycle, wavelength_index, pulse, sensor],
                            irf, mode='same'
                        )
    
    # 3. apply transformation to simulated signals (calibrated from real data),
    #    i.e. sytematic noise
    # These were fitted for Janek's particular data using an optimisation script
    sim_factor = 0.065576
    noise_factor = 1.640936
    INTERCEPT = -0.100539
    
    # systematic noise measured for 21 wavelengths (700-900 nm)
    noise = np.load(f"P.N.3.npz")["raw_data"]
    noise_wavelengths = np.arange(700, 901, 10)
    for cycle in range(sensor_data.shape[0]):
        for wavelength_index in range(sensor_data.shape[1]):
            wavelength = cfg['wavelengths'][wavelength_index]*1e9
            noise_ts = noise[np.argwhere(noise_wavelengths == wavelength).item(), :, :]
            noise_ts = (noise_ts - np.mean(noise_ts[:, 600:1700], axis=1)[:, np.newaxis]) * noise_factor
            
    
    # 4. add appropriate white noise to the signals
    
    # 5. (optional) save the new signals
    if args.save_signals is not None:
        with h5py.File(cfg['save_dir']+'data.h5', 'r+') as f:
            f[args.save_signals] = noisy_sensor_data
        logging.info(f'sensor_data saved to {cfg["save_dir"]}data.h5')
    
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
                tr = simulation.run_time_reversal(
                    sensor_data[cycle, wavelength_index, pulse])
                logging.info(f'time reversal run in \
                             {timeit.default_timer() - start} seconds')
                
                # 7. (optional) save the reconstructed images
                if args.save_images is not None:
                    start = timeit.default_timer()
                    with h5py.File(cfg['save_dir']+'data.h5', 'r+') as f:
                        f[args.save_images][cycle, wavelength_index, pulse] = uf.square_centre_crop(tr, cfg['crop_size'])
                    logging.info(f'reconstruction saved in {timeit.default_timer() - start} seconds')

    
    