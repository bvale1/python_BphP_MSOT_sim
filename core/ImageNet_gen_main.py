import numpy as np
from phantoms.ImageNet_phantom import ImageNet_phantom
from add_noise import make_filter, add_noise
from scipy.ndimage import convolve1d
import json, h5py, os, timeit, logging, argparse, gc, glob
import func.geometry_func as gf
import func.utility_func as uf
import optical_simulation
import acoustic_forward_simulation
import acoustic_inverse_simulation



if __name__ == '__main__':

    '''
    ==================================Workflow==================================

    0. Parse command line arguments
    
    1. Check for checkpointed simulation of the same name
    
    2. load image from the ImageNet dataset
    
    3. Define Simulation Geometry
        - Domain size (x,y,z) in grid points and dx in m
            ~ absorbtion coeff (mu_a) in m^-1
            ~ scattering coeff (mu_s) in m^-1

    2. Generate volume array to define simulation in MCX
        - save volume to binary file
        - save configuration to JSON file

    3. Execute MCX CUDA binary through Python

    4. Get normalised fluence map (array) from MCX output

    6. Calculate inital acoustic pressure (p0) of sample

    7. Propogate p0 using K-wave

    8. Reconstruct images from transducer array data

    9. Repeat 3-8 for nimages argument

    ============================================================================
    '''
    
    # If the configuration file and data file already exist, most of these
    # parsed arguments will be ignored and the simulation will continue from the
    # last checkpoint
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mcx_bin_path', type=str,
        default='/home/wv00017/mcx/bin/mcx',
        action='store',
        help='path to MCX CUDA binary'
    )
    parser.add_argument(
        '--save_dir', type=str,
        default='unnamed_sim',
        action='store',
        help='directory to save simulation data'
    )
    parser.add_argument(
        '--ImageNet_dir', type=str,
        default='/mnt/fast/datasets/still/ImageNet/ILSVRC2012/TrainingSet/',
        action='store',
        help='directory containing ImageNet dataset'
    )
    parser.add_argument(
        '--irf_path', type=str,
        default='/mnt/fast/nobackup/users/wv00017/invision_irf.npy',
        action='store',
        help='path to the impulse response function of the invision transducer'
    )
    parser.add_argument(
        '--weights_dir', type=str, 
        default='/home/wv00017/python_BphP_MSOT_sim/invision_weights/',
        action='store',
        help='directory containing integration weights for combining sensor data'
    )
    parser.add_argument(
        '--in_progress_file', type=str,
        default='/mnt/fast/nobackup/users/wv00017/ImageNet_fluence_correction/ImageNet_checkpoint_file.json',
        action='store', help='file to keep track of which images from ImageNet \
        have been used in simulations'
    )
    parser.add_argument(
        '--ppw', type=int, default=2, action='store',
        help='points per wavelength, only lower to 1 to test code if VRAM is limited'
    )
    parser.add_argument('--nimages', type=int, default=16, action='store')
    parser.add_argument('--seed', type=int, default=None, action='store')
    parser.add_argument('--crop_size', type=int, default=256, action='store')
    parser.add_argument('--sim_git_hash', type=str, default=None, action='store')
    parser.add_argument('--recon_iterations', type=int, default=5, action='store')
    parser.add_argument('--recon_alpha', type=float, default=1.0, action='store')
    parser.add_argument('--forward_model', type=str, default='invision', action='store')
    parser.add_argument('--inverse_model', type=str, default='invision', action='store')
    parser.add_argument('--crop_p0_3d_size', type=int, default=512, action='store')
    parser.add_argument('--delete_p0_3d', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('-v', type=str, help='verbose level', default='INFO')
    parser.add_argument('--Gamma', type=float, default=1.0, action='store', help='Gruneisen parameter')
    parser.add_argument(
        '--interp_data', default=False, action=argparse.BooleanOptionalAction,
        help='interpolate sensor data from 256 to 512 sensors'
    )
    parser.add_argument(
        '--noise_std', type=float, default=1.5, action='store',
        help='standard deviation Guassian noise to add to the sensor data'
    )
    parser.add_argument(
        '--bandpass_filter', default=False, action=argparse.BooleanOptionalAction,
        help='apply bandpass filter to sensor data'
    )
    parser.add_argument(
        '--wavelength_m', type=float, default=None, action='store',
        help='wavelength of light source [m]'
    )
    args = parser.parse_args()
    
    if args.v == 'INFO':
        logging.basicConfig(level=logging.INFO)
    elif args.v == 'DEBUG':
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.info(f'{args.v} not a recognised verbose level, using INFO instead')
        
    # path to MCX binary
    mcx_bin_path = args.mcx_bin_path   
    #mcx_bin_path = '/mcx/bin/mcx' # billy_docker
    #mcx_bin_path = '/home/wv00017/mcx/bin/mcx' # Billy's workstation
        
    if not os.path.exists(args.in_progress_file):
        with open(args.in_progress_file, 'w') as f:
            json.dump({}, f, indent='\t')
        
    if len(args.save_dir) != 0:
        if list(args.save_dir)[-1] != '/':
            args.save_dir += '/'
    
    # check for a checkpointed simulation of the same name
    if (os.path.exists(args.save_dir+'config.json') 
        and os.path.exists(args.save_dir+'data.h5')):
        
        with open(args.save_dir+'config.json', 'r') as f:
            cfg = json.load(f)
        logging.info(f'checkpoint config found {cfg}')
        
        if cfg['seed']:
            seed = cfg['seed']
            logging.info(f'seed provided: {seed}')
        else:
            seed = np.random.randint(0, 2**32 - 1)
            logging.info(f'no seed provided, random seed selected: {seed}')        
        rng = np.random.default_rng(seed)
        
        phantom = ImageNet_phantom(rng, cfg['wavelengths'])
        H2O = phantom.define_H2O()
        
    else:
        if args.seed:
            seed = args.seed
            logging.info(f'seed provided: {seed}')
        else:
            seed = np.random.randint(0, 2**32 - 1)
            logging.info(f'no seed provided, random seed selected: {seed}')        
        rng = np.random.default_rng(seed)
        
        # It is imperative that dx is small enough to support high enough 
        # frequencies and that [nx, ny, nz] have low prime factors i.e. 2, 3, 5
        c_0 = 1500.0 # speed of sound [m s^-1]
        # light source pairs are separted by 0.02474m in the y direction
        mcx_domain_size = [0.082, 0.025, 0.082] # [m]
        kwave_domain_size = [0.082, mcx_domain_size[1], 0.082] # [m]
        pml_size = 10 # perfectly matched layer size in grid points
        [mcx_grid_size, dx] = gf.get_sim_grid_size(
            mcx_domain_size,
            c0_min=c_0,
            pml_size=pml_size,
            points_per_wavelength=args.ppw,
            f_max=6.8e6
        )
        logging.info(f'maximum supported acoustic frequency: {1e-6 * c_0 / (dx * args.ppw)} MHz')
        mcx_domain_size = [mcx_grid_size[0]*dx,
                           mcx_grid_size[1]*dx,
                           mcx_grid_size[2]*dx]# [m], modify domain size for chosen dx
        kwave_grid_size = mcx_grid_size # [m] use same grid for acoustic sim
        kwave_domain_size = mcx_domain_size # [m] use same domain for acoustic sim
        
        if args.wavelength_m is None:
            # randomly select a wavelength between 650nm and 950nm
            args.wavelength_m = round(rng.uniform(65, 95)) * 1e-8
        
        # configure simulation
        cfg = {
            'save_dir' : args.save_dir,
            'sim_git_hash' : args.sim_git_hash, # git hash of simulation code
            'seed' : seed, # used in procedurally generated phantoms
            'nimages' : args.nimages, # number of images to simulate
            'nsensors' : 256,
            'nphotons' : 1e8,
            'nsources' : 10,
            'wavelengths' : [args.wavelength_m], # [m]
            'mcx_domain_size' : mcx_domain_size,
            'kwave_domain_size' : kwave_domain_size,
            'mcx_grid_size': mcx_grid_size,
            'kwave_grid_size' : kwave_grid_size,
            'points_per_wavelength' : args.ppw, # only lower to 1 to test if VRAM is limited
            'dx' : dx,
            'pml_size' : pml_size,
            'gruneisen' : args.Gamma, # gruneisen parameter
            'interp_data' : args.interp_data, # interpolate sensor data from 256 to 512 sensors
            'c_0' : c_0,
            'alpha_coeff' : 0.01,
            'alpha_power' : 1.1,
            'recon_iterations' : args.recon_iterations, # time reversal iterations
            'recon_alpha' : args.recon_alpha, # time reversal alpha (regularisation hyperparameter)
            'crop_size' : args.crop_size, # pixel with of output images and ground truth
            'stage' : 'optical', # for checkpointing (optical, acoustic, inverse)
            'weights_dir' : args.weights_dir, # directory containing weights for combining sensor data
            'forward_model' : args.forward_model, # forward model to use (invision, point)
            'inverse_model' : args.inverse_model, # inverse model to use (invision, point)
            'crop_p0_3d_size' : args.crop_p0_3d_size, # size of 3D p0 to crop to
            'phantom' : 'ImageNet_phantom',
            'delete_p0_3d' : args.delete_p0_3d, # delete p0_3d after each pulse to save memory
            'noise_std' : args.noise_std, # standard deviation of Guassian noise to add to sensor data
            'irf_path' : args.irf_path, # path to impulse response function
            'dt' : 25e-9, # time step [s]
            'Nt' : 2030, # number of time steps
            'bandpass_filter' : args.bandpass_filter, # apply bandpass filter to sensor data
        }
        
        logging.info(f'no checkpoint, creating config {cfg}')
        
        # Energy total delivered is wavelength dependant and normally disributed
        Emean = 70.0727 * 1e-3; # [J]
        Estd = 0.7537 * 1e-3; # [J]
        cfg['LaserEnergy'] = rng.normal(
            loc=Emean, scale=Estd, size=(cfg['nimages'])
        ).astype(np.float32)
        cfg['LaserEnergy'] = cfg['LaserEnergy'].tolist()
        
        # save configuration to JSON file
        uf.create_dir(cfg['save_dir'])
        with open(cfg['save_dir']+'config.json', 'w') as f:
            json.dump(cfg, f, indent='\t')
        
        phantom = ImageNet_phantom(rng, cfg['wavelengths'])
        H2O = phantom.define_H2O()
              
        ImageNet_files = glob.glob(f'{args.ImageNet_dir}/**/*.JPEG', recursive=True)
        with open(args.in_progress_file, 'r+') as f:
            # select nimages from ImageNet dataset
            
            uf.create_lock_dir(args.in_progress_file)
            # do not use the same image twice
            try:
                ckpt_dict = json.load(f)
            except:
                logging.info('checkpoint file is empty')
                ckpt_dict = {}
            used_image_files = ckpt_dict.keys()
            for dir in used_image_files:
                if dir in ImageNet_files:
                    ImageNet_files.remove(dir)
            
            if len(ImageNet_files) < cfg['nimages']:
                cfg['nimages'] = len(ImageNet_files)
                logging.info(f'{cfg["nimages"]} images in left ImageNet dataset')
            if cfg['nimages'] == 0:
                logging.info('no images left in ImageNet dataset')
                exit(0)
            ImageNet_files = rng.choice(ImageNet_files, cfg['nimages'], replace=False)
        
            for file in ImageNet_files:
                ckpt_dict[file] = {'save_dir' : cfg['save_dir']}
                ckpt_dict[file]['seed'] = cfg['seed']
                ckpt_dict[file]['sim_complete'] = False
                
            f.seek(0)  # Go back to the start of the file
            f.truncate()  # Truncate the file to remove old content
            json.dump(ckpt_dict, f, indent='\t')
            uf.delete_lock_dir(args.in_progress_file)
    
        with h5py.File(cfg['save_dir']+'data.h5', 'w') as f:
            logging.info('creating data.h5')
            pass
    
    # load impulse response function
    irf = np.load(args.irf_path)
    
    # intialise bandpass filter
    if args.bandpass_filter:
        filter = make_filter(
            n_samples=cfg['Nt'], fs=1/cfg['dt'], irf=irf,
            hilbert=True, lp_filter=6.5e6, hp_filter=50e3, rise=0.2,
            n_filter=512, window='hann'
        )
        logging.info('bandpass filter initialised')
    
    with h5py.File(cfg['save_dir']+'temp.h5', 'w') as f:
        logging.info('allocating storage for p0_3d temp.h5')
        f.create_dataset(
            'p0_3D',
            shape=(
                cfg['crop_p0_3d_size'],
                cfg['kwave_grid_size'][1],
                cfg['crop_p0_3d_size']
            ), dtype=np.float32
        )
    
    # get files in use by this simulation but not yet completed
    ckpt_dict = uf.load_json(args.in_progress_file)
    ImageNet_files = {
        k : v for k, v in ckpt_dict.items() if v['save_dir'] == cfg['save_dir']
    }
    logging.debug(f'checkpointed files: {ImageNet_files}')
    for i, image_file in enumerate(ImageNet_files.keys()):
        if ImageNet_files[image_file]['sim_complete'] is True:
            logging.info(f'{image_file} {i+1}/{len(ImageNet_files)} is complete')
            continue
        else:
            logging.info(f'simulation {image_file} {i+1}/{len(ImageNet_files)}')

        (volume, bg_mask) = phantom.create_volume(cfg, image_file)
        
        # save 2D slice of the volume to HDF5 file
        h5_group = image_file.replace('/', '__')
        with h5py.File(cfg['save_dir']+'data.h5', 'r+') as f:
            f.require_group(h5_group)
            f[h5_group].create_dataset(
                'mu_a',
                data=uf.square_centre_crop(
                    volume[0,:,(cfg['mcx_grid_size'][1]//2)-1,:], cfg['crop_size']
                ), dtype=np.float32
            )
            f[h5_group].create_dataset(
                'mu_s',
                data=uf.square_centre_crop(
                    volume[1,:,(cfg['mcx_grid_size'][1]//2)-1,:], cfg['crop_size']
                ), dtype=np.float32
            )
            f[h5_group].create_dataset(
                'bg_mask', data=uf.square_centre_crop(
                    bg_mask, cfg['crop_size']
                ), dtype=bool
            )
 
        if cfg['stage'] == 'optical':
            # optical simulation
            simulation = optical_simulation.MCX_adapter(cfg, source='invision')
        
            gc.collect()        
            logging.info(f'mcx, image: {image_file}')
        
            start = timeit.default_timer()
            # out can be energy absorbed, fluence, pressure, sensor data
            # or recontructed pressure, the variable is overwritten
            # multiple times to save memory
            out = simulation.run_mcx(
                mcx_bin_path,
                volume.copy()
            )
            logging.info(f'mcx run in {timeit.default_timer() - start} seconds')
            
            # convert from normalised fluence [mm^-2] -> [J m^-2]
            start = timeit.default_timer()
            out *= cfg['LaserEnergy'][i] * 1e6
            
            # save fluence, to data HDF5 file
            with h5py.File(cfg['save_dir']+'data.h5', 'r+') as f:
                f[h5_group].create_dataset(
                    'Phi',
                    data=uf.square_centre_crop(
                        out[:,(cfg['mcx_grid_size'][1]//2)-1,:], cfg['crop_size']
                    ), dtype=np.float32
                )
            logging.info(f'fluence saved in {timeit.default_timer() - start} seconds')
            
            start = timeit.default_timer()
            # calculate initial pressure [J m^-2] * [m^-1] -> [J m^-3] = [Pa]
            out *= cfg['gruneisen'] * volume[0]
            
            # save 3D p0 to temp.h5
            with h5py.File(cfg['save_dir']+'temp.h5', 'r+') as f:
                f['p0_3D'][()] =  uf.crop_p0_3D(
                    out,
                    [cfg['crop_p0_3d_size'], cfg['kwave_grid_size'][1], cfg['crop_p0_3d_size']]
                )
            logging.info(f'pressure saved in {timeit.default_timer() - start} seconds')    
                                            
        gc.collect()
        
        if cfg['stage'] == 'optical':
            logging.info('optical stage complete')
            cfg['stage'] = 'acoustic'
            
            # delete mcx input and out files, they are not needed anymore
            simulation.delete_temporary_files()
            
            with open(cfg['save_dir']+'config.json', 'w') as f:
                json.dump(cfg, f, indent='\t')
            
        # acoustic forward simulation
        if cfg['stage'] == 'acoustic':
            # initialise kwave forward adapter
            start = timeit.default_timer()
            # overwrite mcx simulation to save memory
            simulation = acoustic_forward_simulation.kwave_forward_adapter(
                cfg, 
                transducer_model=cfg['forward_model']
            )
            simulation.configure_simulation()
            logging.info(f'kwave forward initialised in {timeit.default_timer() - start} seconds')
            
            gc.collect()
            
            logging.info(f'k-wave forward, image: {image_file}')
            
            start = timeit.default_timer()
            with h5py.File(cfg['save_dir']+'temp.h5', 'r') as f:
                out = uf.pad_p0_3D(
                    f['p0_3D'],
                    cfg['kwave_grid_size'][0]
                )
            logging.info(f'p0 loaded in {timeit.default_timer() - start} seconds')
            
            start = timeit.default_timer()
            # run also saves the sensor data to data.h5 as float16
            out = simulation.run_kwave_forward(out)
            logging.info(f'kwave forward run in {timeit.default_timer() - start} seconds')
            
            if not np.any(out):
                logging.error('sensor data is all zeros')
                exit(1)                        
            
            start = timeit.default_timer()
            with h5py.File(cfg['save_dir']+'data.h5', 'r+') as f:
                f[h5_group].create_dataset(
                    'sensor_data',
                    data=out.astype(np.float16)
                )
            logging.info(f'sensor data saved in {timeit.default_timer() - start} seconds')
            
        if cfg['stage'] == 'acoustic':
            logging.info('acoustic stage complete')
            cfg['stage'] = 'inverse'
            with open(cfg['save_dir']+'config.json', 'w') as f:
                json.dump(cfg, f, indent='\t')
        
        start = timeit.default_timer()
        simulation = acoustic_inverse_simulation.kwave_inverse_adapter(
            cfg,
            transducer_model=cfg['inverse_model']
        )
        simulation.configure_simulation()
        logging.info(f'kwave inverse initialised in {timeit.default_timer() - start} seconds')
        
        gc.collect()
        
        # acoustic reconstruction
        if cfg['stage'] == 'inverse':
            with open(cfg['save_dir']+'/config.json', 'w') as f:
                json.dump(cfg, f, indent='\t')
            
            logging.info(f'time reversal, image: {image_file}')
            
            # load sensor data
            start = timeit.default_timer()
            with h5py.File(cfg['save_dir']+'data.h5', 'r') as f:
                out = f[h5_group]['sensor_data'][()].astype(np.float32)
            logging.info(f'sensor data loaded in {timeit.default_timer() - start} seconds')
            
            start = timeit.default_timer()
            # add noise to sensor data
            if cfg['noise_std'] > 0.0:
                (out, cfg) = add_noise(out, cfg, rng, std=cfg['noise_std'])
            # apply convolution with the impulse response function
            out = convolve1d(out, irf, mode='nearest', axis=-1)
            # apply bandpass filter to the noisy sensor data
            if args.bandpass_filter:
                out = np.fft.ifft(np.fft.fft(out, axis=-1) * filter, axis=-1).real.astype(np.float32)
            logging.info(f'noise added in {timeit.default_timer() - start} seconds')
    
            start = timeit.default_timer()
            tr = simulation.run_time_reversal(out)
            logging.info(f'time reversal run in {timeit.default_timer() - start} seconds')

            start = timeit.default_timer()
            with h5py.File(cfg['save_dir']+'data.h5', 'r+') as f:
                f[h5_group].create_dataset(
                    'p0_tr',
                    data=uf.square_centre_crop(tr, cfg['crop_size']),
                    dtype=np.float32
                )
            logging.info(f'p0_recon saved in {timeit.default_timer() - start} seconds')
            
            ckpt_dict = uf.load_json(args.in_progress_file)
            ckpt_dict[image_file]['sim_complete'] = True
            uf.save_json(args.in_progress_file, ckpt_dict)
            cfg['stage'] = 'optical'
            with open(cfg['save_dir']+'config.json', 'w') as f:
                json.dump(cfg, f, indent='\t')
            logging.info(f'{image_file} {i+1}/{len(ImageNet_files)} complete')

    # delete temp p0_3D dataset
    if cfg['delete_p0_3d'] is True:
        try:
            start = timeit.default_timer()
            os.remove(cfg['save_dir']+'temp.h5')
            logging.info(f'temp.h5 (p0_3D) deleted in {timeit.default_timer() - start} seconds')
        except:
            logging.debug('unable to delete temp.h5, (p0_3D) not found')