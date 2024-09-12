import numpy as np
from phantoms.fluence_correction_phantom import fluence_correction_phantom
from scipy.ndimage import convolve1d
import json, h5py, os, timeit, logging, argparse, gc
import func.plot_func as pf
import func.utility_func as uf
import optical_simulation
import acoustic_forward_simulation
import acoustic_inverse_simulation


# same function as in https://github.com/bvale1/MSOT_Diffusion.git
def load_sim(path : str, args='all', verbose=False) -> list:
    data = {}
    with h5py.File(os.path.join(path, 'data.h5'), 'r') as f:
        images = list(f.keys())
        if verbose:
            print(f'images found {images}')
        if args == 'all':
            args = f[images[0]].keys()
            print(f'args found in images[0] {args}')
        for image in images:
            data[image] = {}
            for arg in args:
                if arg not in f[image].keys():
                    print(f'arg {arg} not found in {image}')
                    pass
                # include 90 deg anticlockwise rotation
                elif arg != 'sensor_data':
                    data[image][arg] = np.rot90(
                        np.array(f[image][arg][()]), k=1, axes=(-2,-1)
                    ).copy()
                else:
                    data[image][arg] = np.array(f[image][arg][()])
            
    with open(path+'/config.json', 'r') as f:
        cfg = json.load(f)
        
    return [data, cfg]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Iterative model-based reconstruction of absorption coefficient'
    )
    parser.add_argument(
        '--mcx_bin_path', type=str,
        default='/home/wv00017/mcx/bin/mcx',
        action='store',
        help='path to MCX CUDA binary'
    )
    parser.add_argument(
        '--save_dir', type=str,
        default='mua_recovery_test',
        action='store',
        help='directory to save simulation data'
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
        '--mu_s_guess', type=float, default=10000, action='store',
        help='Guess for scattering coefficient (m^-1)'
    )
    parser.add_argument(
        '--mu_a_guess', type=float, default=30, action='store', 
        help='Guess for absorption coefficient (m^-1)'
    )
    parser.add_argument(
        '--update_scheme', choices=['adjoint', 'gradient'], default='adjoint', 
        action='store', help='adjoint minimises the error between the forward model \
            reconstructed pressure and the pressure reconstructed from the "observed" data, \
            gradient minimises the error between the forward model initial pressure and the \
            pressure reconstructed from the "observed" data'
    )
    parser.add_argument('--dataset', type=str, help='path to dataset')
    parser.add_argument('--niter', type=int, help='Number of iterations', default=10)
    parser.add_argument('--crop_size', type=int, default=256, action='store')
    parser.add_argument('--sim_git_hash', type=str, default=None, action='store')
    parser.add_argument('--recon_iterations', type=int, default=5, action='store')
    parser.add_argument('--recon_alpha', type=float, default=1.0, action='store', )
    parser.add_argument('--forward_model', choices=['invision', 'point'], default='invision', action='store')
    parser.add_argument('--inverse_model', choices=['invision', 'point'], default='invision', action='store')
    parser.add_argument('--crop_p0_3d_size', type=int, default=512, action='store')
    parser.add_argument('--delete_p0_3d', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('-v', type=str, help='verbose level', default='INFO')
    parser.add_argument('--Gamma', type=float, default=1.0, action='store', help='Gruneisen parameter')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=False, help='plot results')
    
    args = parser.parse_args()
    
    if args.v == 'INFO':
        logging.basicConfig(level=logging.INFO)
    elif args.v == 'DEBUG':
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.info(f'{args.v} not a recognised verbose level, using INFO instead')
    
    data, cfg = load_sim(args.dataset, args='all', verbose=False)
    cfg['mcx_bin_path'] = args.mcx_bin_path
    cfg['weights_dir'] = args.weights_dir
    cfg['irf_path'] = args.irf_path
    
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent='\t')
    
    images = list(data.keys())
    p0_recon = data[images[0]]['p0_tr']
    Phi_true = data[images[0]]['Phi']
    mu_a_true = data[images[0]]['mu_a']
    bg_mask = data[images[0]]['bg_mask']
    
    mu_a = args.mu_a_guess * np.ones( # [m^-1] starting guess for absorption coefficient
        (cfg['mcx_grid_size'][0], cfg['mcx_grid_size'][2]),
        dtype=np.float32
    )
    mu_s = args.mu_s_guess # [m^-1] assumed scattering coefficient
    wavelengths_m = [float(images[0].split('_')[-1]) * 1e-9] # [m]
    phantom = fluence_correction_phantom(bg_mask, wavelengths_m=wavelengths_m)
    
    # load impulse response function
    irf = np.load(args.irf_path)
    
    with h5py.File(os.path.join(cfg['save_dir'], 'temp.h5'), 'w') as f:
        logging.info('allocating storage for p0_3d temp.h5')
        f.create_dataset(
            'p0_3D',
            shape=(
                cfg['crop_p0_3d_size'],
                cfg['kwave_grid_size'][1],
                cfg['crop_p0_3d_size']
            ), dtype=np.float32
        )
    
    if args.plot:
        mu_a_plots = [mu_a_true, mu_a.copy()]
    
    # metrics are computed for each iteration
    metrics = {'RMSE': [], 'PSNR': []}
    for n in range(args.niter):
        logging.info(f'iteration {n+1}/{args.niter}')
        volume = phantom.create_volume(mu_a, mu_s, cfg)
        # optical simulation
        simulation = optical_simulation.MCX_adapter(cfg, source='invision')
    
        gc.collect()
    
        start = timeit.default_timer()
        # out can be energy absorbed, fluence, pressure, sensor data
        # or recontructed pressure, the variable is overwritten
        # multiple times to save memory
        out = simulation.run_mcx(
            args.mcx_bin_path,
            volume.copy()
        )
        logging.info(f'mcx run in {timeit.default_timer() - start} seconds')
        
        # convert from normalised fluence [mm^-2] -> [J m^-2]
        start = timeit.default_timer()
        out *= cfg['LaserEnergy'][0] * 1e6
        Phi = uf.square_centre_crop(
            out[0,:,(cfg['mcx_grid_size'][1]//2)-1,:].copy(), cfg['crop_size']
        )
        
        # save fluence, to data HDF5 file
        #with h5py.File(cfg['save_dir']+'data.h5', 'r+') as f:
        #    f[h5_group].create_dataset(
        #        'Phi',
        #        data=uf.square_centre_crop(
        #            out[:,(cfg['mcx_grid_size'][1]//2)-1,:], cfg['crop_size']
        #        ), dtype=np.float32
        #    )
        #logging.info(f'fluence saved in {timeit.default_timer() - start} seconds')
        
        start = timeit.default_timer()
        # calculate initial pressure [J m^-2] * [m^-1] -> [J m^-3] = [Pa]
        out *= cfg['gruneisen'] * volume[0]
        
        # save 3D p0 to temp.h5
        with h5py.File(os.path.join(cfg['save_dir'], 'temp.h5'), 'r+') as f:
            f['p0_3D'][()] =  uf.crop_p0_3D(
                out,
                [cfg['crop_p0_3d_size'], cfg['kwave_grid_size'][1], cfg['crop_p0_3d_size']]
            )
        logging.info(f'pressure saved in {timeit.default_timer() - start} seconds')    
                                        
        gc.collect()
        
        logging.info('optical stage complete')
        
        # delete mcx input and out files, they are not needed anymore
        simulation.delete_temporary_files()
        start = timeit.default_timer()
        # overwrite mcx simulation to save memory
        simulation = acoustic_forward_simulation.kwave_forward_adapter(
            cfg, 
            transducer_model=cfg['forward_model']
        )
        simulation.configure_simulation()
        logging.info(f'kwave forward initialised in {timeit.default_timer() - start} seconds')
        gc.collect()
            
        logging.info(f'k-wave forward simulation {n+1}/{args.niter}')
        start = timeit.default_timer()
        with h5py.File(os.path.join(cfg['save_dir'], 'temp.h5'), 'r') as f:
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
        #start = timeit.default_timer()
        #with h5py.File(cfg['save_dir']+'data.h5', 'r+') as f:
        #    f[h5_group].create_dataset(
        #        'sensor_data',
        #        data=out.astype(np.float16)
        #    )
        #logging.info(f'sensor data saved in {timeit.default_timer() - start} seconds')
        
        logging.info('acoustic forward stage complete')
        
        start = timeit.default_timer()
        simulation = acoustic_inverse_simulation.kwave_inverse_adapter(
            cfg,
            transducer_model=cfg['inverse_model']
        )
        simulation.configure_simulation()
        logging.info(f'kwave inverse initialised in {timeit.default_timer() - start} seconds')
        gc.collect()
            
        # load sensor data
        #start = timeit.default_timer()
        #with h5py.File(cfg['save_dir']+'data.h5', 'r') as f:
        #    out = f[h5_group]['sensor_data'][()].astype(np.float32)
        #logging.info(f'sensor data loaded in {timeit.default_timer() - start} seconds')
        
        start = timeit.default_timer()
        # apply convolution with the impulse response function
        out = convolve1d(out, irf, mode='nearest', axis=-1)
        # apply bandpass filter to the noisy sensor data
        if args.bandpass_filter:
            out = np.fft.ifft(np.fft.fft(out, axis=-1) * filter, axis=-1).real.astype(np.float32)
        logging.info(f'noise added in {timeit.default_timer() - start} seconds')

        start = timeit.default_timer()
        tr = simulation.run_time_reversal(out)
        logging.info(f'time reversal run in {timeit.default_timer() - start} seconds')

        #start = timeit.default_timer()
        #with h5py.File(cfg['save_dir']+'data.h5', 'r+') as f:
        #    f[h5_group].create_dataset(
        #        'p0_tr',
        #        data=uf.square_centre_crop(tr, cfg['crop_size']),
        #        dtype=np.float32
        #    )
        #logging.info(f'p0_recon saved in {timeit.default_timer() - start} seconds')
        
        # update scheme for model absorption coefficient,
        # small number added to denominator to improve numerical stability
        mu_a += (p0_recon - tr) / (cfg['gruneisen'] * Phi + 1e3)
        # non-negativity constraint
        mu_a = np.maximum(mu_a, 0)
        # segmentation mask used as boundary condition
        mu_a *= bg_mask
        
        if np.any(np.isnan(mu_a)):
            logging.info(f'{np.sum(~np.isfinite(mu_a)) / np.prod(mu_a.shape)}% of mu_a is not finite')
            exit(1)
        if np.any(mu_a > 150):
            logging.info(f'mu_a is possibly diverging, {np.sum(mu_a > 150) / np.prod(mu_a.shape)}% \
                of mu_a is greater than 150 m^-1, truncating mu_a to 150 m^-1')
            mu_a = np.minimum(mu_a, 150)
        
        # compute metrics
        metrics['RMSE'].append(np.sqrt(np.mean(((mu_a - mu_a_true)**2)[bg_mask])))
        metrics['PSNR'].append(20 * np.log10(np.max(mu_a_true) / np.sqrt(metrics['RMSE'][-1])))
    
        if args.plot:
            mu_a_plots.append(mu_a.copy())
    
    logging.info(metrics)
    if args.plot:
        mu_a_plots = np.asarray(mu_a_plots)
        labels=['ground truth', 'initial guess n=0']
        for n in range(1, args.niter+1):
            labels.append(f'n={n}')
        (fig, ax, frames) = pf.heatmap(
            mu_a_plots, 
            labels=labels,
            title=r'$\mu_{a}$',
            dx=cfg['dx'],
            sharescale=True,
            cmap='viridis',
            rowmax=4
        )
        fig.savefig(os.path.join(cfg['save_dir'], 'mu_a.png'))
        residuals = mu_a_plots[2:] - mu_a_true
        labels = []
        for n in range(1, args.niter+1):
            labels.append(f'n={n}, RMSE={metrics["RMSE"][n-1]:.2f}')
        (fig, ax, frames) = pf.heatmap(
            residuals, 
            labels=(1+np.arange(args.niter)).astype(str).tolist(),
            title=r'$\mu_{a}$ residuals',
            dx=cfg['dx'],
            sharescale=True,
            cmap='viridis',
            rowmax=4,
            vmin=np.minimum(-np.max(mu_a_true), residuals),
            vmax=np.maximum(np.max(mu_a_true), residuals)
        )
        fig.savefig(os.path.join(cfg['save_dir'], 'mu_a_residuals.png'))
        labels = [r'$\mu_{a}$ (m$^{-1}$)', r'$\mu_{s}$ (m$^{-1}$)',
                  r'$\Phi$ (J m$^{-2}$)', r'$p_{0}$ initial pressure (Pa)',
                  r'$\hat{p}_{0}$ reconstructed (Pa)']
        images = [mu_a_true, data[images[0]]['mu_s'], 
                  data[images[0]]['Phi'], 
                  data[images[0]]['mu_a']*data[images[0]]['Phi'], p0_recon]
        (fig, ax, frames) = pf.heatmap(images, dx=cfg['dx'], rowmax=5, labels=labels)
        fig.savefig(os.path.join(cfg['save_dir'], 'images.png'))
        
    # delete temp p0_3D dataset
    if cfg['delete_p0_3d'] is True:
        try:
            start = timeit.default_timer()
            os.remove(os.path.join(cfg['save_dir'], 'temp.h5'))
            logging.info(f'temp.h5 (p0_3D) deleted in {timeit.default_timer() - start} seconds')
        except:
            logging.debug('unable to delete temp.h5, (p0_3D) not found')