import numpy as np
from phantoms.fluence_correction_phantom import fluence_correction_phantom
from add_noise import make_filter
from scipy.ndimage import convolve1d
import json
import h5py
import os
import timeit
import logging
import argparse
import gc
import func.plot_func as pf
import func.utility_func as uf
import matplotlib.pyplot as plt
import optical_simulation
import acoustic_forward_simulation
import acoustic_inverse_simulation


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
        '--mu_s_guess', type=float, default=None, action='store',
        help='Guess for scattering coefficient (m^-1)'
    )
    parser.add_argument(
        '--mu_a_guess', type=float, default=30, action='store', 
        help='Guess for absorption coefficient (m^-1), if None assume mu_s is known exactly'
    )
    parser.add_argument(
        '--update_scheme', choices=['adjoint', 'gradient'], default='adjoint', 
        action='store', help='adjoint minimises the error between the forward model \
            reconstructed pressure and the pressure reconstructed from the "observed" data, \
            gradient minimises the error between the forward model initial pressure and the \
            pressure reconstructed from the "observed" data'
    )
    parser.add_argument('--step_size', type=float, default=1.0, action='store', help='learning rate/step size')
    parser.add_argument('--epsilon', type=float, default=1e-8, action='store', help='small number to prevent division by zero')
    parser.add_argument('--dataset', type=str, help='path to dataset')
    parser.add_argument('--niter', type=int, help='Number of iterations', default=10)
    parser.add_argument('--sim_git_hash', type=str, default=None, action='store')
    parser.add_argument('--recon_iterations', type=int, default=5, action='store')
    parser.add_argument('--forward_model', choices=['invision', 'point'], default='invision', action='store')
    parser.add_argument('--inverse_model', choices=['invision', 'point'], default='invision', action='store')
    parser.add_argument('--delete_p0_3d', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('-v', type=str, help='verbose level', default='INFO')
    parser.add_argument('--Gamma', type=float, default=1.0, action='store', help='Gruneisen parameter')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=False, help='plot results')
    parser.add_argument(
        '--bandpass_filter', default=False, action=argparse.BooleanOptionalAction,
        help='apply bandpass filter to sensor data'
    )
    
    args = parser.parse_args()
    
    if args.v == 'INFO':
        logging.basicConfig(level=logging.INFO)
    elif args.v == 'DEBUG':
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.info(f'{args.v} not a recognised verbose level, using INFO instead')
    
    data, cfg = uf.load_sim(args.dataset, args='all', verbose=False)
    cfg['mcx_bin_path'] = args.mcx_bin_path
    cfg['weights_dir'] = args.weights_dir
    cfg['irf_path'] = args.irf_path
    
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent='\t')
    
    # for now only one image is used
    images = list(data.keys())
    data = {images[0]: data[images[0]]}
    p0_recon = data[images[0]]['p0_tr'].copy()
    p0_recon = uf.square_centre_pad(p0_recon, cfg['mcx_grid_size'][0])
    mu_a_true = data[images[0]]['mu_a'].copy()
    mu_a_true = uf.square_centre_pad(mu_a_true, cfg['mcx_grid_size'][0])
    Phi_true = data[images[0]]['Phi'].copy()
    Phi_true = uf.square_centre_pad(Phi_true, cfg['mcx_grid_size'][0])
    bg_mask = data[images[0]]['bg_mask'].copy().astype(bool)
    bg_mask = uf.square_centre_pad(bg_mask, cfg['mcx_grid_size'][0])
    
    # simulation is orientated at 90 deg anticlockwise
    p0_recon = np.rot90(p0_recon, k=1, axes=(-2,-1))
    mu_a_true = np.rot90(mu_a_true, k=1, axes=(-2,-1))
    bg_mask = np.rot90(bg_mask, k=1, axes=(-2,-1))
    
    wavelengths_m = [float(images[0].split('_')[-1]) * 1e-9] # [m]
    phantom = fluence_correction_phantom(bg_mask, wavelengths_m=wavelengths_m)
    H2O = phantom.define_H2O()
    
    mu_a = args.mu_a_guess * bg_mask.astype(np.float32) # [m^-1] starting guess for absorption coefficient
    mu_a += H2O['mu_a'][0] * (~bg_mask).astype(np.float32) # [m^-1] H2O outside of segmentation mask
    if args.mu_s_guess:
        mu_s = args.mu_s_guess # [m^-1] assumed scattering coefficient
    else: # mu_s is known exactly
        mu_s = np.rot90(data[images[0]]['mu_s'].copy(), k=1, axes=(-2,-1))
        mu_s = uf.square_centre_pad(mu_s, cfg['mcx_grid_size'][0])
    
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
    
    with h5py.File(os.path.join(args.save_dir, 'temp.h5'), 'w') as f:
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
        mu_a_plots = [uf.square_centre_crop(
                          np.rot90(mu_a_true.copy(), k=-1, axes=(-2,-1)), cfg['crop_size']
                      ),
                      uf.square_centre_crop(
                          np.rot90(mu_a.copy(), k=-1, axes=(-2,-1)), cfg['crop_size']
                      )]
        recon_plots = [uf.square_centre_crop(
                           np.rot90(p0_recon.copy(), k=-1, axes=(-2,-1)), cfg['crop_size']
                       )]
        Phi_plots = [uf.square_centre_crop(Phi_true.copy(), cfg['crop_size'])]
        mu_a_line_profiles = [mu_a_plots[0][mu_a_plots[0].shape[0]//2,:],
                              mu_a_plots[1][mu_a_plots[1].shape[0]//2,:]]
        recon_line_profiles = [recon_plots[0][recon_plots[0].shape[0]//2,:]]
    
    # metrics are computed for each iteration
    metrics = {'RMSE_mu_a': [], 'RMSE_p0_tr': [],
               'PSNR_mu_a': [], 'PSNR_p0_tr': [],
               'SSIM_mu_a': [], 'SSIM_p0_tr': []}
    for n in range(args.niter):
        logging.info(f'iteration {n+1}/{args.niter}')
        volume = phantom.create_volume(mu_a, mu_s, cfg)
        volume = np.rot90(volume, k=2, axes=(-3,-1))
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
        Phi = out[:,(cfg['mcx_grid_size'][1]//2)-1,:].copy()
        Phi = np.rot90(Phi, k=2, axes=(-2,-1))
        
        if args.update_scheme == 'gradient':
            mu_a = p0_recon / (cfg['gruneisen'] * Phi + args.epsilon)
        
        else: # adjoint
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
            with h5py.File(os.path.join(args.save_dir, 'temp.h5'), 'r+') as f:
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
            with h5py.File(os.path.join(args.save_dir, 'temp.h5'), 'r') as f:
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
                out = np.fft.ifft(
                    np.fft.fft(out, axis=-1) * filter, axis=-1
                ).real.astype(np.float32)
            logging.info(f'noise added in {timeit.default_timer() - start} seconds')

            start = timeit.default_timer()
            tr = simulation.run_time_reversal(out)
            tr = np.rot90(tr, k=2, axes=(-2,-1))
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
            logging.info(f'mu_a {mu_a.dtype} {mu_a.shape}')
            mu_a = mu_a.astype(np.float32)
            mu_a += args.step_size * (p0_recon - tr) / (cfg['gruneisen'] * Phi + args.epsilon)
            # non-negativity constraint
            mu_a = np.maximum(mu_a, 0)
            # segmentation mask used as boundary condition
            mu_a *= bg_mask.astype(np.float32) # [m^-1] absorption coefficient
            mu_a += H2O['mu_a'][0] * (~bg_mask).astype(np.float32) # [m^-1] H2O outside of segmentation mask
            
        if np.any(np.isnan(mu_a)):
            logging.info(f'{np.sum(~np.isfinite(mu_a)) / np.prod(mu_a.shape)}% of mu_a is not finite')
            exit(1)
        if np.any(mu_a > 150):
            logging.info(f'mu_a is possibly diverging, {np.sum(mu_a > 150) / np.prod(mu_a.shape)}% of mu_a is greater than 150 m^-1, truncating mu_a to 150 m^-1')
            mu_a = np.minimum(mu_a, 150)
        
        # compute metrics
        metrics['RMSE_mu_a'].append(uf.masked_RMSE(mu_a_true, mu_a, bg_mask))
        metrics['RMSE_p0_tr'].append(uf.masked_RMSE(p0_recon, tr, bg_mask))
        metrics['PSNR_mu_a'].append(uf.masked_PSNR(mu_a_true, mu_a, bg_mask))
        metrics['PSNR_p0_tr'].append(uf.masked_PSNR(p0_recon, tr, bg_mask))
        metrics['SSIM_mu_a'].append(uf.masked_SSIM(mu_a_true, mu_a, bg_mask))
        metrics['SSIM_p0_tr'].append(uf.masked_SSIM(p0_recon, tr, bg_mask))
            
        if args.plot:
            mu_a_plots.append(uf.square_centre_crop(
                np.rot90(mu_a.copy(), k=-1, axes=(-2,-1)), cfg['crop_size']
            ))
            Phi_plots.append(uf.square_centre_crop(
                np.rot90(Phi.copy(), k=-1, axes=(-2,-1)), cfg['crop_size']
            ))
            mu_a_line_profiles.append(mu_a_plots[-1][mu_a_plots[-1].shape[0]//2,:])
            if args.update_scheme == 'adjoint':
                recon_plots.append(uf.square_centre_crop(
                    np.rot90(tr.copy(), k=-1, axes=(-2,-1)), cfg['crop_size']
                ))
                recon_line_profiles.append(recon_plots[-1][recon_plots[-1].shape[0]//2,:])
    
    logging.info(metrics)
    if args.plot:
        mu_a_plots = uf.square_centre_crop(np.asarray(mu_a_plots), cfg['crop_size'])
        labels=['ground truth', 'initial guess n=0']
        for n in range(1, args.niter+1):
            labels.append(f'n={n}')
        (fig, ax, frames) = pf.heatmap(
            mu_a_plots, 
            labels=labels,
            title=r'$\mu_{\mathrm{a}}$',
            dx=cfg['dx'],
            sharescale=True,
            cmap='viridis',
            rowmax=4,
            cbar_label=r'm$^{-1}$'
        )
        fig.savefig(os.path.join(args.save_dir, 'mu_a.png'))
        residuals = mu_a_plots[2:] - uf.square_centre_crop(
            np.rot90(mu_a_true.copy(), k=-1, axes=(-2,-1)), cfg['crop_size']
        )
        labels = []
        for n in range(1, args.niter+1):
            labels.append(f'n={n}, RMSE_mu_a={metrics["RMSE_mu_a"][n-1]:.2f}')
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
        fig.savefig(os.path.join(args.save_dir, 'mu_a_residuals.png'))
        labels=['ground truth']
        for n in range(1, args.niter+1):
            labels.append(f'n={n}')
            
        (fig, ax) = plt.subplots(1, 1, figsize=(5, 5))
        labels = ['ground truth', 'initial guess n=0']
        for n in range(1, args.niter+1):
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
        )
        for i in range(len(mu_a_line_profiles)):
            ax.plot(line_profile_axis, mu_a_line_profiles[i], label=labels[i],
                    color=colors[i], alpha=0.8)
        ax.set_title('Line profile')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel(r'$\mu_{\mathrm{a}}$ (m$^{-1}$)')
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_xlim(np.min(line_profile_axis), np.max(line_profile_axis))
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.save_dir, 'mu_a_line_profile.png'))
            
        if args.update_scheme == 'adjoint':
            (fig, ax) = plt.subplots(1, 1, figsize=(5, 5))
            for i in range(len(recon_line_profiles)):
                ax.plot(line_profile_axis, recon_line_profiles[i], 
                        label=labels[i], color=colors[i], alpha=0.8)
            ax.set_title('Line profile')
            ax.set_xlabel('x (mm)')
            ax.set_ylabel(r'$\hat{p}_{0}$ (Pa)')
            ax.grid(True)
            ax.set_axisbelow(True)
            ax.set_xlim(np.min(line_profile_axis), np.max(line_profile_axis))
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(args.save_dir, 'reconstructions_line_profile.png'))
            
            (fig, ax, frames) = pf.heatmap(
                np.asarray(recon_plots), 
                labels=labels,
                title=r'$\hat{p}_{0}$',
                dx=cfg['dx'],
                sharescale=True,
                cmap='viridis',
                rowmax=4,
                cbar_label='Pa'
            )
            fig.savefig(os.path.join(args.save_dir, 'p0_recon.png'))
        (fig, ax, frames) = pf.heatmap(
            np.asarray(Phi_plots), 
            labels=labels,
            title=r'$\Phi$',
            dx=cfg['dx'],
            sharescale=True,
            cmap='viridis',
            rowmax=4,
            cbar_label=r'J m$^{-2}$'
        )
        fig.savefig(os.path.join(args.save_dir, 'Phi.png'))
        labels = [r'$\mu_{a}$ (m$^{-1}$)', r'$\mu_{s}$ (m$^{-1}$)',
                  r'$\Phi$ (J m$^{-2}$)', r'$p_{0}$ initial pressure (Pa)',
                  r'$\hat{p}_{0}$ reconstructed (Pa)']
        images = [data[images[0]]['mu_a'], 
                  data[images[0]]['mu_s'], 
                  data[images[0]]['Phi'], 
                  data[images[0]]['mu_a']*data[images[0]]['Phi'],
                  data[images[0]]['p0_tr']]
        (fig, ax, frames) = pf.heatmap(
            np.asarray(images), dx=cfg['dx'], rowmax=5, labels=labels
        )
        fig.savefig(os.path.join(args.save_dir, 'images.png'))
        
    # delete temp p0_3D dataset
    if args.delete_p0_3d is True:
        try:
            start = timeit.default_timer()
            os.remove(os.path.join(cfg['save_dir'], 'temp.h5'))
            logging.info(f'temp.h5 (p0_3D) deleted in {timeit.default_timer() - start} seconds')
        except:
            logging.debug('unable to delete temp.h5, (p0_3D) not found')