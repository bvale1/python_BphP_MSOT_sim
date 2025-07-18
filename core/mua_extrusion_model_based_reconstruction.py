import numpy as np
from phantoms.fluence_correction_phantom import fluence_correction_phantom
from add_noise import make_filter, add_noise
from scipy.ndimage import convolve1d, zoom
from scipy.interpolate import interp1d
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


# H_recon_ref(x,z) <- the reference image, reconstructed from the measured signals 
# H_recon_pred(x,z) <- the predicted image, reconstructed from the model
# H_pred(x,z) <- the predicted heat energy deposition image, from the model
# PSF(i,j) <- the point spread function of the imaging system
# mu_a_ref(x,z) <- the reference absorption coefficient
# mu_a_pred(x,z) <- the predicted absorption coefficient
# mu_s(x,z) <- the scattering coefficient, either known exactly or approximated as uniform
# Phi_pred(x,z) <- the predicted fluence, from the model


def TV(mu : np.ndarray) -> float:
    # This is anisotropic total variation, normalised by the number of partitions ((2*X*Z)-X-Z)
    # https://en.wikipedia.org/wiki/Total_variation_denoising 
    # currently not in use because the gradient with respect to pixel (x,z)
    # depends only on (x,z), (x+1,z) and (x,z+1)
    X, Z = mu.shape
    TV_x = np.abs(mu[1:,:] - mu[:-1,:])
    TV_z = np.abs(mu[:,1:] - mu[:,:-1])
    return (TV_x.sum() + TV_z.sum()) / ((2*X*Z)-X-Z)


def compute_number_partitions(mask : np.ndarray) -> np.ndarray:
    # compute the number of partitions in a binary mask
    # if pixels in the image are vertices and an edge connects each adjacent pixel
    # then this function counts the number of edges connected to 1s at both ends
    kernel = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.float32)
    # this is used to normalise the total variation
    return padded_convolution(mask.astype(np.float32), kernel)[mask.astype(bool)].sum() // 2


def masked_TV(mu : np.ndarray, mask : np.ndarray) -> float:
    # In this version of anisotropic total variation, each partition is counted twice,
    # which is factored into the normalisation (compute_number_partitions(mask))
    # only pixels in the mask contribute to the total variation
    muxz_minus_muxplus1z = np.abs(mu[:-1,:] - mu[1:,:])[mask[1:,:]] # |(x, z) - (x+1, z)|
    muxz_minus_muxzplus1 = np.abs(mu[:,:-1] - mu[:,1:])[mask[:,1:]] # |(x, z) - (x, z+1)|
    muxz_minus_muxminus1z = np.abs(mu[1:,:] - mu[:-1,:])[mask[:-1,:]] # |(x, z) - (x-1, z)|
    muxz_minus_muxzminus1 = np.abs(mu[:,1:] - mu[:,:-1])[mask[:,:-1]] # |(x, z) - (x, z-1)|
    n_partitions = compute_number_partitions(mask)
    return (muxz_minus_muxplus1z.sum() + muxz_minus_muxzplus1.sum() \
        + muxz_minus_muxminus1z.sum() + muxz_minus_muxzminus1.sum()) / n_partitions


def masked_grad_TV(mu : np.ndarray, mask : np.ndarray, eps=1e-8) -> float:
    # compute the gradient of masked_TV(X, mask) with respect to Xij
    gradTV = np.zeros_like(mu, dtype=np.float32)
    # small number eps added to prevent division by zero
    muxz_minus_muxplus1z = mu[:-1,:] - mu[1:,:] # (x, z) - (x+1, z)
    gradTV[:-1,:] += muxz_minus_muxplus1z * mask[1:,:] / (np.abs(muxz_minus_muxplus1z) + eps)
    muxz_minus_muxzplus1 = mu[:,:-1] - mu[:,1:] # (x, z) - (x, z+1)
    gradTV[:,:-1] += muxz_minus_muxzplus1 * mask[:,1:] / (np.abs(muxz_minus_muxzplus1) + eps)
    muxz_minus_muxminus1z = mu[1:,:] - mu[:-1,:] # (x, z) - (x-1, z)
    gradTV[1:,:] += muxz_minus_muxminus1z * mask[:-1,:] / (np.abs(muxz_minus_muxminus1z) + eps)
    muxz_minus_muxzminus1 = mu[:,1:] - mu[:,:-1] # (x, z) - (x, z-1)
    gradTV[:,1:] += muxz_minus_muxzminus1 * mask[:,:-1] / (np.abs(muxz_minus_muxzminus1) + eps)
    n_partitions = compute_number_partitions(mask)
    return gradTV / n_partitions


def padded_convolution(H : np.ndarray, PSF : np.ndarray) -> np.ndarray:
    I = PSF.shape[0]//2
    J = PSF.shape[1]//2
    # pad with zeros
    H = np.pad(H, ((I, J), (I, J)), mode='constant') # (x+2I, z+2J)
    # perform convolution using sliding window view
    H_window = np.lib.stride_tricks.sliding_window_view(H, PSF.shape, axis=(0, 1)) # (x, y, i, j)
    # compute the convolution
    return np.sum(H_window * PSF, axis=(-1, -2)) # (x, y)


def masked_MSE(H_ref : np.ndarray, H_pred : np.ndarray, mu_pred : np.ndarray, mask : np.ndarray) -> dict:
    mask_sum = float(mask.sum())
    masked_SE = ((H_ref[mask] - H_pred[mask])**2)
    return {'masked_MSE' : masked_SE.sum() / mask_sum, 'squared_error' : masked_SE}


def grad_masked_MSE_loss(H_recon_ref : np.ndarray,
                         H_recon_pred : np.ndarray,
                         PSF : np.ndarray,
                         Phi_pred : np.ndarray,
                         mask : np.ndarray) -> np.ndarray:
    I = PSF.shape[0]//2
    J = PSF.shape[1]//2
    # compute the gradient of the masked MSE and TV loss with respect to mu_pred
    mask_sum = float(mask.sum())
    H_recon_window = np.pad((H_recon_pred - H_recon_ref), ((I, J), (I, J)), mode='constant') # (x+2I, z+2J)
    H_recon_window = np.lib.stride_tricks.sliding_window_view(H_recon_window, PSF.shape, axis=(0, 1)) # (x, y, i, j)
    PSF = np.flip(PSF, axis=(-2, -1)) # flip PSF for convolution
    grad_MSE = 2 * Phi_pred * np.sum(H_recon_window * PSF, axis=(-1, -2)) # (x, y)
    return grad_MSE / mask_sum


class TestMetricCalculator():
    def __init__(self) -> None:
        self.metrics = {
            'RMSE' : [],
            'MAE' : [],
            'Rel_Err' : [],
            'PSNR' : [],
            'SSIM' : [],
            'R2' : []
        }
    
    def __call__(self, Y : np.ndarray, Y_hat : np.ndarray, Y_mask=None) -> None:
        assert Y.shape == Y_hat.shape, f"Y.shape {Y.shape} must equal \
            Y_hat.shape {Y_hat.shape}"
        assert Y.ndim == 2, f"Y.ndim {Y.ndim} must be of shape (H, W)"
        Y = Y.reshape(-1)
        Y_hat = Y_hat.reshape(-1)
        if type(Y_mask) == np.ndarray:
            Y_mask = Y_mask.reshape(-1)
            Y_mask_sum = Y_mask.sum()
            Y_max = (Y*Y_mask).max()
        else:
            Y_max = Y.max()
        
        if type(Y_mask) == np.ndarray:
            RMSE = np.sqrt((((Y - Y_hat)*Y_mask)**2).sum() / Y_mask_sum)
            MAE = np.abs((Y - Y_hat)*Y_mask).sum() / Y_mask_sum
            Rel_Err = 100 * np.abs((Y - Y_hat)*Y_mask/Y).sum() / Y_mask_sum
            mean_Y = (Y*Y_mask).sum() / Y_mask_sum
            mean_Y_hat = (Y_hat*Y_mask).sum() / Y_mask_sum
            var_Y = (((Y - mean_Y)**2)*Y_mask).sum() / Y_mask_sum
            var_Y_hat = (((Y_hat - mean_Y_hat)**2)*Y_mask).sum() / Y_mask_sum
            cov_Y_Y_hat = ((Y - mean_Y)*(Y_hat - mean_Y_hat)*Y_mask).sum() / Y_mask_sum
            SSr = (((Y - Y_hat)**2)*Y_mask).sum()
            SSt = (((Y - mean_Y)**2)*Y_mask).sum()
        else:
            RMSE = np.sqrt(np.mean((Y - Y_hat)**2))
            MAE = np.mean(np.abs(Y - Y_hat))
            Rel_Err = np.mean(100 * np.abs(Y - Y_hat) / Y)
            mean_Y = np.mean(Y)
            mean_Y_hat = np.mean(Y_hat)
            var_Y = np.var(Y)
            var_Y_hat = np.var(Y_hat)
            cov_Y_Y_hat = np.mean((Y - mean_Y)*(Y_hat - mean_Y_hat))
            SSr = ((Y - Y_hat)**2).sum()
            SSt = ((Y - mean_Y)**2).sum()
            
        PSNR = 20*np.log10(Y_max / RMSE)
        c1 = (0.01 * Y_max)**2
        c2 = (0.03 * Y_max)**2
        SSIM = (2*mean_Y*mean_Y_hat + c1)*(2*cov_Y_Y_hat + c2) / \
            ((mean_Y**2 + mean_Y_hat**2 + c1)*(var_Y + var_Y_hat + c2))
        R2 = 1 - (SSr / SSt)
        
        self.metrics['RMSE'] += [RMSE]
        self.metrics['MAE'] += [MAE]
        self.metrics['Rel_Err'] += [Rel_Err]
        self.metrics['PSNR'] += [PSNR]
        self.metrics['SSIM'] += [SSIM]
        self.metrics['R2'] += [R2]
        
    def get_metrics(self) -> dict:
        return self.metrics
    
    def save_metrics_all_test_samples(self, save_path : str) -> None:
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Iterative model-based reconstruction of absorption coefficient'
    )
    parser.add_argument(
        '--dataset', type=str,
        #default='/home/wv00017/MSOT_Diffusion/20250716_digimouse_extrusion_MSOT_Dataset',
        default='/home/wv00017/MSOT_Diffusion/20250327_digimouse_extrusion_MSOT_Dataset',
        help='path to dataset'
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
        #default='/mnt/fast/nobackup/users/wv00017/invision_irf.npy',
        default='/home/wv00017/python_BphP_MSOT_sim/invision_irf.npy',
        action='store',
        help='path to the impulse response function of the invision transducer'
    )
    parser.add_argument(
        '--psf_path', type=str, 
        default='/home/wv00017/MSOT_Diffusion/20250716_ImageNet_MSOT_Dataset/PSF.h5',
        help='path to the point spread function (PSF) of the imaging system (approximated as a 2D kernel)'
    )
    parser.add_argument(
        '--weights_dir', type=str, 
        default='/home/wv00017/python_BphP_MSOT_sim/invision_weights/',
        action='store',
        help='directory containing integration weights for combining sensor data'
    )
    parser.add_argument(
        '--mu_s_guess', type=float, default=None, action='store',
        help='Guess for scattering coefficient (m^-1), if None assume mu_s is known exactly'
    )
    parser.add_argument(
        '--mu_a_guess', type=float, default=30, action='store', 
        help='Guess for absorption coefficient (m^-1)'
    )
    parser.add_argument('--step_size', type=float, default=0.8, action='store', help='learning rate/step size')
    parser.add_argument('--epsilon', type=float, default=1e-8, action='store', help='small number to prevent division by zero')
    parser.add_argument('--image_name', type=str, help='name of image to reconstruct')
    parser.add_argument('--niter', type=int, help='Number of iterations', default=10)
    parser.add_argument('--sim_git_hash', type=str, default=None, action='store')
    parser.add_argument('--recon_iterations', type=int, default=5, action='store')
    parser.add_argument('--forward_model', choices=['invision', 'point'], default='invision', action='store')
    parser.add_argument('--inverse_model', choices=['invision', 'point'], default='invision', action='store')
    parser.add_argument('--delete_p0_3d', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-v', type=str, help='verbose level', default='INFO')
    parser.add_argument('--Gamma', type=float, default=1.0, action='store', help='Gruneisen parameter')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=False, help='plot results')
    parser.add_argument(
        '--noise_std', type=float, default=0.0, action='store',
        help='standard deviation Guassian noise to add to the sensor data'
    )
    parser.add_argument(
        '--bandpass_filter', default=False, action=argparse.BooleanOptionalAction,
        help='apply bandpass filter to sensor data'
    )
    parser.add_argument(
        '--resample_time_array', default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        '--resample_k_grid', default=0.2, action='store',
        help='factor to resample k-grid to reduce inverse crime, \
            data was simulated with a [748, 236, 748] size grid (excluding pml), \
            the domian size is kept as close to [0.082, 0.025871657754010697, 0.082] \
            as possible when resampling the grid'
    )
    parser.add_argument(
        '--tv_weight', type=float, default=0.01, action='store',
        help='weight to total variation regularisation of the estimated mu_a'
    )
    
    args = parser.parse_args()
    
    if args.v == 'INFO':
        logging.basicConfig(level=logging.INFO)
    elif args.v == 'DEBUG':
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.info(f'{args.v} not a recognised verbose level, using INFO instead')
    
    cfg = json.load(open(os.path.join(args.dataset, 'sim_config.json')))
    with h5py.File(os.path.join(args.dataset, 'dataset.h5'), 'r') as f:
        laser_energy = f['samples'][args.image_name]['wavelength_nm'][()] * 1e-9 # [m]
        data = {
            'H_recon_true' : f['samples'][args.image_name]['X'][()] * laser_energy, # [Pa J^-1] -> [Pa]
            'mu_a_true' : f['samples'][args.image_name]['mu_a'][()], # [m^-1]
            'Phi_true' : f['samples'][args.image_name]['Phi'][()] * laser_energy, # [m^-2] -> [J m^-2]
            'bg_mask' : f['samples'][args.image_name]['bg_mask'][()], # [bool]
            'wavelength_nm' : f['samples'][args.image_name]['wavelength_nm'][()], # [nm]
            'sensor_data' : f['samples'][args.image_name]['sensor_data'][()], # [Pa]
            'mu_s_true' : f['samples'][args.image_name]['mu_s'][()] # [m^-1]
        }
        
    cfg['image_LaserEnergy'] = laser_energy
    cfg = {**cfg, **vars(args)}
    logging.info(f'loaded simulation data from {args.dataset}')
    logging.info(f'simulation config: {cfg}')
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'cfg.json'), 'w') as f:
        json.dump(cfg, f, indent='\t')
    
    rng = np.random.default_rng(cfg['seed'])
    
    # load impulse response function
    irf = np.load(args.irf_path)
    with h5py.File(args.psf_path, 'r') as f:
        PSF = f['PSF'][()]
    
    # intialise bandpass filter
    if args.bandpass_filter:
        filter = make_filter(
            n_samples=cfg['Nt'], fs=1/cfg['dt'], irf=irf,
            hilbert=True, lp_filter=6.5e6, hp_filter=50e3, rise=0.2,
            n_filter=512, window='hann'
        )
        logging.info('bandpass filter initialised')
    
    H_recon_true = data['H_recon_true'].copy()
    mu_a_true = data['mu_a_true'].copy()
    Phi_true = data['Phi_true'].copy()
    bg_mask = data['bg_mask'].copy().astype(bool)
    
    
    # simulation is orientated at 90 deg anticlockwise
    H_recon_true = np.rot90(H_recon_true, k=1, axes=(-2,-1))
    mu_a_true = np.rot90(mu_a_true, k=1, axes=(-2,-1))
    bg_mask = np.rot90(bg_mask, k=1, axes=(-2,-1))
    
    if args.resample_k_grid:
        intitial_k_grid_size = cfg['kwave_grid_size']
        # resample k-grid to reduce inverse crime
        cfg['kwave_grid_size'] = [
            int(cfg['kwave_grid_size'][0] * args.resample_k_grid),
            int(cfg['kwave_grid_size'][1] * args.resample_k_grid),
            int(cfg['kwave_grid_size'][2] * args.resample_k_grid)
        ]
        cfg['crop_p0_3d_size'] = int(cfg['crop_p0_3d_size'] * args.resample_k_grid)
        cfg['crop_size'] = int(cfg['crop_size'] * args.resample_k_grid)
        cfg['mcx_grid_size'] = [
            int(cfg['mcx_grid_size'][0] * args.resample_k_grid),
            int(cfg['mcx_grid_size'][1] * args.resample_k_grid),
            int(cfg['mcx_grid_size'][2] * args.resample_k_grid)
        ]
        cfg['dx'] = cfg['mcx_domain_size'][0] / cfg['mcx_grid_size'][0]
        cfg['mcx_domain_size'][1] = cfg['dx'] * cfg['mcx_grid_size'][1]
        cfg['kwave_grid_size'][1] = cfg['dx'] * cfg['kwave_grid_size'][1]
        logging.info(f'resampled k-grid size: {cfg["kwave_grid_size"]}')
        # resample mu_a_true, Phi_true and bg_mask
        zoom_factor = cfg['kwave_grid_size'][0] / intitial_k_grid_size[0]
        mu_a_true = zoom(mu_a_true, zoom=zoom_factor, order=1)
        Phi_true = zoom(Phi_true, zoom=zoom_factor, order=1)
        bg_mask = zoom(bg_mask.astype(np.float32), zoom=zoom_factor, order=0).astype(bool)
        PSF = zoom(PSF, zoom=zoom_factor, order=1)
        
    H_recon_true = uf.square_centre_pad(H_recon_true, cfg['mcx_grid_size'][0])
    mu_a_true = uf.square_centre_pad(mu_a_true, cfg['mcx_grid_size'][0])
    Phi_true = uf.square_centre_pad(Phi_true, cfg['mcx_grid_size'][0])
    bg_mask = uf.square_centre_pad(bg_mask, cfg['mcx_grid_size'][0])
    
    # re-compute reconstruction with noise added
    start = timeit.default_timer()
    simulation = acoustic_inverse_simulation.kwave_inverse_adapter(
        cfg,
        transducer_model=cfg['inverse_model']
    )
    simulation.configure_simulation()
    logging.info(f'kwave inverse initialised in {timeit.default_timer() - start} seconds')
    gc.collect()
    start = timeit.default_timer()
    # apply convolution with the impulse response function
    (sensor_data, cfg) = add_noise(data['sensor_data'].astype(np.float32),
                                   cfg, rng, std=args.noise_std)
    sensor_data = convolve1d(sensor_data, irf, mode='nearest', axis=-1)
    # apply bandpass filter to the noisy sensor data
    if args.bandpass_filter:
        out = np.fft.ifft(
            np.fft.fft(sensor_data, axis=-1) * filter, axis=-1
        ).real.astype(np.float32)
    logging.info(f'noise added in {timeit.default_timer() - start} seconds')

    start = timeit.default_timer()
    H_recon_true = simulation.run_time_reversal(sensor_data)
    data['H_recon_true'] = np.rot90(H_recon_true.copy(), k=1, axes=(-2,-1))
    data['H_recon_true'] = uf.square_centre_crop(data['H_recon_true'].copy(), cfg['crop_size'])
    H_recon_true = np.rot90(H_recon_true, k=2, axes=(-2,-1))
    logging.info(f'time reversal run in {timeit.default_timer() - start} seconds')
    
    # define numerical phantom for forward model
    wavelengths_m = [float(args.image_name.split('_')[-1]) * 1e-9] # [m]
    phantom = fluence_correction_phantom(bg_mask, wavelengths_m=wavelengths_m)
    H2O = phantom.define_H2O()
    
    mu_a = args.mu_a_guess * bg_mask.astype(np.float32) # [m^-1] starting guess for absorption coefficient
    mu_a += H2O['mu_a'][0] * (~bg_mask).astype(np.float32) # [m^-1] H2O outside of segmentation mask
    if args.mu_s_guess:
        mu_s = args.mu_s_guess # [m^-1] assumed scattering coefficient
    else: # mu_s is known exactly
        mu_s = np.rot90(data['mu_s_true'].copy(), k=1, axes=(-2,-1))
        mu_s = zoom(mu_s, zoom=zoom_factor, order=1)
        mu_s = uf.square_centre_pad(mu_s, cfg['mcx_grid_size'][0])
        
    
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
                           np.rot90(H_recon_true.copy(), k=-1, axes=(-2,-1)), cfg['crop_size']
                       )]
        Phi_plots = [uf.square_centre_crop(Phi_true.copy(), cfg['crop_size'])]
        mu_a_line_profiles = [mu_a_plots[0][mu_a_plots[0].shape[0]//2,:],
                              mu_a_plots[1][mu_a_plots[1].shape[0]//2,:]]
        recon_line_profiles = [recon_plots[0][recon_plots[0].shape[0]//2,:]]
        grad_TV_plots = []
        grad_MSE_plots = []
    
    # intialise bandpass filter and acoustic-electric transfer function
    # with new sampling frequency and time step to reduce inverse crime
    irf_fft = np.abs(np.fft.fft(irf))
    if args.resample_time_array:
        irf = interp1d(
            np.arange(cfg['Nt'])/(cfg['Nt']*cfg['dt']), irf_fft, kind='linear', fill_value=0.0
        )(np.arange(1500)/(1500*30e-9))
        if args.bandpass_filter:
            filter = make_filter(
                n_samples=1500, fs=1/30e-9, irf=irf,
                hilbert=True, lp_filter=6.5e6, hp_filter=50e3, rise=0.2,
                n_filter=512, window='hann'
            )
            logging.info('bandpass filter re-initialised')
    
    # metrics are computed for each iteration
    metrics_mu_a = TestMetricCalculator()
    metrics_H_recon = TestMetricCalculator()
    metrics_mu_a(mu_a_true, mu_a, Y_mask=bg_mask)
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
        out *= cfg['image_LaserEnergy'] * 1e6
        Phi = out[:,(cfg['mcx_grid_size'][1]//2)-1,:].copy()
        Phi = np.rot90(Phi, k=2, axes=(-2,-1))
        
        # optical_and_acoustic
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
        """ # not needed when using the PSF as an approximation of the forward and adjoint operators
        # overwrite mcx simulation to save memory
        simulation = acoustic_forward_simulation.kwave_forward_adapter(
            cfg, 
            transducer_model=cfg['forward_model']
        )
        simulation.configure_simulation()
        if args.resample_time_array:
            # change time step size to reduce inverse crime
            simulation.kgrid.setTime(1500, 30e-9)
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
        if args.resample_time_array:
            simulation.kgrid.setTime(1500, 30e-9)
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
        H_recon_pred = simulation.run_time_reversal(out)
        H_recon_pred = np.rot90(H_recon_pred, k=2, axes=(-2,-1))
        logging.info(f'time reversal run in {timeit.default_timer() - start} seconds')

        #start = timeit.default_timer()
        #with h5py.File(cfg['save_dir']+'data.h5', 'r+') as f:
        #    f[h5_group].create_dataset(
        #        'H_recon_true',
        #        data=uf.square_centre_crop(tr, cfg['crop_size']),
        #        dtype=np.float32
        #    )
        #logging.info(f'p0_recon saved in {timeit.default_timer() - start} seconds')
        """
        H_recon_pred = padded_convolution(out, PSF) # [Pa]
        
        # update scheme for model absorption coefficient,
        # small number added to denominator to improve numerical stability
        logging.info(f'mu_a {mu_a.dtype} {mu_a.shape}')
        mu_a = mu_a.astype(np.float32)
        grad_MSE = grad_masked_MSE_loss(H_recon_true, H_recon_pred, PSF, Phi, bg_mask)
        grad_TV = masked_grad_TV(mu_a, bg_mask, eps=args.epsilon)
        grad = grad_MSE + args.tv_weight * grad_TV # [m^-1]
        mu_a += args.step_size * grad 
                
        #mu_a += args.step_size * (p0_recon - tr) / (cfg['gruneisen'] * Phi + args.epsilon) # depricated
        
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
        metrics_mu_a(mu_a_true, mu_a, Y_mask=bg_mask)
        metrics_H_recon(H_recon_true, H_recon_pred, Y_mask=bg_mask)
        
        if args.plot:
            mu_a_plots.append(uf.square_centre_crop(
                np.rot90(mu_a.copy(), k=-1, axes=(-2,-1)), cfg['crop_size']
            ))
            Phi_plots.append(uf.square_centre_crop(
                np.rot90(Phi.copy(), k=-1, axes=(-2,-1)), cfg['crop_size']
            ))
            mu_a_line_profiles.append(mu_a_plots[-1][mu_a_plots[-1].shape[0]//2,:])
            recon_plots.append(uf.square_centre_crop(
                np.rot90(H_recon_pred.copy(), k=-1, axes=(-2,-1)), cfg['crop_size']
            ))
            recon_line_profiles.append(recon_plots[-1][recon_plots[-1].shape[0]//2,:])
            grad_TV_plots.append(uf.square_centre_crop(
                np.rot90(grad_TV.copy(), k=-1, axes=(-2,-1)), cfg['crop_size']
            ))
            grad_MSE_plots.append(uf.square_centre_crop(
                np.rot90(grad_MSE.copy(), k=-1, axes=(-2,-1)), cfg['crop_size']
            ))
    
    if args.plot:
        with h5py.File(os.path.join(args.save_dir, 'results.h5'), 'w') as f:
            f.create_group('ground_truth')
            for key in list(data.keys()):
                f['ground_truth'].create_dataset(
                    key, data=data[key], dtype=np.float32
                )
            f.create_group('results')
            f['results'].create_dataset(
                'mu_a', data=np.asarray(mu_a_plots), dtype=np.float32
            )
            f['results'].create_dataset(
                'Phi', data=np.asarray(Phi_plots), dtype=np.float32
            )
            f['results'].create_dataset(
                'H_recon', data=np.asarray(recon_plots), dtype=np.float32
            )
            f['results'].create_dataset(
                'grad_TV', data=np.asarray(grad_TV_plots), dtype=np.float32
            )
            f['results'].create_dataset(
                'grad_MSE', data=np.asarray(grad_MSE_plots), dtype=np.float32
            )

    logging.info(metrics_mu_a.get_metrics())
    logging.info(metrics_H_recon.get_metrics())
    with open(os.path.join(args.save_dir, 'metrics.json'), 'w') as f:
        json.dump({'metrics_mu_a' : metrics_mu_a.get_metrics(),
                   'metrics_H_recon_true' : metrics_H_recon.get_metrics()}, f, indent='\t')
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
        images = [data['mu_a_true'], 
                    data['mu_s_true'], 
                    data['Phi_true'], 
                    data['mu_a_true']*data['Phi_true'],
                    data['H_recon_true']]
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