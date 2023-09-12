import numpy as np
from phantoms.Clara_experiment_phantom import Clara_experiment_phantom
import json
import h5py
import os
import geometry_func as gf
import BphP_func as bf
import utility_func as uf
import optical_simulation
import acoustic_forward_simulation
import acoustic_inverse_simulation
#import plot_func as pf
import timeit
import logging
import argparse
import gc


if __name__ == '__main__':

    '''
    ==================================Workflow==================================

    1. Define Simulation Geometry in Python
        - Domain size (x,y,z) in grid points and dx in m
        - Dictionary of all absorbers needed for simulation
        - For each include:
            ~ molar absorption at key wavelengths (epsilon_a) in m^2 mol^-1
            ~ scattering coefficient at key wavelengths (mu_s) in m^2 mol^-1
            ~ spaci
        - Define desired volumes and concentration of absorbers present
        - Calculate per volume based on linear combination of absorbers:
            ~ absorbtion coeff (mu_a) in m^-1
            ~ scattering coeff (mu_s) in m^-1

    2. Generate volume array to define simulation in MCX
        - save volume to binary file
        - save configuration to JSON file

    3. Execute MCX CUDA binary through Python

    4. Get normalised fluence map (array) from MCX output

    5. Adjust rsOAP (Pf and Pfr) concentrations accordingly
        - abjust absorbtion coeff (mu_a) of volume array
        - overwrite volume array binary file

    6. Calculate inital acoustic pressure (p0) of sample

    7. Propogate p0 using K-wave

    8. Reconstruct images from transducer array data

    9. Repeat 3-8 for every timestep/laser pulse of sample

    ============================================================================
    '''
    logging.basicConfig(level=logging.INFO)
    # TODO: use argparse to set mcx_bin_path and other arguements for cfg
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mcx_bin_path',
        type=str,
        default='/mcx/bin/mcx',
        action='store'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='',
        action='store'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='20230912_Clara_phantom_ReBphP_0p001/',
        action='store'
    )
    parser.add_argument('--npulses', type=int, default=16, action='store')
    
    args = parser.parse_args()
    
    save_dir = args.save_dir
    # path to MCX binary
    mcx_bin_path = args.mcx_bin_path   
    #mcx_bin_path = '/mcx/bin/mcx' # billy_docker
    mcx_bin_path = '/home/wv00017/mcx/bin/mcx' # Billy's workstation
    
    # check for a checkpointed simulation of the same name
    if (os.path.exists(args.save_dir+args.name+'config.json') 
        and os.path.exists(args.save_dir+args.name+'data.h5') 
        and os.path.exists(args.save_dir+args.name+'temp.h5')):
        
        with open(args.save_dir+args.name+'config.json', 'r') as f:
            cfg = json.load(f)
        logging.info(f'checkpoint config found {cfg}')
        
        phantom = Clara_experiment_phantom()
        H2O = phantom.define_water()
        ReBphP_PCM = phantom.define_ReBphP_PCM()
        # NOTE: ensure sample is contained within crop_size*crop_size of the centre
        # of the xz plane, all other voxels are background
        (volume, ReBphP_PCM_Pr_c, ReBphP_PCM_Pfr_c) = phantom.create_volume(cfg)
        
    else:
        # It is imperative that dx is small enough to support high enough 
        # frequencies and that [nx, ny, nz] have low prime factors i.e. 2, 3, 5
        c_0 = 1500.0 # speed of sound [m s^-1]
        mcx_domain_size = [0.082, 0.0205, 0.082] # [m]
        kwave_domain_size = [0.082, mcx_domain_size[1]/2, 0.082] # [m]
        pml_size = 10 # perfectly matched layer size in grid points
        [mcx_grid_size, dx] = gf.get_optical_grid_size(
            mcx_domain_size,
            c0_min=c_0,
            pml_size=pml_size,
            points_per_wavelength=1
        )
        mcx_domain_size = [mcx_grid_size[0]*dx,
                           mcx_grid_size[1]*dx,
                           mcx_grid_size[2]*dx]# [m], modify grid size for chosen dx
        [kwave_grid_size, dx] = gf.get_acoustic_grid_size(
            dx, 
            domain_size=kwave_domain_size,
            pml_size=pml_size
        )
        kwave_domain_size = [kwave_grid_size[0]*dx,
                             kwave_grid_size[1]*dx,
                             kwave_grid_size[2]*dx]# [m], modify grid size for chosen dx
        
        # configure simulation
        cfg = {
            'name' : save_dir+args.name,
            'seed' : None, # TODO: use to procedurally generate phantom
            'nsensors' : 256,
            'ncycles' : 1,
            'npulses' : args.npulses,
            'nphotons' : 1e8,
            'nsources' : 10,
            'wavelengths' : [680e-9, 770e-9],
            'mcx_domain_size' : mcx_domain_size,
            'kwave_domain_size' : kwave_domain_size,
            'mcx_grid_size': mcx_grid_size,
            'kwave_grid_size' : kwave_grid_size,
            'dx' : dx,
            'pml_size' : pml_size,
            'gruneisen' : 1.0,
            'c_0' : c_0,
            'alpha_coeff' : 0.01,
            'alpha_power' : 1.1,
            'recon_iterations' : 1, # time reversal iterations
            'crop_size' : 256, # pixel with of output images and ground truth
            'cycle' : 0, # for checkpointing
            'wavelength_index' : 0, # for checkpointing
            'pulse' : 0, # for checkpointing
            'stage' : 'optical', # for checkpointing (optical, acoustic, inverse)
        }
        
        logging.info(f'no checkpoint, creating config {cfg}')
        
        # Energy total delivered is wavelength dependant and normally disributed
        Emean = np.array([58.0050, 70.0727]) * 1e-3; # [J]
        Estd = np.array([1.8186, 0.7537]) * 1e-3; # [J]
        # index laser energy as [cycle, wavelength, pulse]
        cfg['LaserEnergy'] = np.zeros(
            (cfg['ncycles'], len(cfg['wavelengths']), cfg['npulses']), 
            dtype=np.float32
        )
        for i in range(len(cfg['wavelengths'])):
            cfg['LaserEnergy'][:,i,:] = (
                np.random.normal(
                    loc=Emean[i],
                    scale=Estd[i],
                    size=(cfg['ncycles'], 1, cfg['npulses']),
                )
            )
        cfg['LaserEnergy'] = cfg['LaserEnergy'].tolist()
        
        # save configuration to JSON file
        uf.create_dir(cfg['name'])
        with open(cfg['name']+'config.json', 'w') as f:
            json.dump(cfg, f, indent='\t')
            
        phantom = Clara_experiment_phantom()
        H2O = phantom.define_water()
        ReBphP_PCM = phantom.define_ReBphP_PCM()
        # NOTE: ensure sample is contained within crop_size*crop_size of the centre
        # of the xz plane, all other voxels are background
        (volume, ReBphP_PCM_Pr_c, ReBphP_PCM_Pfr_c) = phantom.create_volume(cfg)
            
        # save 2D slice of the volume to HDF5 file
        with h5py.File(cfg['name']+'data.h5', 'w') as f:
            f.create_dataset(
                'background_mua_mus', 
                data=uf.square_centre_crop(volume[:,:,:,cfg['mcx_grid_size'][1]//2,:],
                                        cfg['crop_size'])
            )
            # protein concentration ground truth is the most important besides the
            # reconstructed images of the pressure
            f.create_dataset(
                'ReBphP_PCM_c_tot', 
                data=(
                    uf.square_centre_crop(
                        ReBphP_PCM_Pr_c[:,cfg['mcx_grid_size'][1]//2,:],
                        cfg['crop_size']
                    ) +
                    uf.square_centre_crop(
                        ReBphP_PCM_Pfr_c[:,cfg['mcx_grid_size'][1]//2,:],
                        cfg['crop_size']
                    )
                )
            )
            # allocate storage for the fluence, initial and reconstructed pressure
            # index as data['arg'][cycle, wavelength, pulse, x, z]
            logging.info('allocating storage for data.h5')
            for arg in ['Phi', 'p0', 'p0_recon', 'ReBphP_PCM_Pr_c', 'ReBphP_PCM_Pfr_c']:
                logging.info(arg)
                f.create_dataset(
                    arg,
                    shape=(
                        cfg['ncycles'],
                        len(cfg['wavelengths']),
                        cfg['npulses'],
                        cfg['crop_size'],
                        cfg['crop_size']
                    ),
                        dtype=np.float32
                )
                        
        with h5py.File(cfg['name']+'temp.h5', 'w') as f:
            # p0 will be saved in 3D for the acoustic simulation before finally
            # being deleted
            # 1 cycle * 2 wavelengths * 16 pulses * 512 * (1024**2) * 32 bits = 64 GB
            # the entire sample should be contained within crop_size*crop_size in 
            # the centre of the xz plane, all other voxels are background equal zero
            # 1 cycle * 2 wavelengths * 16 pulses * 512 * (256**2) * 32 bits = 4 GB
            # hdf5 compression also helps reduce size
            logging.info('allocating storage for p0_3d temp.h5')
            f.create_dataset(
                'p0_3D',
                shape=(
                    cfg['ncycles'],
                    len(cfg['wavelengths']),
                    cfg['npulses'],
                    cfg['crop_size'],
                    cfg['kwave_grid_size'][1],
                    cfg['crop_size']
                ), 
                dtype=np.float32
            )
 
    # optical simulation
    simulation = optical_simulation.MCX_adapter(cfg)
    
    gc.collect()
    
    if cfg['stage'] == 'optical':
        for cycle in range(cfg['cycle'], cfg['ncycles']):
            cfg['cycle'] = cycle
            for wavelength_index in range(cfg['wavelength_index'], len(cfg['wavelengths'])):
                cfg['wavelength_index'] = wavelength_index
                for pulse in range(cfg['pulse'], cfg['npulses']):
                    cfg['pulse'] = pulse
                    with open(cfg['name']+'config.json', 'w') as f:
                        json.dump(cfg, f, indent='\t')
        
                    logging.info(f'mcx, cycle: {cycle+1}, wavelength_index: {wavelength_index+1}, pulse: {pulse+1}')
                    
                    start = timeit.default_timer()
                    # out can be energy absorbed, fluence, pressure, sensor data
                    # or recontructed pressure, the variable is overwritten
                    # multiple times to save memory
                    out = simulation.run_mcx(
                        mcx_bin_path,
                        volume[wavelength_index], 
                        ReBphP_PCM_Pr_c,
                        ReBphP_PCM_Pfr_c,
                        ReBphP_PCM,
                        wavelength_index
                    )
                    logging.info(f'mcx run in {timeit.default_timer() - start} seconds')
                    
                    # convert from [mm^-2] -> [J m^-2]
                    start = timeit.default_timer()
                    out *= cfg['LaserEnergy'][cycle][wavelength_index][pulse] * 1e6
                    
                    # save fluence, Pr and Pfr concentrations (additional ground truth) to data HDF5 file
                    with h5py.File(cfg['name']+'data.h5', 'r+') as f:
                        f['Phi'][cycle,wavelength_index,pulse] = uf.square_centre_crop(
                            out[:,cfg['mcx_grid_size'][1]//2,:], cfg['crop_size'])
                        f['ReBphP_PCM_Pr_c'][cycle,wavelength_index,pulse] = uf.square_centre_crop(
                            ReBphP_PCM_Pr_c[:,cfg['mcx_grid_size'][1]//2,:], cfg['crop_size'])
                        f['ReBphP_PCM_Pfr_c'][cycle,wavelength_index,pulse] = uf.square_centre_crop(
                            ReBphP_PCM_Pfr_c[:,cfg['mcx_grid_size'][1]//2,:], cfg['crop_size'])
                    logging.info(f'fluence and protein concentrations saved in {timeit.default_timer() - start} seconds')
                    
                    start = timeit.default_timer()
                    # convert fluence to initial pressure [J m^-2] -> [J m^-3]
                    out *= cfg['gruneisen'] * (volume[wavelength_index, 0] +
                            ReBphP_PCM_Pr_c * ReBphP_PCM['Pr']['epsilon_a'][wavelength_index] + 
                            ReBphP_PCM_Pfr_c * ReBphP_PCM['Pfr']['epsilon_a'][wavelength_index])
                    
                    # save 3D p0 to temp.h5
                    with h5py.File(cfg['name']+'temp.h5', 'r+') as f:
                        f['p0_3D'][cycle,wavelength_index,pulse] =  uf.crop_p0_3D(
                            out,
                            [cfg['crop_size'], cfg['kwave_grid_size'][1], cfg['crop_size']]
                        )
                    with h5py.File(cfg['name']+'data.h5', 'r+') as f:
                        f['p0'][cycle,wavelength_index,pulse] =  uf.square_centre_crop(
                            out[:,cfg['mcx_grid_size'][1]//2,:], cfg['crop_size']
                        )
                    logging.info(f'pressure saved in {timeit.default_timer() - start} seconds')    
                                        
                    start = timeit.default_timer()
                    # compute photoisomerisation
                    ReBphP_PCM_Pr_c, ReBphP_PCM_Pfr_c = bf.switch_BphP(
                        ReBphP_PCM['Pr'], 
                        ReBphP_PCM['Pfr'],
                        ReBphP_PCM_Pr_c,
                        ReBphP_PCM_Pfr_c,
                        out,
                        cfg['wavelengths'],
                        wavelength_index
                    )
                    
                    logging.info(f'photoisomerisation computed in {timeit.default_timer() - start} seconds')
            
    gc.collect()
    
    start = timeit.default_timer()
    # deleted mcx input and out files, they are not needed anymore
    simulation.delete_temporary_files()
    # overwrite mcx simulation to save memory
    simulation = acoustic_forward_simulation.kwave_forward_adapter(cfg)
    # k-wave automatically determines dt and Nt, update cfg
    cfg = simulation.cfg
    
    # save updated cfg to JSON file
    with open(cfg['name']+'config.json', 'w') as f:
        json.dump(cfg, f, indent='\t')
    
    
    simulation.configure_simulation()
    simulation.create_point_sensor_array()
    logging.info(f'kwave forward initialised in {timeit.default_timer() - start} seconds')
    
    if cfg['stage'] == 'optical':
        start = timeit.default_timer()
        # create dataset for sensor data based on k-wave Nt
        with h5py.File(cfg['name']+'data.h5', 'r+') as f:
            f.create_dataset(
                'sensor_data',
                shape=(   
                    cfg['ncycles'],
                    len(cfg['wavelengths']),
                    cfg['npulses'],
                    cfg['nsensors'],
                    cfg['Nt']
                ),
                dtype=np.float32# dtype=np.float16
            )
        logging.info(f'sensor data dataset created in {timeit.default_timer() - start} seconds')
    
    if cfg['stage'] == 'optical':
        logging.info('optical stage complete')
        cfg['stage'] = 'acoustic'; cfg['cycle'] = 0; cfg['wavelength_index'] = 0; cfg['pulse'] = 0
    
    gc.collect()
    
    # acoustic forward simulation
    if cfg['stage'] == 'acoustic':
        for cycle in range(cfg['cycle'], cfg['ncycles']):
            cfg['cycle'] = cycle
            for wavelength_index in range(cfg['wavelength_index'], len(cfg['wavelengths'])):
                cfg['wavelength_index'] = wavelength_index
                for pulse in range(cfg['pulse'], cfg['npulses']):
                    cfg['pulse'] = pulse
                    with open(cfg['name']+'config.json', 'w') as f:
                       json.dump(cfg, f, indent='\t')
                    
                    logging.info(f'k-wave forward, cycle: {cycle+1}, wavelength_index: {wavelength_index+1}, pulse: {pulse+1}')
                    
                    start = timeit.default_timer()
                    with h5py.File(cfg['name']+'temp.h5', 'r') as f:
                        out = uf.pad_p0_3D(
                            f['p0_3D'][cycle,wavelength_index,pulse],
                            cfg['kwave_grid_size'][0]
                        )
                    logging.info(f'p0 loaded in {timeit.default_timer() - start} seconds')
                    
                    start = timeit.default_timer()
                    # run also saves the sensor data to data.h5 as float16
                    out = simulation.run_kwave_forward(out)
                    logging.info(f'kwave forward run in {timeit.default_timer() - start} seconds')
                    
                    start = timeit.default_timer()
                    with h5py.File(cfg['name']+'data.h5', 'r+') as f:
                        f['sensor_data'][cycle,wavelength_index,pulse] = out
                    logging.info(f'sensor data saved in {timeit.default_timer() - start} seconds')
        
    if cfg['stage'] == 'acoustic':
        logging.info('acoustic stage complete')
        cfg['stage'] = 'inverse'; cfg['cycle'] = 0; cfg['wavelength_index'] = 0; cfg['pulse'] = 0
    
    # delete temp p0_3D dataset
    start = timeit.default_timer()
    os.remove(cfg['name']+'temp.h5')
    logging.info(f'temp.h5 (p0_3D) deleted in {timeit.default_timer() - start} seconds')
    
    start = timeit.default_timer()
    simulation = acoustic_inverse_simulation.kwave_inverse_adapter(cfg)
    simulation.configure_simulation()
    simulation.create_point_source_array()
    logging.info(f'kwave inverse initialised in {timeit.default_timer() - start} seconds')
    
    gc.collect()
    
    # acoustic reconstruction
    if cfg['stage'] == 'inverse':
        for cycle in range(cfg['cycle'], cfg['ncycles']):
            cfg['cycle'] = cycle
            for wavelength_index in range(cfg['wavelength_index'], len(cfg['wavelengths'])):
                cfg['wavelength_index'] = wavelength_index
                for pulse in range(cfg['pulse'], cfg['npulses']):
                    cfg['pulse'] = pulse
                    with open(cfg['name']+'/config.json', 'w') as f:
                        json.dump(cfg, f, indent='\t')
                    
                    logging.info(f'time reversal, cycle: {cycle+1}, wavelength_index: {wavelength_index+1}, pulse: {pulse+1}')
                    
                    start = timeit.default_timer()
                    with h5py.File(cfg['name']+'data.h5', 'r') as f:
                        out = f['sensor_data'][cycle,wavelength_index,pulse]
                    logging.info(f'sensor data loaded in {timeit.default_timer() - start} seconds')
                    
                    start = timeit.default_timer()
                    out = simulation.run_time_reversal(out)
                    logging.info(f'time reversal run in {timeit.default_timer() - start} seconds')

                    start = timeit.default_timer()
                    with h5py.File(cfg['name']+'data.h5', 'r+') as f:
                        f['p0_recon'][cycle,wavelength_index,pulse] = uf.square_centre_crop(out, cfg['crop_size'])
                    logging.info(f'p0_recon saved in {timeit.default_timer() - start} seconds')
        
    
