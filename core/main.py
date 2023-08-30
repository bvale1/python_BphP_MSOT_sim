import numpy as np
from core.phantoms.Clara_experiment_phantom import Clara_experiment_phantom
import json
import h5py
import geometry_func as gf
import BphP_func as bf
import utility_func as uf
import optical_simulation
#import acoustic_simulation
import plot_func as pf

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

    # path to MCX binary
    mcx_bin_path = '/home/wv00017/mcx/bin/mcx'
    
    c_0 = 1500.0 # speed of sound [m s^-1]
    domain_size = [0.082, 0.041, 0.082] # [m]
    pml_size = 10 # perfectly matched layer size in grid points
    grid_size, dx = gf.get_optical_grid_size(
        domain_size,
        c0_min=c_0,
        pml_size=pml_size,
        points_per_wavelength=1
    )
    domain_size = [grid_size[0]*dx,
                   grid_size[1]*dx,
                   grid_size[2]*dx]# [m], modify grid size for chosen dx
    
    # configure simulation
    cfg = {
        'name' : '202307020_python_Clara_phantom_ReBphP_0p001',
        'seed' : None,
        'ncycles' : 1,
        'npulses' : 1,
        'nphotons' : 1e8,
        'nsources' : 10,
        'wavelengths' : [680e-9, 770e-9],
        'domain_size' : domain_size,
        'grid_size': grid_size,
        'dx' : dx,
        'pml_size' : pml_size,
        'gruneisen' : 1.0,
        'c_0' : c_0,
        'alpha_coeff' : 0.01,
        'alpha_power' : 1.1,
    }
    
    print('main config ', cfg)
    
    phantom = Clara_experiment_phantom()
    H2O = phantom.define_water()
    ReBphP_PCM = phantom.define_ReBphP_PCM()
    (volume, ReBphP_PCM_Pr_c, ReBphP_PCM_Pfr_c) = phantom.create_volume(cfg)
    
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
    with open(cfg['name']+'/config.json', 'w') as outfile:
        json.dump(cfg, outfile)
        
    # save 2D slice of the volume to HDF5 file
    with h5py.File(cfg['name']+'/data.h5', 'w') as f:
        f.create_dataset(
            'background_volume', 
            data=volume[:,:,:,cfg['grid_size'][1]//2,:]
        )
        f.create_dataset(
            'ReBphP_PCM_c_tot', 
            data=ReBphP_PCM_Pr_c[:,cfg['grid_size'][1]//2,:] +
                 ReBphP_PCM_Pfr_c[:,cfg['grid_size'][1]//2,:]
        )     
        # allocate storage for the fluence, initial and reconstructed pressure
        # index as data['arg'][cycle, wavelength, pulse, x, z]
        for arg in ['fluence', 'p0', 'p0_recon', 'ReBphP_PCM_Pr_c', 'ReBphP_PCM_Pfr_c']:
            f.create_dataset(
                arg,
                np.zeros(
                    (
                        cfg['ncycles'],
                        len(cfg['wavelengths']),
                        cfg['npulses'],
                        cfg['grid_size'][0],
                        cfg['grid_size'][2]
                    ),
                    dtype=np.float32
                )
                
            )

    # optical simulation
    mcx_simulation = optical_simulation.MCX_adapter(cfg)
    
    for cycle in range(cfg['ncycles']):
        for wavelength_index in range(len(cfg['wavelengths'])):
            for pulse in range(cfg['npulses']):
                
                print('cycle: ', cycle+1, ', pulse: ', pulse+1)
                
                mcx_out = mcx_simulation.run_mcx(
                    mcx_bin_path,
                    volume[wavelength_index], 
                    ReBphP_PCM_Pr_c,
                    ReBphP_PCM_Pfr_c,
                    ReBphP_PCM,
                    wavelength_index
                )
                
                # convert from [voxel^-1] to [J voxel^-1]
                mcx_out *= cfg['LaserEnergy'][cycle, wavelength_index, pulse]
                
                # Convert from [J voxel^-1] to [J m^-3]
                mcx_out /= cfg['dx']**3 
                
                # divide by absorption coefficient to get fluence
                mcx_out /= volume[wavelength_index, 0] # [J m^-3] -> [J m^-2]
                
                # zero nan values (since all background voxels have zero absorption)
                mcx_out = np.nan_to_num(mcx_out, nan=0.0, posinf=0.0, neginf=0.0)
                
                # save fluence, Pr and Pfr concentrations to HDF5 file
                with h5py.File(cfg['name']+'/data.h5', 'r+') as f:
                    f['fluence'][cycle,wavelength_index,pulse,:,:] = mcx_out[:,cfg['grid_size'][1]//2,:]
                    f['ReBphP_PCM_Pr_c'][cycle,wavelength_index,pulse,:,:] = ReBphP_PCM_Pr_c[:,cfg['grid_size'][1]//2,:]
                    f['ReBphP_PCM_Pfr_c'][cycle,wavelength_index,pulse,:,:] = ReBphP_PCM_Pfr_c[:cfg['grid_size'][1]//2,:]
                
                # compute photoisomerisation
                bf.switch_BphP(
                    ReBphP_PCM['Pr'], 
                    ReBphP_PCM['Pfr'],
                    ReBphP_PCM_Pr_c,
                    ReBphP_PCM_Pfr_c,
                    mcx_out,
                    cfg['wavelengths'],
                    wavelength_index
                )
                
    mcx_simulation.delete_temporary_files()
    
    '''
    # acoustic forward simulation
    for cycle in range(cfg['ncycles']):
        for wavelength_index in range(len(cfg['wavelengths'])):
            for pulse in range(cfg['npulses']):
    '''
    # acoustic reconstruction