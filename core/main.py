import numpy as np
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

    6. Calculate inital acoustic pressure (P_0) of sample

    7. Propogate P_0 using K-wave

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
    
    # properties of the phytochrome
    ReBphP_PCM = {
        'Pr' : {
            'epsilon_a': [0.8e4, 0.05e4],
            'eta' : [0.01, 0.0]
            },
        'Pfr' : {
            'epsilon_a': [0.6e4, 0.8e4],
            'eta' : [0.015, 0.0]
        }   
    }
    cfg['ReBphP_PCM'] = ReBphP_PCM
    
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
    
    # temperature coefficients https://pubs.acs.org/doi/10.1021/jp010093m
    temp = 34; # [celcius]
    # absorption 380nm-700nm https://opg.optica.org/ao/viewmedia.cfm?uri=ao-38-7-1216&seq=0
    # absorption and scattering 300nm-800nm Optical properties of pure water Hendrik Buiteveld
    # https://www.spiedigitallibrary.org/conference-proceedings-of-spie/2258/0000/Optical-properties-of-pure-water/10.1117/12.190060.full
    water = {
        'mu_a' : [ 
            0.4318 + (temp - 20) * (-5e-5), # [m^-1]
            2.7542 + (temp - 20) * (-152e-5) # [m^-1]
        ],
        'mu_s' : [0.0006, 0.0003], # [m^-1],
        'n' : [1.33, 1.33],
        'g' : [0.9, 0.9]
    }
    
    
    # initialise proteins, Pr to Pfr ratio is the steady state
    Pr_frac, Pfr_frac = bf.steady_state_BphP(
        ReBphP_PCM['Pr'],
        ReBphP_PCM['Pfr'],
        wavelength_idx=0
    )
    ReBphP_PCM_Pr_c = 0.001 * gf.cylinder_mask(
        cfg['dx'],
        cfg['grid_size'],
        1.5e-3,
        [(cfg['domain_size'][0]/2)-4e-3, 0.0, (cfg['domain_size'][2]/2)-2e-3]
    )
    ReBphP_PCM_Pfr_c = Pfr_frac * ReBphP_PCM_Pr_c
    ReBphP_PCM_Pr_c = Pr_frac * ReBphP_PCM_Pr_c
    
    # define volume scattering and absorption coefficients
    # index as [1, ..., lambda]->[mu_a, mu_s]->[x]->[y]->[z]
    volume = np.zeros(
        (
            len(cfg['wavelengths']),
            2, 
            cfg['grid_size'][0], 
            cfg['grid_size'][1], 
            cfg['grid_size'][2]            
        ), dtype=np.float32
    )
    for i in range(len(cfg['wavelengths'])):
        volume[i,0,:,:,:] = water['mu_a'][i] * gf.cylinder_mask(
            cfg['dx'],
            cfg['grid_size'],
            0.01,
            [(cfg['domain_size'][0]/2), 0.0, (cfg['domain_size'][2]/2)]
        )
        volume[i,1,:,:,:] = water['mu_s'][i] * gf.cylinder_mask(
            cfg['dx'],
            cfg['grid_size'],
            0.01,
            [(cfg['domain_size'][0]/2), 0.0, (cfg['domain_size'][2]/2)] 
        )
        
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

    '''
    # use this to get the fluence resulting from the MCX source with no sample
    volume = volume = np.zeros(
        (
            len(cfg['wavelengths']),
            2, 
            cfg['grid_size'][0], 
            cfg['grid_size'][1], 
            cfg['grid_size'][2]            
        ), dtype=np.float32
    )
    for i in range(len(cfg['wavelengths'])):
        volume[i,0,:,:,:] = water['mu_a'][i]
        volume[i,1,:,:,:] = water['mu_s'][i]
    '''

    mcx_simulation = optical_simulation.MCX_adapter(cfg)

    for cycle in range(cfg['ncycles']):
        
        for wavelength_index in range(len(cfg['wavelengths'])):
            
            for pulse in range(cfg['npulses']):
                
                print('cycle: ', cycle+1, ', pulse: ', pulse+1)
                energy_absorbed = mcx_simulation.run_mcx(
                    mcx_bin_path,
                    volume[wavelength_index], 
                    ReBphP_PCM_Pr_c,
                    ReBphP_PCM_Pfr_c,
                    ReBphP_PCM,
                    wavelength_index
                )
                
                # kwave simulation needs to be initialised for each pulse
                # then removed otherwise it will run out of memory
                #p_0 = cfg['gruneisen'] * Phi * 

    mcx_simulation.delete_temporary_files()

    with h5py.File(cfg['name']+'/data.h5', 'a') as f:
        f.create_dataset('norm_fluence', data=energy_absorbed)
    