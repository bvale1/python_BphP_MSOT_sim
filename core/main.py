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
        'nsensors' : 256,
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
        'recon_iterations' : 1
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
    with open(cfg['name']+'/config.json', 'w') as f:
        json.dump(cfg, f)
        
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
        print('allocating storage for data.h5')
        for arg in ['Phi', 'p0', 'p0_recon', 'ReBphP_PCM_Pr_c', 'ReBphP_PCM_Pfr_c']:
            print(arg)
            f.create_dataset(
                arg,
                shape=(
                    cfg['ncycles'],
                    len(cfg['wavelengths']),
                    cfg['npulses'],
                    cfg['grid_size'][0],
                    cfg['grid_size'][2]
                ),
                    dtype=np.float32
            )
                        
    with h5py.File(cfg['name']+'/temp.h5', 'w') as f:
        # p0 will be saved in 3D for the acoustic simulation before finally
        # being condensed to 2D slices so the dataset isn't too large
        # 1 cycle * 2 wavelengths * 16 pules * 512 * (1024**2) * 32 bits = 64 GB
        f.create_dataset(
            'p0_3D',
            shape=(
                cfg['ncycles'],
                len(cfg['wavelengths']),
                cfg['npulses'],
                cfg['grid_size'][0],
                cfg['grid_size'][1],
                cfg['grid_size'][2]
            ), 
            dtype=np.float32
        )

    # optical simulation
    simulation = optical_simulation.MCX_adapter(cfg)
    
    for cycle in range(cfg['ncycles']):
        for wavelength_index in range(len(cfg['wavelengths'])):
            for pulse in range(cfg['npulses']):
                
                print('cycle: ', cycle+1, 'wavelength_index', wavelength_index+1, ', pulse: ', pulse+1)
                
                # out can be energy absorbed, fluence, pressure, sensor data
                # or recontructed pressure, each is overwritten when no longer
                # needed to save space
                out = simulation.run_mcx(
                    mcx_bin_path,
                    volume[wavelength_index], 
                    ReBphP_PCM_Pr_c,
                    ReBphP_PCM_Pfr_c,
                    ReBphP_PCM,
                    wavelength_index
                )
                
                # convert from [voxel^-1] to [J voxel^-1]
                out *= cfg['LaserEnergy'][cycle][wavelength_index][pulse]
                
                # save 3D p0 to temp.h5
                with h5py.File(cfg['name']+'/temp.h5', 'r+') as f:
                    f['p0_3D'][cycle,wavelength_index,pulse] = cfg['gruneisen'] * out
                
                # Convert from [J voxel^-1] to [J m^-3]
                out /= cfg['dx']**3 
                
                # divide by absorption coefficient to get fluence
                out /= volume[wavelength_index, 0] # [J m^-3] -> [J m^-2]
                
                # zero nan values (since all background voxels have zero absorption)
                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
                
                # save fluence, Pr and Pfr concentrations to data HDF5 file
                with h5py.File(cfg['name']+'/data.h5', 'r+') as f:
                    f['Phi'][cycle,wavelength_index,pulse] = out[:,cfg['grid_size'][1]//2,:]
                    f['ReBphP_PCM_Pr_c'][cycle,wavelength_index,pulse] = ReBphP_PCM_Pr_c[:,cfg['grid_size'][1]//2,:]
                    f['ReBphP_PCM_Pfr_c'][cycle,wavelength_index,pulse] = ReBphP_PCM_Pfr_c[:,cfg['grid_size'][1]//2,:]
                    
                # compute photoisomerisation
                bf.switch_BphP(
                    ReBphP_PCM['Pr'], 
                    ReBphP_PCM['Pfr'],
                    ReBphP_PCM_Pr_c,
                    ReBphP_PCM_Pfr_c,
                    out,
                    cfg['wavelengths'],
                    wavelength_index
                )
                
    # deleted mcx input and out files, they are not needed anymore
    simulation.delete_temporary_files()
    # overwrite mcx simulation to save memory
    simulation = acoustic_forward_simulation.kwave_forward_adapter(cfg)
    # k-wave automatically determines dt and Nt, update cfg
    cfg = simulation.cfg
    
    # save updated cfg to JSON file
    with open(cfg['name']+'/config.json', 'w') as f:
        json.dump(cfg, f)
    # create dataset for sensor data based on k-wave Nt
    with h5py.File(cfg['name']+'/data.h5', 'r+') as f:
        f.create_dataset(
            'sensor_data',
            shape=(   
                cfg['ncycles'],
                len(cfg['wavelengths']),
                cfg['npulses'],
                cfg['nsensors'],
                cfg['Nt']
            ),
            dtype=np.float16
        )            
    
    simulation.configure_simulation()
    simulation.create_point_sensor_array()
    
    # acoustic forward simulation
    for cycle in range(cfg['ncycles']):
        for wavelength_index in range(len(cfg['wavelengths'])):
            for pulse in range(cfg['npulses']):
                
                print('cycle: ', cycle+1, 'wavelength_index', wavelength_index+1, ', pulse: ', pulse+1)
                
                with h5py.File(cfg['name']+'/temp.h5', 'r') as f:
                    out = f['p0_3D'][cycle,wavelength_index,pulse]
                
                # run also saves the sensor data to data.h5 as float16
                out = simulation.run_kwave_forward(out)
    
    
    
    # delete temp p0_3D dataset
    os.remove(cfg['name']+'/temp.h5')
    
    simulation = acoustic_inverse_simulation.kwave_inverse_adapter(cfg)
    simulation.configure_simulation()
    simulation.create_point_source_array()
    
    # acoustic reconstruction
    for cycle in range(cfg['ncycles']):
        for wavelength_index in range(len(cfg['wavelengths'])):
            for pulse in range(cfg['npulses']):
                
                print('time reversal, cycle: ', cycle+1, 'wavelength_index', wavelength_index+1, ', pulse: ', pulse+1)
                
                with h5py.File(cfg['name']+'/data.h5', 'r') as f:
                    out = f['sensor_data'][cycle,wavelength_index,pulse]
                    
                out = simulation.run_time_reversal(out)

                with h5py.File(cfg['name']+'/data.h5', 'r+') as f:
                    f['p0_recon'][cycle,wavelength_index,pulse] = out
    
    
    