from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.utils.kwave_array import kWaveArray
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DG
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DG
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.ksource import kSource
from kwave.utils.mapgen import make_cart_circle, make_circle
from kwave.utils.interp import interp_cart_data
from kwave.utils.conversion import cart2grid
import utility_func as uf
import numpy as np
import h5py
import timeit


# k-wave simulation may be run as a function, or as a class
# function sets up the simulation i.e. transducer array, grid, medium, etc...
# every time it is called making it slower but more memory efficient

'''
====================================Workflow====================================

1. define kwave simulation grid object (KWaveGrid)

2. define transducer object (KWaveArray)

3. run simulation simulation on GPU with CUDA binaries (KWaveFirstOrder3DG)

4. save senor data to HDF5 file

5. TODO: add bandlimited impulse response (BLI) and noise

6. run time reversal reconstruction (KWaveFirstOrder3DG)

7. TODO: implement iterative time reversal reconstruction (ITR)

8. Save reconstruction to HDF5 file

================================================================================
'''

def kwave_MSOT_sim(cfg, p0):
    kgrid = kWaveGrid(
        cfg['grid_size'],
        [cfg['dx'], cfg['dx'], cfg['dx']],
    )
    # let k-wave automatically determine time step and number of time steps
    kgrid.makeTime(cfg['c_0'])
    cfg['dt'] = kgrid.dt
    cfg['Nt'] = kgrid.Nt
        
    # TODO: use frequency power law to add attenuation
    # Acoustical Characteristics of Biological Media, Jeffrey C. Bamber, 1997
    # k-wave uses Neper radian units (Np rad s^-1 m^-1)
    # 1 Np = 8.685889638 dB = 20.0 log_10(e) dB
    # 1 dB = 0.115129254 Np = 0.05 ln(10) Np
    medium = kWaveMedium(
        sound_speed=cfg['c_0'],
        absorbing=False
    )

    # transducer array geometry provided by 
    # Janek Gröhl https://github.com/jgroehl
    # and based of the IThera MSOT inVision 256-TF
    # https://ithera-medical.com/products/preclinical-research/invision/
    print('creating transducer array...')
    start = timeit.default_timer()
    number_detector_elements = 256
    radius_mm = 40.5
    radius_2_mm = 37
    pitch_mm = 0.74
    element_size = 0.635
    element_length = 15
    num_interpolation_points_x = 7
    num_interpolation_points_y = 101
    x_range = (
        np.linspace(
            -num_interpolation_points_x / 2,
            num_interpolation_points_x / 2,
            num_interpolation_points_x
        )
    )
    y_range = (
        np.linspace(
            -num_interpolation_points_y / 2,
            num_interpolation_points_y / 2,
            num_interpolation_points_y
        )
    )

    pitch_angle = pitch_mm / radius_mm
    x_increment_angle = element_size / radius_mm / num_interpolation_points_x
    y_increment_angle = element_length / radius_2_mm / num_interpolation_points_y
    det_elements = np.linspace(
        number_detector_elements / 2,
        -number_detector_elements / 2,
        number_detector_elements
    )

    # initializse transducer array object
    karray = kWaveArray(bli_tolerance=0.05, upsampling_rate=10)
    
    theta = np.pi / 2
    rotate = uf.Ry3D(np.pi / 2)
    # add each detector element to the transducer array
    for det_idx in range(number_detector_elements):
        # print progress percentage every 10 elements
        #if det_idx % 10 == 0:
        #    print(f'{100*det_idx/number_detector_elements} %')
        detector_positions = np.zeros(
            (3, num_interpolation_points_x * num_interpolation_points_y)
        )
        for x_idx in range(len(x_range)):
            
            x_inc = x_range[x_idx] * x_increment_angle
            
            for y_idx in range(len(y_range)):
                
                y_inc = y_range[y_idx] * y_increment_angle
                idx = (x_idx-1) * num_interpolation_points_y + y_idx
                
                detector_positions[0, idx] = np.sin(
                    np.pi/2 + pitch_angle * det_elements[det_idx] + x_inc
                ) * np.sin(theta + y_inc) * (radius_mm - 0.5 * element_size)
                
                detector_positions[1, idx] = np.cos(
                    theta + y_inc
                ) * (radius_mm - 0.5 * element_size)
                
                detector_positions[2, idx] = np.cos(
                    np.pi/2 + pitch_angle * det_elements[det_idx] + x_inc
                ) * np.sin(theta + y_inc) * (radius_mm - 0.5 * element_size)
                
        detector_positions = np.matmul(rotate, detector_positions)
        karray.add_custom_element(
            detector_positions * 1e-3, 
            element_size * element_length * 1e-3, 
            2, # 2D surface in 3D space
            str(det_idx)
        )
    print(f'transducer array created in {timeit.default_timer() - start} seconds')
    
    start = timeit.default_timer()
    print('creating sensor mask...')
    sensor_mask = karray.get_array_binary_mask(kgrid)
    print(f'sensor mask created in {timeit.default_timer() - start} seconds')
    
    sensor = kSensor(sensor_mask) # records pressure by default

    # configure simulation
    simulation_options = SimulationOptions(
        pml_inside=False,
        pml_size=cfg['pml_size'],
        data_cast='single',
        save_to_disk=True,
    )
    
    execution_options = SimulationExecutionOptions(
        is_gpu_simulation=True
    )
    
    # define source
    source = kSource()
    source.p0 = p0
    
    # run forward simulation
    sensor_data = kspaceFirstOrder3DG(
        kgrid,
        source,
        sensor,
        medium,
        simulation_options,
        execution_options
    )['p'].T
    print('sensor data')
    print(sensor_data.shape)
    
    start = timeit.default_timer()
    print("combining sensor data...")
    sensor_data = karray.combine_sensor_data(kgrid, sensor_data, sensor_mask)
    print(f'sensor data combined in {timeit.default_timer() - start} seconds')
    
    import matplotlib.pyplot as plt
    plt.imshow(sensor_data)
    plt.savefig('sensor_data.png')
    
    with h5py.File(cfg['name']+'/data.h5', 'w') as f:
        # MSOT uses a 12 Bit DAS
        f.create_dataset(
            'sensor_data', 
            data=sensor_data.astype(dtype=np.float16)
        )
    
    # TODO: finish this
    # for iterative time reversal reconstruction with positivity contraint
    # see k-wave example Matlab script (http://www.k-wave.org)
    # example_pr_2D_TR_iterative.m
    
    # redefine kgrid for 2D reconstruction
    kgrid = kWaveGrid(
        [cfg['grid_size'][0], cfg['grid_size'][2]],
        [cfg['dx'], cfg['dx']]
    )
    kgrid.setTime(cfg['Nt'], cfg['dt'])
    #kgrid.makeTime()
    
    # redefine sensor for 2D time reversal reconstruction
    # model each sensor is now a point source
    time_reversal_source_xz = np.matmul(
        uf.Ry2D(225 * np.pi / 180),
        make_cart_circle(
            radius_mm * 1e-3, 
            number_detector_elements,
            np.array([0.0, 0.0]),
            3 * np.pi / 2,
            plot_circle=False
        )
    )
    
    plt.clf()
    plt.scatter(time_reversal_source_xz[0], time_reversal_source_xz[1])
    plt.savefig('sensor_points.png')
    '''
    # define points to interpolate onto
    time_reversal_source_xz_interp = np.matmul(
        uf.Ry2D(225 * np.pi / 180),
        make_cart_circle(
            radius_mm * 1e-3, 
            number_detector_elements * 4,
            np.array([0.0, 0.0]),
            3 * np.pi / 2,
            plot_circle=False
        )
    )
    # convert interpolation points to binary mask
    source_mask_interp = cart2grid(
        kgrid,
        source_xz_interp
    )[0]
    # interpolate sensor data onto new mask
    source_mask_interp = interp_cart_data(
        kgrid,
        sensor_data, 
        source_xz,
        source_mask_interp,
        interp='nearest' # 'nearest' is default
    )
    '''
    
    # record 'final' pressure a.k.a. initial pressure reconstruction
    sensor_mask = cart2grid(kgrid, time_reversal_source_xz)[0]
    sensor = kSensor(sensor_mask)
    #sensor.mask = sensor_mask
    sensor.record = ['p_final']
    
    # reverse time axis
    sensor_data = np.flip(sensor_data, axis=1)
    # use sensor data as source with dirichlet boundary condition
    source = kSource()
    source.p = sensor_data
    source.p_mask = sensor_mask
    source.p_mode = 'dirichlet'
    
    print('kgrid.k')
    print(type(kgrid.k), kgrid.k.shape)
    print('source mask')
    print(type(source.p_mask), source.p_mask.shape)
    print('sensor mask')
    print(type(sensor.mask), sensor.mask.shape)
    print('kgrid')
    print(kgrid.Nx, kgrid.Ny)
    print('source.p')
    print(source.p.shape)
    print('kgrid.t_array')
    print(type(kgrid.t_array.shape), kgrid.t_array.shape)
    
    # NOTE: if the kgrid is so small that some point sources fall within the 
    # same grid points, then the simulation will fail with the following error:
    # ValueError: The number of time series in source.p must match the number of source elements in source.p_mask
    
    # run time reversal reconstruction
    p0_estimate = kspaceFirstOrder2DG(
        kgrid,
        source,
        sensor,
        medium,
        simulation_options,
        execution_options
    )['p_final']
        
    # apply positivity constraint
    p0_estimate = p0_estimate * (p0_estimate > 0.0)
    
    plt.imshow(p0_estimate)
    plt.savefig('p0_estimate.png')
    
    '''
    for i in range(cfg['TR_iterations']):
        
        source = kSource()
        source.p0 = p0_estimate
        
        sensor_data_new = kspaceFirstOrder2DG(
            kgrid,
            source,
            sensor,
            medium,
            simulation_options,
            execution_options
        )['p']
        
        source.p = sensor_data - sensor_data_new
    '''
    
    return p0_estimate

if __name__ == '__main__':
    N = 512 # number of grid points
    domain_size = 0.1
    dx = domain_size / (N-20)
    cfg = {
        'name': 'test_script',
        'grid_size': [N-20, (N//2)-20, N-20],
        'dx': 0.001,
        'c_0': 1500,
        'pml_size': 10,
        'TR_iterations': 3,
    }
    p0 = np.zeros(
        np.array(cfg['grid_size']),
        dtype=np.float32
    )
    p0[N//2-16:N//2+16, N//2-16:N//2+16, N//2-16:N//2+16] = 1.0
    uf.create_dir(cfg['name'])    
    p0_recon = kwave_MSOT_sim(cfg, p0)
    print(f'RMSE: {np.sqrt(np.mean((p0 - p0_recon)**2))}')