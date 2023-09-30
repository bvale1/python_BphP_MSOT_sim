from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
#from kwave.utils.kwave_array import kWaveArray
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DG
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.ksource import kSource
from kwave.utils.mapgen import make_cart_circle
from kwave.utils.interp import interp_cart_data
from kwave.utils.conversion import cart2grid
import utility_func as uf
import numpy as np
import h5py

class kwave_inverse_adapter():
    '''
    ==================================Workflow==================================

    1. define kwave simulation grid object (KWaveGrid) and medium object (KWaveMedium)
    
    2. define source with dirichlet boundary condition object (KSource)
    
    3. define sensor object (KSensor) to record final pressure field
    
    4. reverse time axis of sensor data and assign to source object
    
    5. run time reversal simulation on GPU with CUDA binaries (KWaveFirstOrder2DG)
            
    6. TODO: implement iterative time reversal reconstruction (ITR)

    7. Return inital pressure reconstruction

    ============================================================================
    '''
    def __init__(self, cfg):
        self.kgrid = kWaveGrid(
            [cfg['kwave_grid_size'][0], cfg['kwave_grid_size'][2]],
            [cfg['dx'], cfg['dx']],
        )
        self.kgrid.setTime(cfg['Nt'], cfg['dt'])
        self.cfg = cfg
        
        # Acoustical Characteristics of Biological Media, Jeffrey C. Bamber, 1997
        # k-wave uses Neper radian units (Np rad s^-1 m^-1)
        # 1 Np = 8.685889638 dB = 20.0 log_10(e) dB
        # 1 dB = 0.115129254 Np = 0.05 ln(10) Np
        # TODO: implement absorbing medium, k-wave documentation uses Np but
        # k-wave-python used dB, investigate further
        self.medium = kWaveMedium(
            sound_speed=cfg['c_0'],
            absorbing=False
        )
        
        
    def configure_simulation(self):
        self.simulation_options = SimulationOptions(
            pml_inside=False,
            pml_size=self.cfg['pml_size'],
            data_cast='single',
            save_to_disk=True,
        )
        
        self.execution_options = SimulationExecutionOptions(
            is_gpu_simulation=True
        )
        
        
    def create_point_source_array(self):
        number_detector_elements = 256
        radius_mm = 40.5
        
        self.reconstruction_source_xz = np.matmul(
            uf.Ry2D(225 * np.pi / 180),
            make_cart_circle(
                radius_mm * 1e-3, 
                number_detector_elements,
                np.array([0.0, 0.0]),
                3 * np.pi / 2,
                plot_circle=False
            )
        )
        
        [self.source_mask, self.mask_order_index, self.mask_reorder_index] = cart2grid(
            self.kgrid, self.reconstruction_source_xz
        )
        
        
    def run_time_reversal(self, sensor_data0):
        # for iterative time reversal reconstruction with positivity contraint
        # see k-wave example Matlab script (http://www.k-wave.org)
        # example_pr_2D_TR_iterative.m
        # along with S. R. Arridge et al. On the adjoint operator in
        # photoacoustic tomography. 2016. equation 28.
        # https://iopscience.iop.org/article/10.1088/0266-5611/32/11/115012/pdf
        
        # for cropping pml out of reconstruction
        pml = self.cfg['pml_size']
        # reverse time axis for sensor data at first iteration
        sensor_data0 = np.flip(sensor_data0, axis=1)
        # use sensor data as source with dirichlet boundary condition
        sensor = kSensor(self.source_mask)
        sensor.record = ['p_final']
        
        source = kSource()
        source.p_mask = self.source_mask
        source.p_mode = 'dirichlet'
        source.p = sensor_data0
        
        # run time reversal reconstruction
        p0_recon = kspaceFirstOrder2DG(
            self.kgrid,
            source,
            sensor,
            self.medium,
            self.simulation_options,
            self.execution_options
        )['p_final'][pml:-pml, pml:-pml]# crop pml from reconstruction
        
        # apply positivity constraint
        p0_recon *= (p0_recon > 0.0)
        
        # comment out later
        with h5py.File(self.cfg['save_dir']+'data.h5', 'r+') as f:
            try:
                print(f'creating p0_tr_{1} dataset')
                f.create_dataset(
                    f'p0_tr_{1}', data=uf.square_centre_crop(
                        p0_recon, self.cfg['crop_size']
                    )
                )
            except:
                print(f'p0_tr_{1} already exists')
        
        if self.cfg['recon_iterations'] > 1:
            for i in range(2, self.cfg['recon_iterations']+1):
                print(f'time reversal iteration {i} of {self.cfg["recon_iterations"]}')
                
                # run 2D simulation forward
                sensor = kSensor(self.source_mask)
                sensor.record = ['p']
                source = kSource()
                source.p0 = p0_recon
                
                # subtract residual sensor data from sensor data
                sensor_datai = kspaceFirstOrder2DG(
                    self.kgrid,
                    source,
                    sensor,
                    self.medium,
                    self.simulation_options,
                    self.execution_options
                )['p'].T
                
                # redefine sensor and source for time reversal
                sensor = kSensor(self.source_mask)
                sensor.record = ['p_final']
                source = kSource()
                source.p_mask = self.source_mask
                source.p_mode = 'dirichlet'
                source.p =  np.flip(sensor_datai, axis=1) - sensor_data0
                
                # run time reversal reconstruction
                p0_recon -= self.cfg['recon_alpha'] * kspaceFirstOrder2DG(
                    self.kgrid,
                    source,
                    sensor,
                    self.medium,
                    self.simulation_options,
                    self.execution_options
                )['p_final'][pml:-pml, pml:-pml]

                # apply positivity constraint
                p0_recon *= (p0_recon > 0.0)
        
                with h5py.File(self.cfg['save_dir']+'data.h5', 'r+') as f:
                    try:
                        print(f'creating p0_tr_{i} dataset')
                        f.create_dataset(
                            f'p0_tr_{i}', data=uf.square_centre_crop(
                                p0_recon, self.cfg['crop_size']
                            )
                        )
                    except:
                        print(f'p0_tr_{i} already exists')
        
        return p0_recon
    
    
    def reorder_sensor_xz(self):
        # this function should only be needed before running backprojection
        # k-wave indexes the binary sensor mask in column wise linear order
        # so senor_data[sensor_idx] is also in this order
        
        # inefficient sorting algorithm but it works for now
        source_grid_pos = []
        # index binary mask in column wise linear order
        for x in range(self.source_mask.shape[0]):
            for z in range(self.source_mask.shape[1]):
                if self.source_mask[x, z] == 1:
                    source_grid_pos.append([x, z])
        # convert to cartesian coordinates
        source_grid_pos = np.asarray(source_grid_pos).astype(np.float32)
        source_grid_pos -= self.cfg['crop_size']
        self.bp_source_xz = source_grid_pos.T * self.cfg['dx']
    '''
    def interpolate_sensor_data(self, sensor_data):
        
        sensor_data_interp = interp_cart_data(
            self.kgrid, sensor_data, 
        )
    '''
    def run_backprojection(self, sensor_data):
        # TODO: FIX THIS
        # I THINK I AM INDEXING THE SENSOR DATA IN THE WRONG ORDER
        
        # minus one since MATLAB indexes from 1
        print('self.mask_order_index.shape')
        print(self.mask_order_index.shape)
        #print(self.mask_order_index)
        print('self.mask_reorder_index.shape')
        print(self.mask_reorder_index.shape)
        #print(self.mask_reorder_index)
        
        print('sensor_data.shape')
        print(sensor_data.shape)
        sensor_data = sensor_data[self.mask_order_index-1,:][:,0,:]
        print('sensor_data.shape')
        print(sensor_data.shape)
        # reconstruct only region within 'crop_size'
        crop_size = self.cfg['crop_size']
        [X, Z] = np.meshgrid(
            (np.arange(crop_size) - crop_size/2) * self.cfg['dx'], 
            (np.arange(crop_size) - crop_size/2) * self.cfg['dx'],
            indexing='ij'
        )
        
        with h5py.File(self.cfg['save_dir']+'data.h5', 'r+') as f:
            try:
                print('creating p0_bp_sensors dataset')
                f.create_dataset(
                    'p0_bp_sensors',
                    data=np.zeros(
                        (sensor_data.shape[0], crop_size, crop_size), 
                        dtype=np.float32
                    )
                )
            except:
                print('p0_bp_sensors already exists')
        
        # X and Z can be flat
        X = X.ravel(); Z = Z.ravel()
        # compute euclidian distance from each sensor position to each grid point
        X = np.repeat(X[:, np.newaxis], sensor_data.shape[0], axis=1)
        Z = np.repeat(Z[:, np.newaxis], sensor_data.shape[0], axis=1)
        print('X.shape')
        print(X.shape)
        # time for wave to travel from each sensor to each grid point
        print('self.reconstruction_source_xz.shape')
        print(self.reconstruction_source_xz.shape)
        #print('self.bp_source_xz.shape')
        #print(self.bp_source_xz.shape)
        # reconstruction_source_xz should be shape (2, 256)
        delay = np.sqrt(
            (X - self.reconstruction_source_xz[0, :])**2 + 
            (Z - self.reconstruction_source_xz[1, :])**2
        ) / self.cfg['c_0']
        #delay = np.sort(delay, axis=0)
        print('delay.shape')
        print(type(delay))
        print(delay.shape)
        signal_amplitude = np.zeros_like(delay, dtype=np.float32)
        t_array = np.arange(self.cfg['Nt'], dtype=np.float32) * self.cfg['dt']
        for i, sensor in enumerate(sensor_data):
            print(f'backprojection sensor {i+1}/{sensor_data.shape[0]}')
            print('sensor.shape')
            print(sensor.shape)
            print('delay[:,i].shape')
            print(delay[:,i].shape)
            print('t_array')
            print(t_array.shape)
            signal_amplitude[:,i] = np.interp(
                delay[:,i], t_array, sensor
            )
            with h5py.File(self.cfg['save_dir']+'data.h5', 'r+') as f:
                f['p0_bp_sensors'][i] = np.reshape(
                    signal_amplitude[:,i], (crop_size, crop_size)
                )
        
        print('signal_amplitude.shape')
        print(signal_amplitude.shape)
        
        bp_recon = np.sum(signal_amplitude, axis=1)
        print('bp_recon.shape')
        print(bp_recon.shape)
        
        # apply positivity constraint
        bp_recon *= (bp_recon > 0.0)
        
        return np.reshape(bp_recon, (self.cfg['crop_size'], self.cfg['crop_size']))