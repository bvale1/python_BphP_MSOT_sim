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
from kwave.utils.kwave_array import kWaveArray
import func.utility_func as uf
import numpy as np
import h5py
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class kwave_inverse_adapter():
    '''
    ==================================Workflow==================================

    1. define kwave simulation grid object (KWaveGrid) and medium object (KWaveMedium)
    
    2. define source with dirichlet boundary condition object (KSource)
    
    3. define sensor object (KSensor) to record final pressure field
    
    4. reverse time axis of sensor data and assign to source object
    
    5. run time reversal simulation on GPU with CUDA binaries (KWaveFirstOrder2DG)
            
    6. perform iterative time reversal reconstruction (ITR)

    7. Return inital pressure reconstruction

    ============================================================================
    '''
    def __init__(self, cfg : dict, transducer_model='invision'):
        self.kgrid = kWaveGrid(
            [cfg['kwave_grid_size'][0], cfg['kwave_grid_size'][2]],
            [cfg['dx'], cfg['dx']],
        )
        self.kgrid.setTime(cfg['Nt'], cfg['dt'])
        if cfg['interp_data'] and transducer_model=='invision':
            logger.info('WARNING: cannot interpolate sensor data with invision transducer model')
            cfg['interp_data'] = None
        self.cfg = cfg
        
        if transducer_model == 'invision':
            self.create_arc_source_array()
            self.combine_data = True
        else:
            if transducer_model != 'point':
                logger.info(f'WARNING: transducer model {transducer_model} not recognised, using point source array')
            self.combine_data = False
            self.create_point_source_array()
        
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
        #logger.info(f'cfg[c_0] = {cfg["c_0"]}')
        #logger.info(f'medium.c_0 = {self.medium.sound_speed}')
        
        
    def configure_simulation(self):
        self.simulation_options = SimulationOptions(
            pml_inside=False,
            pml_size=self.cfg['pml_size'],
            data_cast='single',
            save_to_disk=True,
        )
        
        self.execution_options = SimulationExecutionOptions(
            is_gpu_simulation=True,
            verbose_level=2,
        )
        
        
    def create_point_source_array(self):
        # TODO: fix sensor data indexing order
        # k-wave indexes the binary sensor mask in column wise linear order
        n = 256 # number of elements
        r  = 0.0402 # [m]
        arc_angle = 270*np.pi/(n*180) # [rad]
        theta = np.linspace((5*np.pi/4)-(arc_angle), (-np.pi/4)+(arc_angle), n) # [rad]
        x = r*np.sin(theta) # [m]
        z = r*np.cos(theta) # [m]
        
        #detector_positions = np.array([x, z])
        detector_positions = np.matmul(uf.Ry2D(np.pi / 2), np.array([x, z]))
    
        [self.sensor_mask, self.mask_order_index, self.mask_reorder_index] = cart2grid(
            self.kgrid, detector_positions
        )
        self.combine_data = False
        # these two fields are used for backprojection
        self.source_x = detector_positions[0, :]
        self.source_z = detector_positions[1, :]
        
        
    def create_arc_source_array(self):
        r = 0.0402 # [m]
        n = 256 # number of elements
        # dimensions of each element
        arc_angle = 270*np.pi/(n*180) # [rad]
        cord = 2*r*np.sin(arc_angle/2) # [m]
        # center of each element
        theta = np.linspace((5*np.pi/4)-(arc_angle), (-np.pi/4)+(arc_angle), n) # [rad]
        x = r*np.sin(theta) # [m]
        z = r*np.cos(theta) # [m]
               
        # initializse transducer array object
        karray = kWaveArray(bli_tolerance=0.05, upsampling_rate=10, single_precision=True)
        
        Ry = uf.Ry2D(np.pi / 2) # euclidian rotation matrix
        
        for i in range(n):
            position = np.matmul(Ry, np.array([x[i], z[i]])).tolist()
            #position = [x[i], z[i]]
            karray.add_arc_element(position, r, cord, [0.0, 0.0])
        
        self.source_x = x
        self.source_z = z
        (self.sensor_mask, self.sensor_weights, self.sensor_local_ind, self.save_path) = self.check_for_grid_weights()
        if self.sensor_mask is None:
            logger.info('no viable binary sensor mask found, computing...')
            self.save_weights = True
            self.sensor_mask = karray.get_array_binary_mask(self.kgrid)
        elif self.sensor_weights and self.sensor_local_ind:
            self.save_weights = False
            logger.info('binary mask and grid weights found')
            
        self.karray = karray
        self.combine_data = True
        
        
    def interpolate_sensor_data(self, sensor_data, nsensors=512):
        # Currently sensors are in the wrong position
        radius_mm = 40.5
        interp_source_xz = np.matmul(
            uf.Ry2D(225 * np.pi / 180),
            make_cart_circle(
                radius_mm * 1e-3, 
                nsensors, # from 256 to 512 sensors
                np.array([0.0, 0.0]),
                3 * np.pi / 2,
                plot_circle=False
            )
        )
        
        [interp_mask, interp_mask_order_index, interp_mask_reorder_index] = cart2grid(
            self.kgrid, interp_source_xz
        )
        
        interp_sensor_data = interp_cart_data(
            self.kgrid, sensor_data, self.reconstruction_source_xz, interp_mask, 
        )
        
        [self.sensor_mask, self.mask_order_index, self.mask_reorder_index] = [
            interp_mask, interp_mask_order_index, interp_mask_reorder_index
        ]
        
        return interp_sensor_data
    
        
    def run_time_reversal(self, sensor_data0):
        # for iterative time reversal reconstruction with positivity contraint
        # see k-wave example Matlab script (http://www.k-wave.org)
        # example_pr_2D_TR_iterative.m
        # also example_at_array_as_source.m for using an off grid arc as a source
        # along with S. R. Arridge et al. On the adjoint operator in
        # photoacoustic tomography. 2016. equation 28.
        # https://iopscience.iop.org/article/10.1088/0266-5611/32/11/115012/pdf
        
        # for cropping pml out of reconstruction
        pml = self.cfg['pml_size']
        # reverse time axis for sensor data at first iteration
        sensor_data0 = np.flip(sensor_data0, axis=1)
        
        # interpolate sensor data to 512 sensors
        if self.cfg['interp_data']:
            sensor_data0 = self.interpolate_sensor_data(
                sensor_data0, nsensors=self.cfg['interp_data']
            )
        
        # use sensor data as source with dirichlet boundary condition
        sensor = kSensor(self.sensor_mask)
        sensor.record = ['p_final']
        
        source = kSource()
        source.p_mask = self.sensor_mask
        source.p_mode = 'additive'
        if self.combine_data: # arc source array
            (source.p, self.sensor_weights, self.sensor_local_ind) = self.karray.get_distributed_source_signal(
                self.kgrid, 
                sensor_data0, 
                mask=self.sensor_mask,
                sensor_weights=self.sensor_weights,
                sensor_local_ind=self.sensor_local_ind
            )
        elif self.cfg['forward_model'] == 'invision': # point source array with invision forward model
            source.p = sensor_data0[self.mask_reorder_index,:]
        else: # point source array with point source forward model
            source.p = sensor_data0
        
        # run time reversal reconstruction
        p0_recon = kspaceFirstOrder2DG(
            self.kgrid,
            source,
            sensor,
            self.medium,
            self.simulation_options,
            self.execution_options
        )['p_final'][pml:-pml, pml:-pml].T # crop pml from reconstruction
        
        if self.combine_data and self.save_weights:
            self.save_sensor_weights(
                self.sensor_mask,
                self.sensor_weights, 
                self.sensor_local_ind
            )
        
        # apply positivity constraint
        p0_recon *= (p0_recon > 0.0)
        
        # uncomment for debugging to save first iteration when ['recon_iterations'] > 1
        '''
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
                f[f'p0_tr_{1}'][()] = uf.square_centre_crop(
                    p0_recon, self.cfg['crop_size']
                )
        '''
        if self.cfg['recon_iterations'] > 1:
            for i in range(2, self.cfg['recon_iterations']+1):
                logger.info(f'time reversal iteration {i} of {self.cfg["recon_iterations"]}')
                
                # run 2D simulation forward
                sensor = kSensor(self.sensor_mask)
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
                
                if self.combine_data: # arc source array
                    (sensor_datai, _, _) = self.karray.combine_sensor_data(
                        self.kgrid,
                        sensor_datai,
                        mask=self.sensor_mask,
                        sensor_weights=self.sensor_weights,
                        sensor_local_ind=self.sensor_local_ind
                    )
                
                # redefine sensor and source for time reversal
                sensor = kSensor(self.sensor_mask)
                sensor.record = ['p_final']
                source = kSource()
                source.p_mask = self.sensor_mask
                source.p_mode = 'additive'
                
                if self.combine_data: # arc source array
                    (source.p, _, _) = self.karray.get_distributed_source_signal(
                        self.kgrid, 
                        np.flip(sensor_datai, axis=1) - sensor_data0,
                        mask=self.sensor_mask,
                        sensor_weights=self.sensor_weights,
                        sensor_local_ind=self.sensor_local_ind
                    )
                elif self.cfg['forward_model'] == 'invision': # point source array with invision forward model
                    source.p = (np.flip(sensor_datai, axis=1) - sensor_data0)[self.mask_reorder_index]
                else: # point source array with point source forward model
                    source.p = (np.flip(sensor_datai, axis=1) - sensor_data0)
                
                # run time reversal reconstruction
                p0_recon -= self.cfg['recon_alpha'] * kspaceFirstOrder2DG(
                    self.kgrid,
                    source,
                    sensor,
                    self.medium,
                    self.simulation_options,
                    self.execution_options
                )['p_final'][pml:-pml, pml:-pml].T

                # apply positivity constraint
                p0_recon *= (p0_recon > 0.0)
        
                # uncomment to save each iteration
                '''
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
                        f[f'p0_tr_{i}'][()] = uf.square_centre_crop(
                            p0_recon, self.cfg['crop_size']
                        )
                '''
        
        return p0_recon
    
    
    def reorder_sensor_xz(self):
        # k-wave indexes the binary sensor mask in column wise linear order
        # so senor_data[sensor_idx] is also in this order
        
        # inefficient sorting algorithm but it works for now
        source_grid_pos = []
        # index binary mask in column wise linear order
        for x in range(self.sensor_mask.shape[0]):
            for z in range(self.sensor_mask.shape[1]):
                if self.sensor_mask[x, z] == 1:
                    source_grid_pos.append([x, z])
        # convert to cartesian coordinates
        source_grid_pos = np.asarray(source_grid_pos).astype(np.float32)
        source_grid_pos -= self.cfg['crop_size']
        self.bp_source_xz = source_grid_pos.T * self.cfg['dx']
        
    
    def run_backprojection(self, sensor_data):
        
        # known bug
        #Traceback (most recent call last):
        #  File "/scratch/condor/dir_2955140/python_BphP_MSOT_sim/core/main.py", line 532, in <module>
            # bp = simulation.run_backprojection(out)
        #  File "/scratch/condor/dir_2955140/python_BphP_MSOT_sim/core/acoustic_inverse_simulation.py", line 3$    sensor_data = sensor_data[self.mask_reorder_index]
        # IndexError: arrays used as indices must be of integer (or boolean) type
        #if self.combine_data == False and self.cfg['forward_model'] == 'point':
        #    print(f'self.mask_reorder_index: {self.mask_reorder_index.shape}, {self.mask_reorder_index.dtype}')
        #    sensor_data = sensor_data[self.mask_reorder_index]
        
        logger.debug('sensor_data.shape')
        logger.debug(sensor_data.shape)
        # reconstruct only region within 'crop_size'
        crop_size = self.cfg['crop_size']
        [X, Z] = np.meshgrid(
            (np.arange(crop_size) - crop_size/2) * self.cfg['dx'], 
            (np.arange(crop_size) - crop_size/2) * self.cfg['dx'],
            indexing='ij'
        )
        ''' # uncomment to save each sensor data to file for debugging
        with h5py.File(self.cfg['save_dir']+'data.h5', 'r+') as f:
            try:
                logger.info('creating p0_bp_sensors dataset')
                f.create_dataset(
                    'p0_bp_sensors',
                    data=np.zeros(
                        (sensor_data.shape[0], crop_size, crop_size), 
                        dtype=np.float32
                    )
                )
            except:
                logger.info('p0_bp_sensors already exists')
        '''
        # X and Z can be flat
        X = X.ravel(); Z = Z.ravel()
        # compute euclidian distance from each sensor position to each grid point
        X = np.repeat(X[:, np.newaxis], sensor_data.shape[0], axis=1)
        Z = np.repeat(Z[:, np.newaxis], sensor_data.shape[0], axis=1)
        logger.debug('X.shape')
        logger.debug(X.shape)
        # time for wave to travel from each sensor to each grid point
        #logger.debug('self.reconstruction_source_xz.shape')
        #logger.debug(self.reconstruction_source_xz.shape)
        #print('self.bp_source_xz.shape')
        #print(self.bp_source_xz.shape)
        # reconstruction_source_xz should be shape (2, 256)
        delay = np.sqrt(
            (X - self.source_x)**2 + 
            (Z - self.source_z)**2
        ) / self.cfg['c_0']
        #delay = np.sort(delay, axis=0)
        logger.debug('delay.shape')
        logger.debug(type(delay))
        logger.debug(delay.shape)
        signal_amplitude = np.zeros_like(delay, dtype=np.float32)
        t_array = np.arange(self.cfg['Nt'], dtype=np.float32) * self.cfg['dt']
        for i, sensor in enumerate(sensor_data):
            logger.debug(f'backprojection sensor {i+1}/{sensor_data.shape[0]}')
            logger.debug('sensor.shape')
            logger.debug(sensor.shape)
            logger.debug('delay[:,i].shape')
            logger.debug(delay[:,i].shape)
            logger.debug('t_array')
            logger.debug(t_array.shape)
            signal_amplitude[:,i] = np.interp(
                delay[:,i], t_array, sensor
            )
            # save each sensor data to file for debugging
            # with h5py.File(self.cfg['save_dir']+'data.h5', 'r+') as f:
            #    f['p0_bp_sensors'][i] = np.reshape(
            #        signal_amplitude[:,i], (crop_size, crop_size)
            #    )
        
        logger.debug('signal_amplitude.shape')
        logger.debug(signal_amplitude.shape)
        
        bp_recon = np.sum(signal_amplitude, axis=1)
        logger.debug('bp_recon.shape')
        logger.debug(bp_recon.shape)
        
        # apply positivity constraint
        #bp_recon *= (bp_recon > 0.0)
        
        return np.reshape(bp_recon, (self.cfg['crop_size'], self.cfg['crop_size']))
    
    def check_for_grid_weights(self) -> tuple:
        # checks if the source grid weights have been computed and saved by a
        # previous simulation of the same geometry 
        sensor_mask = None 
        sensor_weights = None
        sensor_local_ind = None
        save_path = None
        
        if not os.path.exists(self.cfg["weights_dir"]):
            logger.debug(f'directory not found: {self.cfg["weights_dir"]}')
            return (sensor_mask, sensor_weights, sensor_local_ind, save_path)
        
        for folder in os.listdir(self.cfg['weights_dir']):
            cfg_path = os.path.join(self.cfg['weights_dir'], folder, 'weights_config.json')
            logger.debug(f'checking for grid weights in {folder}')
            
            if os.path.exists(cfg_path) and os.path.isfile(cfg_path):
                with open(cfg_path, 'r') as f:
                    cfg = json.load(f)
                logger.debug(f'found config file: {cfg}')
        
                if (cfg['dx'] == self.cfg['dx'] 
                    and cfg['kwave_grid_size'] == self.cfg['kwave_grid_size'] 
                    and cfg['kwave_domain_size'] == self.cfg['kwave_domain_size']
                    and cfg['transducer_model'] == 'invision'):
                    if os.path.exists(os.path.join(self.cfg['weights_dir'], folder, 'weights2d.h5')):
                        logger.debug(f'viable simulation found in {cfg_path}, 2d weights found, loading...')
                        sensor_weights = []
                        sensor_local_ind = []
                        with h5py.File(os.path.join(self.cfg['weights_dir'], folder, 'weights2d.h5'), 'r') as f:
                            sensor_mask = f['sensor_mask'][()].astype(bool)
                            for i in range(self.cfg['nsensors']):
                                sensor_weights.append(f[f'sensor_weights_{i}'][()].astype(np.float32))
                                sensor_local_ind.append(f[f'sensor_local_ind_{i}'][()].astype(bool))
                        break
                    else: # only the 3d simulation was run
                        logger.debug(f'viable simulation found in {cfg_path}, 2d weights not found')
                        save_path = os.path.join(self.cfg['weights_dir'], folder)                  
                    
            else:
                logger.debug(f'config file not found: {cfg_path}')
                
        return (sensor_mask, sensor_weights, sensor_local_ind, save_path)
    
    def save_sensor_weights(self, sensor_mask : np.ndarray, sensor_weights : list, sensor_local_ind : list):
        uf.create_dir(self.cfg['weights_dir'])
        if self.save_path is None: # create new directory for weights
            self.save_path = self.cfg['weights_dir'] + datetime.utcnow().strftime('%Y%m%d_%H_%M_%S')
            uf.create_dir(self.save_path)
            with open(self.save_path + '/weights_config.json', 'w') as f:
                json.dump(weights_cfg, f, indent='\t')

        weights_cfg = {
            'dx' : self.cfg['dx'],
            'kwave_grid_size' : self.cfg['kwave_grid_size'],
            'kwave_domain_size' : self.cfg['kwave_domain_size'],
            'sim_git_hash' : self.cfg['sim_git_hash'],
            'save_dir' : self.cfg['save_dir'],
            'transducer_model' : 'invision'
        }
        with h5py.File(self.save_path + '/weights2d.h5', 'w') as f:
            f.create_dataset('sensor_mask', data=sensor_mask, dtype=bool)
            for i in range(self.cfg['nsensors']):
                f.create_dataset(f'sensor_weights_{i}', data=sensor_weights[i], dtype=np.float32)
                f.create_dataset(f'sensor_local_ind_{i}', data=sensor_local_ind[i], dtype=bool)
            
        