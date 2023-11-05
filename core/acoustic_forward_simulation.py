from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.utils.kwave_array import kWaveArray
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DG
#from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DG
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.ksource import kSource
from kwave.utils.mapgen import make_cart_circle, make_circle
from kwave.utils.conversion import cart2grid
import utility_func as uf
from datetime import datetime
import numpy as np
import h5py, json, timeit, os, logging


logger = logging.getLogger(__name__)

# k-wave simulation may be run as a function, or as a class
# function sets up the simulation i.e. transducer array, grid, medium, etc...
# every time it is called making it slower but more memory efficient

class kwave_forward_adapter():
    '''
    ==================================Workflow==================================

    1. define kwave simulation grid object (KWaveGrid) and medium object (KWaveMedium)
    
    2. define transducer object (KWaveArray)
    
    3. run simulation on GPU with CUDA binaries (KWaveFirstOrder3DG)
    
    4. return save senor data to HDF5
    
    5. TODO: add bandlimited impulse response (BLI) and noise

    ============================================================================
    '''
    def __init__(self, cfg : dict, transducer_model='invision'):
        self.kgrid = kWaveGrid(
            cfg['kwave_grid_size'],
            [cfg['dx'], cfg['dx'], cfg['dx']],
        )
            
        #self.kgrid.makeTime(cfg['c_0']) # k-wave automatically determines Nt and dt
        self.kgrid.setTime(2030, 25e-9) # sampling rate used by the MSOT DAS
        #self.kgrid.setTime(2, 25e-9) # for testing
        cfg['dt'] = self.kgrid.dt
        cfg['Nt'] = self.kgrid.Nt
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
        
        if transducer_model == 'invision':
            self.combine_data = True
            logger.info('initialising invision transducer array')
            self.create_transducer_array()
        else:
            if transducer_model != 'point':
                logger.info(f'WARNING: transducer model {transducer_model} not recognised, using point source array')
            else:
                logger.info('initialising point source array')
            self.combine_data = False
            self.create_point_sensor_array()
        
        
    def create_point_sensor_array(self):
        # k-wave indexes the binary sensor mask in column wise linear order
        n = 256 # number of elements
        r  = 0.0402 # [m]
        arc_angle = 270*np.pi/(n*180) # [rad]
        theta = np.linspace((5*np.pi/4)-(arc_angle), (-np.pi/4)+(arc_angle), n) # [rad]
        x = r*np.sin(theta) # [m]
        z = r*np.cos(theta) # [m]
        
        detector_positions = np.array([x, np.zeros(n), z])
        #detector_positions = np.matmul(uf.Ry3D(-np.pi / 2), np.array([x, np.zeros(n) z]))
    
        [self.sensor_mask, self.mask_order_index, self.mask_reorder_index] = cart2grid(
            self.kgrid, detector_positions
        )
        self.combine_data = False
        # these two fields are used for backprojection
        self.source_x = detector_positions[0, :]
        self.source_z = detector_positions[1, :]
        
        
    def create_transducer_array(self):
        # transducer array geometry provided by Janek Gröhl https://github.com/jgroehl
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
        karray = kWaveArray(bli_tolerance=0.05, upsampling_rate=10, single_precision=True)
        
        #Ry = uf.Ry3D(-np.pi / 2) # euclidian rotation matrix
        
        theta = np.pi/2
        for det_idx in range(len(det_elements)):
            detector_positions = np.zeros(
                (3, num_interpolation_points_x * num_interpolation_points_y)
            )
            for x_idx in range(len(x_range)):
                x_inc = x_range[x_idx] * x_increment_angle
                for y_idx in range(len(y_range)):
                    y_inc = y_range[y_idx] * y_increment_angle
                    idx = (x_idx-1) * num_interpolation_points_y + y_idx
                    detector_positions[0, idx] = np.sin(np.pi/2 + pitch_angle * det_elements[det_idx] + x_inc) * np.sin(theta + y_inc) * (radius_mm - 0.5 * element_size)
                    detector_positions[1, idx] = np.cos(theta + y_inc) * (radius_mm - 0.5 * element_size)
                    detector_positions[2, idx] = np.cos(np.pi/2 + pitch_angle * det_elements[det_idx] + x_inc) * np.sin(theta + y_inc) * (radius_mm - 0.5 * element_size)
            #detector_positions = np.matmul(Ry, detector_positions)
            karray.add_custom_element(
                detector_positions * 1e-3, 
                element_size * element_length * 1e-3, 
                2, # 2D surface in 3D space
                str(det_idx)
            )
            
        self.combine_data = True
        start = timeit.default_timer()
        (self.sensor_mask, self.sensor_weights, self.sensor_local_ind) = self.check_for_grid_weights()
        logger.info(f'grid weights checked in {timeit.default_timer() - start} seconds')
        if self.sensor_mask is None:
            logger.info('no viable binary sensor mask found, computing...')
            self.sensor_mask = karray.get_array_binary_mask(self.kgrid)
            self.save_weights = True
        elif self.sensor_weights and self.sensor_local_ind:
            logger.info('binary mask and grid weights found')
            self.save_weights = False
            
        self.karray = karray
        # records pressure by default
        self.sensor = kSensor(self.sensor_mask)#, record=['p'])
        
    
    def configure_simulation(self):
        self.simulation_options = SimulationOptions(
            pml_inside=False,
            pml_size=self.cfg['pml_size'],
            data_cast='single',
            save_to_disk=True,
        )
        
        self.execution_options = SimulationExecutionOptions(
            is_gpu_simulation=True,
            verbose_level=2
        )
        
        
    def run_kwave_forward(self, p0):
        
        sensor = kSensor(self.sensor_mask)
        sensor.record = ['p']
        sensor = sensor
        source = kSource()
        source.p0 = p0
        
        # run forward simulation
        sensor_data = kspaceFirstOrder3DG(
            self.kgrid,
            source,
            sensor,
            self.medium,
            self.simulation_options,
            self.execution_options
        )['p'].T
        
        if self.combine_data:
            if self.sensor_weights is None or self.sensor_local_ind is None:
                self.save_weights = True
                logger.info('no viable grid weights found, computing...')
            else:
                self.save_weights = False
            
            start = timeit.default_timer()
            logger.info("combining sensor data...")
            (sensor_data, self.sensor_weights, self.sensor_local_ind) = self.karray.combine_sensor_data(
                self.kgrid, 
                sensor_data,
                mask=self.sensor_mask,
                sensor_weights=self.sensor_weights, 
                sensor_local_ind=self.sensor_local_ind
            )
            logger.info(f'sensor data combined in {timeit.default_timer() - start} seconds')
                
            if self.save_weights:
                start = timeit.default_timer()
                self.save_sensor_weights(
                    self.sensor_mask,
                    self.sensor_weights, 
                    self.sensor_local_ind
                )
                logger.info(f'sensor weights saved in {timeit.default_timer() - start} seconds')
                
        return sensor_data
    
    def check_for_grid_weights(self) -> tuple:
        # checks if the source grid weights have been computed and saved by a
        # previous simulation of the same geometry 
        sensor_mask = None 
        sensor_weights = None
        sensor_local_ind = None
        
        if not os.path.exists(self.cfg["weights_dir"]):
            logger.debug(f'directory not found: {self.cfg["weights_dir"]}')
            return (sensor_mask, sensor_weights, sensor_local_ind)
        
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
                    logger.debug(f'viable grid weights found in {cfg_path}, loading...')
                    sensor_weights = []
                    sensor_local_ind = []
                    
                    with h5py.File(os.path.join(self.cfg['weights_dir'], folder, 'weights3d.h5'), 'r') as f:
                        sensor_mask = f['sensor_mask'][()].astype(bool)
                        for i in range(self.cfg['nsensors']):
                            sensor_weights.append(f[f'sensor_weights_{i}'][()].astype(np.float32))
                            sensor_local_ind.append(f[f'sensor_local_ind_{i}'][()].astype(bool))
                    break
            else:
                logger.debug(f'config file not found: {cfg_path}')
                
        return (sensor_mask, sensor_weights, sensor_local_ind)
    
    
    def save_sensor_weights(self, sensor_mask : np.ndarray, sensor_weights : list, sensor_local_ind : list):
        uf.create_dir(self.cfg['weights_dir'])
        save_path = self.cfg['weights_dir'] + datetime.utcnow().strftime('%Y%m%d_%H_%M_%S')
        uf.create_dir(save_path)

        weights_cfg = {
            'dx' : self.cfg['dx'],
            'kwave_grid_size' : self.cfg['kwave_grid_size'],
            'kwave_domain_size' : self.cfg['kwave_domain_size'],
            'sim_git_hash' : self.cfg['sim_git_hash'],
            'save_dir' : self.cfg['save_dir'],
            'transducer_model' : 'invision'
        }
        with h5py.File(save_path + '/weights3d.h5', 'w') as f:
            f.create_dataset('sensor_mask', data=self.sensor_mask, dtype=bool)
            for i in range(self.cfg['nsensors']):
                f.create_dataset(f'sensor_weights_{i}', data=sensor_weights[i], dtype=np.float32)
                f.create_dataset(f'sensor_local_ind_{i}', data=sensor_local_ind[i], dtype=bool)
            
        with open(save_path + '/weights_config.json', 'w') as f:
            json.dump(weights_cfg, f, indent='\t')