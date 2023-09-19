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
import timeit

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
    def __init__(self, cfg):
        self.kgrid = kWaveGrid(
            cfg['kwave_grid_size'],
            [cfg['dx'], cfg['dx'], cfg['dx']],
        )
        self.kgrid.makeTime(cfg['c_0'])
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
        
        
    def create_point_sensor_array(self):
        number_detector_elements = 256
        radius_mm = 40.5
        sensor_xz = np.matmul(
            uf.Ry2D(225 * np.pi / 180),
            make_cart_circle(
                radius_mm * 1e-3, 
                number_detector_elements,
                np.array([0.0, 0.0]),
                3 * np.pi / 2,
                plot_circle=False
            )
        )
        
        sensor_xyz = np.concatenate(
            (
                np.expand_dims(sensor_xz[0], axis=0),
                np.zeros((1, self.cfg['nsensors']), dtype=np.float32),
                np.expand_dims(sensor_xz[1], axis=0)
            ), axis=0
        )
        
        self.sensor_mask = cart2grid(self.kgrid, sensor_xyz)[0]
        self.combine_data = False
        
        
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
        karray = kWaveArray(bli_tolerance=0.05, upsampling_rate=10)
        
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
            karray.add_custom_element(
                detector_positions * 1e-3, 
                element_size * element_length * 1e-3, 
                2, # 2D surface in 3D space
                str(det_idx)
            )
        
        self.sensor_mask = karray.get_array_binary_mask(self.kgrid)
        print('sensor mask')
        print(type(self.sensor_mask), self.sensor_mask.shape, self.sensor_mask.dtype)
        self.karray = karray
        # records pressure by default
        self.sensor = kSensor(self.sensor_mask)#, record=['p'])
        self.combine_data = True
        
    
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
            start = timeit.default_timer()
            print("combining sensor data...")
            sensor_data = self.karray.combine_sensor_data(
                self.kgrid, 
                sensor_data, 
                self.sensor_mask
            )
            print(f'sensor data combined in {timeit.default_timer() - start} seconds')
                
        return sensor_data#.astype(np.float16)
    
    
# test script
if __name__ == '__main__':
    cfg = {
        'kwave_grid_size': [108, 108, 108],
        'dx': 0.001,
        'c_0': 1500,
        'pml_size': 10
    }
    p0 = np.zeros(cfg['kwave_grid_size'], dtype=np.float32)
    p0[48:81, 48:81, 48:81] = 1.0
    forawrd_simulation = kwave_forward_adapter(cfg)
    forawrd_simulation.create_transducer_array()
    forawrd_simulation.run_kwave_forward(p0)