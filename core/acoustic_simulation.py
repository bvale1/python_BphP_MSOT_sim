from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.utils.kwave_array import kWaveArray
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DG
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.ksource import kSource
import numpy as np
import h5py


class kwave_adapter():
    '''
    ==================================Workflow==================================

    1. define kwave simulation grid object (KWaveGrid)
    
    2. define transducer object (KWaveArray)
    
    3. run simulation simulation on GPU with CUDA binaries (KWaveFirstOrder3DG)
    
    4. save senor data to HDF5 file
    
    5. TODO: add bandlimited impulse response (BLI) and noise
    
    6. run time reversal reconstruction (KWaveFirstOrder3DG)
    
    7. TODO: implement iterative time reversal reconstruction (ITR)
    
    8. Save reconstruction to HDF5 file

    ============================================================================
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.kgrid = kWaveGrid(
            cfg['grid_size'],
            [cfg['dx'], cfg['dx'], cfg['dx']],
        )
        self.kgrid.makeTime(cfg['c_0'])
        
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
                    y_inc = y_range[y_idx] * y_increment_angle;
                    idx = (x_idx-1) * num_interpolation_points_y + y_idx;
                    detector_positions[0, idx] = np.sin(np.pi/2 + pitch_angle * det_elements[det_idx] + x_inc) * np.sin(theta + y_inc) * (radius_mm - 0.5 * element_size)
                    detector_positions[1, idx] = np.cos(theta + y_inc) * (radius_mm - 0.5 * element_size)
                    detector_positions[2, idx] = np.cos(np.pi/2 + pitch_angle * det_elements[det_idx] + x_inc) * np.sin(theta + y_inc) * (radius_mm - 0.5 * element_size)
            karray.add_custom_element(
                detector_positions * 1e-3, 
                element_size * element_length * 1e-3, 
                2, # 2D surface in 3D space
                str(det_idx)
            )
        
        sensor_mask = karray.get_array_binary_mask(self.kgrid)
        print('sensor mask')
        print(type(sensor_mask))
        print(sensor_mask.shape)
        print(sensor_mask.dtype)
        print(sensor_mask)
        self.sensor = kSensor(sensor_mask)#, record=['p'])
        
        
    def run_kwave_forward(self, p_0):
        
        # configure simulation
        simulation_options = SimulationOptions(
            pml_inside=False,
            pml_size=self.cfg['pml_size'],
            data_cast='single',
            save_to_disk=True,
        )
        
        execution_options = SimulationExecutionOptions(
            is_gpu_simulation=True
        )
        
        # define source
        source = kSource()
        source.p0 = p_0
        
        sensor_data = kspaceFirstOrder3DG(
            self.kgrid,
            source,
            self.sensor,
            self.medium,
            simulation_options,
            execution_options
        )
        print('sensor data')
        print(type(sensor_data))
    
    #def run_kwave_backward(self, sensor_data):
    
    
# test script
if __name__ == '__main__':
    cfg = {
        'grid_size': [108, 108, 108],
        'dx': 0.001,
        'c_0': 1500,
        'pml_size': 10
    }
    p_0 = np.zeros(cfg['grid_size'], dtype=np.float32)
    p_0[48:81, 48:81, 48:81] = 1.0
    kwave_simulation = kwave_adapter(cfg)
    kwave_simulation.create_transducer_array()
    kwave_simulation.run_kwave_forward(p_0)