from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.utils.kwave_array import kWaveArray
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
import timeit


class kwave_inverse_adapter():
    '''
    ==================================Workflow==================================

    1. define kwave simulation grid object (KWaveGrid)
    
    2. define transducer object (KWaveArray)
    
    3
    
    3. run time reversal simulation on GPU with CUDA binaries (KWaveFirstOrder2DG)
            
    7. TODO: implement iterative time reversal reconstruction (ITR)
    
    8. Save reconstruction to HDF5 file

    ============================================================================
    '''
    def __init__(self, cfg):
        self.kgrid = kWaveGrid(
            [cfg['grid_size'][0], cfg['grid_size'][2]],
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
        
        sensor_mask = cart2grid(self.kgrid, time_reversal_source_xz)[0]
        sensor = kSensor(sensor_mask)
        sensor.record = ['p_final']
        self.sensor = sensor
        
        
    def run_time_reversal(self, sensor_data):
        # reverse time axis
        sensor_data = np.flip(sensor_data, axis=1)
        # use sensor data as source with dirichlet boundary condition
        source = kSource()
        source.p = sensor_data
        source.p_mask = self.sensor.mask
        source.p_mode = 'dirichlet'
        
        # run time reversal reconstruction
        p0_estimate = kspaceFirstOrder2DG(
            self.kgrid,
            source,
            self.sensor,
            self.medium,
            self.simulation_options,
            self.execution_options
        )['p_final']
        
        # apply positivity constraint
        p0_estimate = p0_estimate * (p0_estimate > 0.0)
       
        import matplotlib.pyplot as plt
        plt.cla()
        plt.imshow(p0_estimate)
        plt.savefig('p0_estimate.png')
        
        return p0_estimate