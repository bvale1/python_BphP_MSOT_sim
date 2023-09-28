from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
#from kwave.utils.kwave_array import kWaveArray
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DG
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.ksource import kSource
from kwave.utils.mapgen import make_cart_circle
#from kwave.utils.interp import interp_cart_data
from kwave.utils.conversion import cart2grid
import utility_func as uf
import numpy as np
import h5py
import timeit


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
        
        self.source_mask = cart2grid(self.kgrid, time_reversal_source_xz)[0]
        
        
    def run_time_reversal(self, sensor_data0, alpha=1.0):
        # for iterative time reversal reconstruction with positivity contraint
        # see k-wave example Matlab script (http://www.k-wave.org)
        # example_pr_2D_TR_iterative.m
        # along with S. R. Arridge et al. On the adjoint operator in
        # photoacoustic tomography. 2016. equation 28.
        # https://iopscience.iop.org/article/10.1088/0266-5611/32/11/115012/pdf
        
        # for cropping pml out of reconstruction
        pml = self.cfg['pml_size']
        
        # reverse time axis for sensor data at first iteration
        sensor_data0 = np.flip(sensor_data0, axis=1).astype(np.float32)
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
                p0_recon += alpha * kspaceFirstOrder2DG(
                    self.kgrid,
                    source,
                    sensor,
                    self.medium,
                    self.simulation_options,
                    self.execution_options
                )['p_final'][pml:-pml, pml:-pml]

                # apply positivity constraint
                p0_recon *= (p0_recon > 0.0)
        
        return p0_recon