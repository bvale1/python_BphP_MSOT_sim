import numpy as np
import subprocess
import os
import json
import struct
import gc
import utility_func as uf


class MCX_adapter():
    '''
    This class is a rip off of the SIMPA class
    optical_forward_model_mcx_adapter.py
    https://github.com/IMSY-DKFZ/simpa
    
    ==================================Workflow==================================

    1. save simulation configuration to json file
    
    2. save volume array to binary file
    
    3. Execute MCX CUDA binary through Python
    
    4. Get normalised fluence map (array) from MCX output
    
    5. return initial acoustic pressure (p_0) of sample
    
    6. remove temporary files (optional)

    ============================================================================
    '''
    def __init__(self, cfg) -> None:
        
        uf.create_dir('temp')
        
        self.mcx_config_file = 'temp/InVision_BphP_MCX_Simulation.json'
        self.mcx_volume_binary_file = 'temp/InVision_BphP_MCX_Simulation.bin'
        self.mcx_out_file = 'temp/InVision_BphP_MCX_Simulation_out'
        self.grunesien = cfg['gruneisen']
        
        # initialize simulation configuration
        self.mcx_cfg = {
            'Session': {
                'ID': self.mcx_out_file,
                'DoAutoThread': 1,
                'Photons': cfg['nphotons'] // cfg['nsources'], 
                'DoMismatch': 0, 
                'RNGSeed': 4711
            }, 
            'Forward': {
                'T0': 0, 'T1': 1e-09, 'Dt': 1e-09
            }, 
            'Optode': {
                'Source': {
                    'Type': 'invision',
                    'Pos': [0, 0, 0], 
                    'Dir': [0, 0, 0], 
                    'Param1': [0.5, 0, 0, 0], 
                    'Param2': [0, 0, 0, 0]
                }
            }, 
            'Domain': {
                'OriginType': 0,
                'LengthUnit': cfg['dx'] * 1e3, 
                'Media': [
                    {'mua': 0, 'mus': 0, 'g': 1, 'n': 1.33}, 
                    {'mua': 1, 'mus': 1, 'g': 0.9, 'n': 1.33}
                ], 
                'MediaFormat': 'muamus_float',
                'Dim': cfg['grid_size'], 
                'VolumeFile': self.mcx_volume_binary_file
            }
        }
        
    '''
    def set_source(self, source_no) -> None:
        # source number must be 0 to 9 for the invision source,
        # don't really know what these number mean, found them in SIMPA
        dx = self.mcx_cfg['Domain']['LengthUnit']
        angle = -2.51327 # [rad]
        angle -= 1.256635 * np.floor(source_no / 2)
        det_iso_distance = 37.025 # [mm]
        illumination_angle = -0.41608649 * ((-1)**source_no) # [rad]
        det_sep_half = 12.37 * ((-1)**source_no) # [mm]
        
        # [mm] -> [grid points]
        postion = np.array([np.sin(angle) * det_iso_distance,
                            det_sep_half,
                            np.cos(angle) * det_iso_distance]) / dx + 1
        # mcx bottom front left corner is (0,0,0)
        postion += np.asarray(self.mcx_cfg['Domain']['Dim']) / 2
        # [vector]
        direction = np.array([-np.sin(angle),
                              np.sin(illumination_angle),
                              np.cos(angle)])

        self.mcx_cfg['Optode']['Source']['Pos'] = postion.tolist()
        # normalize direction vector
        self.mcx_cfg['Optode']['Source']['Dir'] = (
            direction / np.linalg.norm(direction)
        ).tolist()
        self.mcx_cfg['Optode']['Source']['Param1'] = [0.5, source_no, 0, 0]
    '''
    def set_source(self, source_no) -> None:

        dx_mm = self.mcx_cfg['Domain']['LengthUnit'] # [mm]
        angle = 0.0 # [rad]
        det_sep_half = 24.74 / 2 # [mm]
        detector_iso_distance = 74.05 / 2 # [mm]
        illumination_angle = -0.41608649 # [rad]

        if source_no == 0:
            angle = 0.0
        elif source_no == 1:
            angle = 0.0
            det_sep_half = -det_sep_half
            illumination_angle = -illumination_angle
        elif source_no == 2:
            angle = 1.25664
        elif source_no == 3:
            angle = 1.25664
            det_sep_half = -det_sep_half
            illumination_angle = -illumination_angle
        elif source_no == 4:
            angle = -1.25664
        elif source_no == 5:
            angle = -1.25664
            det_sep_half = -det_sep_half
            illumination_angle = -illumination_angle
        elif source_no == 6:
            angle = 2.51327
        elif source_no == 7:
            angle = 2.51327
            det_sep_half = -det_sep_half
            illumination_angle = -illumination_angle
        elif source_no == 8:
            angle = -2.51327
        elif source_no == 9:
            angle = -2.51327
            det_sep_half = -det_sep_half
            illumination_angle = -illumination_angle

        # [mm]
        device_position_mm = np.array([np.sin(angle) * detector_iso_distance,
                                       det_sep_half,
                                       np.cos(angle) * detector_iso_distance])

        # [dimensionless vector]
        source_direction_vector = np.array([-np.sin(angle),
                                            np.sin(illumination_angle),
                                            np.cos(angle)])
        
        # [mm] -> [grid points]
        self.mcx_cfg['Optode']['Source']['Pos'] = (
            (device_position_mm  / (dx_mm) + 1) + (np.asarray(self.mcx_cfg['Domain']['Dim']) / 2)
        ).tolist()
        
        # [dimensionless vector] -> [unit vector]
        self.mcx_cfg['Optode']['Source']['Dir'] = (
            source_direction_vector / np.linalg.norm(source_direction_vector)
        ).tolist()
        
        self.mcx_cfg['Optode']['Source']['Param1'] = [0.5, source_no, 0, 0]


    def save_mcx_config(self) -> None:
        with open(self.mcx_config_file, "w") as json_file:
            json.dump(self.mcx_cfg, json_file, indent="\t")
    
    
    def save_mcx_volume_binary(self, volume):
        # volume must be a numpy array of shape (2, nx, ny, nz)
        # possibly needs to be float64 but if not float32 is prefered
        # no idea how this works but it does
        volume = list(np.reshape(volume, volume.size, "F"))
        volume  = struct.pack("f" * len(volume), *volume)
        with open(self.mcx_volume_binary_file, "wb") as input_file:
            input_file.write(volume)        
        
        
    def run_mcx(self, 
                mcx_bin_path, 
                volume,
                ReBphP_PCM_Pr_c,
                ReBphP_PCM_Pfr_c,
                ReBphP_PCM,
                wavelength_index
                ) -> np.array:
        """
        runs subprocess calling MCX with the flags built with `self.get_command`. Rises a `RuntimeError` if the code
        exit of the subprocess is not 0.

        :param cmd: list defining command to parse to `subprocess.run`
        :return: None
        """
        # add the absorption coefficient of BphPs to the volume
        volume[0] += (ReBphP_PCM_Pr_c * ReBphP_PCM['Pr']['epsilon_a'][wavelength_index] + 
                      ReBphP_PCM_Pfr_c * ReBphP_PCM['Pfr']['epsilon_a'][wavelength_index])
        volume *= 1e-3 # [m^-1] -> [mm^-1]
        
        self.save_mcx_volume_binary(volume)
        
        # Output flag 'E' returns the energy absorbed in each voxel
        # Output flag 'F' returns the fluence in each voxel
        cmd = [mcx_bin_path, '-f', self.mcx_config_file, '-O', 'E']
        energy_absorbed = np.zeros(self.mcx_cfg['Domain']['Dim'], dtype=np.float32)
        
        for i in range(10):
            self.set_source(i)
            self.save_mcx_config()
            print('mcx config ', self.mcx_cfg)
            
            results = None
            try:
                results = subprocess.run(cmd)
            except:
                raise RuntimeError(
                    f"MCX failed to run: {cmd}, source no: {i}, results: {results}"
                )
                
            energy_absorbed += self.read_mcx_output(self.mcx_out_file)
            print('energy absorbed:')
            print(np.shape(energy_absorbed))
            print(energy_absorbed[182, 46, 182])
        
        return energy_absorbed
        # intial acoustic pressure (mu_a [mm^-1] -> [m^-1])
        #p0 = self.grunesien * fluence * (volume[0] * 1e3)
        
        #return p_0
        

    def read_mcx_output(self, mcx_out_file) -> np.array:
            #reads the temporary output generated with MCX
            with open(mcx_out_file+'.mc2', 'rb') as f:
                energy_absorbed = f.read()
            energy_absorbed = struct.unpack(
                '%df' % (len(energy_absorbed) / 4), energy_absorbed
            )
            energy_absorbed = np.asarray(energy_absorbed).reshape(
                (self.mcx_cfg['Domain']['Dim']), order='F'
            )
            
            return energy_absorbed
        
        
    def delete_temporary_files(self) -> None:
        os.remove(self.mcx_config_file)
        os.remove(self.mcx_volume_binary_file)
        os.remove(self.mcx_out_file+'.mc2')
        os.rmdir('temp')
        # collect garbage
        gc.collect()