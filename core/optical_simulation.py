import numpy as np
import subprocess, os, json, struct, logging
import func.utility_func as uf


class MCX_adapter():
    '''
    invision source:
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
    def __init__(self, cfg, source='invision') -> None:
        
        if source not in ['planar', 'invision']:
            raise ValueError(f'source must be "planar" or "invision", not "{source}"')
        
        uf.create_dir('temp')
        
        self.mcx_config_file = 'temp/InVision_BphP_MCX_Simulation.json'
        self.mcx_volume_binary_file = 'temp/InVision_BphP_MCX_Simulation.bin'
        self.mcx_out_file = 'temp/InVision_BphP_MCX_Simulation_out'
        
        # initialize simulation configuration
        self.mcx_cfg = {
            'Session': {
                'ID': self.mcx_out_file,
                'DoAutoThread': 1,
                'Photons': cfg['nphotons'] // cfg['nsources'], 
                'DoMismatch': 0
            }, 
            'Forward': {
                'T0': 0, 'T1': 1e-09, 'Dt': 1e-09
            }, 
            'Optode': {
                'Source': {
                    'Type': source,
                    'Pos': [0, 0, 0], 
                    'Dir': [0, 0, 0], 
                    'Param1': [cfg['dx'] * 1e3, 0, 0, 0], 
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
                'Dim': cfg['mcx_grid_size'], 
                'VolumeFile': self.mcx_volume_binary_file
            }
        }
        
    def set_planar_source(self, source_no) -> None:
        nxz = self.mcx_cfg['Domain']['Dim'][0]
        ny = self.mcx_cfg['Domain']['Dim'][1]
        nsource = 10        
        # SRCPOS Approximates the light source as a ring around the image plane
        # The ring is approximated as nsource quadrilaterals

        # angle between n straight lines which approximate the ring
        srcang = np.pi - (2 * np.pi / nsource)
        # source length
        srclen = (nxz / 2) * np.sin(2 * np.pi / nsource) / np.sin(srcang / 2)
        # position of 1st line source
        pos = np.array([[1 + (nxz / 2) - (srclen / 2)], [1], [1]])  # - [nxz*0.1, 0, 0];
        # specify vectors for the sides of the quadrilateral-shaped source
        param1 = np.array([[0], [ny], [0]])
        param2 = np.array([[srclen], [0], [0]])
        # incident photon direction of 1st source
        srcdir = np.array([[0], [0], [1]])
        # rotation matrix operator by 2*pi/nsource rads about y axis
        rotate = uf.Ry3D(-2 * np.pi / nsource)
        # iteratively place other sources around the xz plane
        for i in range(0, source_no + 1):
            pos = pos + param2
            param1 = np.dot(rotate, param1)
            param2 = np.dot(rotate, param2)
            srcdir = np.dot(rotate, srcdir)

        self.mcx_cfg['Optode']['Source']['Pos'] = pos[:,0].tolist()
        self.mcx_cfg['Optode']['Source']['Dir'] = srcdir[:,0].tolist()
        self.mcx_cfg['Optode']['Source']['Param1'] = param1[:,0].tolist()
        self.mcx_cfg['Optode']['Source']['Param2'] = param2[:,0].tolist()
    
    
    def set_invision_source(self, source_no) -> None:

        dx_mm = self.mcx_cfg['Domain']['LengthUnit'] # [mm]
        angle = 0.0 # [rad]
        det_sep_half = 24.74 / 2 # [mm]
        detector_iso_distance = 74.05 / 2 # [mm]
        illumination_angle = -0.41608649 # [rad]

        if source_no == 0:
            angle += 0.0
        elif source_no == 1:
            angle += 0.0
            det_sep_half = -det_sep_half
            illumination_angle = -illumination_angle
        elif source_no == 2:
            angle += 1.25664
        elif source_no == 3:
            angle += 1.25664
            det_sep_half = -det_sep_half
            illumination_angle = -illumination_angle
        elif source_no == 4:
            angle += -1.25664
        elif source_no == 5:
            angle += -1.25664
            det_sep_half = -det_sep_half
            illumination_angle = -illumination_angle
        elif source_no == 6:
            angle += 2.51327
        elif source_no == 7:
            angle += 2.51327
            det_sep_half = -det_sep_half
            illumination_angle = -illumination_angle
        elif source_no == 8:
            angle += -2.51327
        elif source_no == 9:
            angle += -2.51327
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
        
        self.mcx_cfg['Optode']['Source']['Param1'][1] = source_no


    def save_mcx_config(self) -> None:
        with open(self.mcx_config_file, "w") as json_file:
            json.dump(self.mcx_cfg, json_file, indent="\t")
    
    
    def save_mcx_volume_binary(self, volume):
        # volume must be a float32 numpy array of shape (2, nx, ny, nz)
        volume = list(np.reshape(volume, volume.size, "F"))
        volume  = struct.pack("f" * len(volume), *volume)
        with open(self.mcx_volume_binary_file, "wb") as input_file:
            input_file.write(volume)
            
        
    def run_mcx(self, 
                mcx_bin_path : str, 
                volume : np.array,
                ) -> np.array:
        """
        runs subprocess calling MCX with the flags built with `self.get_command`.
        Rises a `RuntimeError` if the code exit of the subprocess is not 0.

        :param cmd: list defining command to parse to `subprocess.run`
        :return: None
        """        
        volume *= 1e-3 # [m^-1] -> [mm^-1]
        
        self.save_mcx_volume_binary(volume)
        
        # Output flag 'E' returns the energy absorbed in each voxel
        # Output flag 'F' returns the fluence in each voxel
        cmd = [mcx_bin_path, '-f', self.mcx_config_file, '-O', 'F']
        mcx_out = np.zeros(self.mcx_cfg['Domain']['Dim'], dtype=np.float32)
        
        for i in range(10):
            print('source no: ', i)
            if self.mcx_cfg['Optode']['Source']['Type'] == 'planar':
                self.set_planar_source(i)
            elif self.mcx_cfg['Optode']['Source']['Type'] == 'invision':
                self.set_invision_source(i)
            self.save_mcx_config()
            print(f'mcx config: {self.mcx_cfg}')
            
            results = None
            try:
                results = subprocess.run(cmd)
                logging.info(f"MCX run: {cmd}, source no: {i}, results: {results}")
            except:
                raise RuntimeError(
                    f"MCX failed to run: {cmd}, source no: {i}, results: {results}"
                )
                
            mcx_out += self.read_mcx_output(self.mcx_out_file)
            
        # normalise
        mcx_out /= 10           
        
        return mcx_out
        

    def read_mcx_output(self, mcx_out_file) -> np.array:
        #reads the temporary output generated with MCX
        with open(mcx_out_file+'.mc2', 'rb') as f:
            mcx_out = f.read()
        mcx_out = struct.unpack(
            '%df' % (len(mcx_out) / 4), mcx_out
        )
        mcx_out = np.asarray(mcx_out).reshape(
            (self.mcx_cfg['Domain']['Dim']), order='F'
        )
        
        return mcx_out
        
        
    def delete_temporary_files(self) -> None:
        try:
            os.remove(self.mcx_config_file)
            os.remove(self.mcx_volume_binary_file)
            os.remove(self.mcx_out_file+'.mc2')
            os.rmdir('temp')
        except:
            print('could not delete temporary files')
        