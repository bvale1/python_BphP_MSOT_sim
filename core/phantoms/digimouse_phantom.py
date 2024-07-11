import numpy as np
from phantoms.phantom import phantom
from scipy.ndimage import zoom


class digimouse_phantom(phantom):
    
    def __init__(self, digimouse_path : str):
        super().__init__()
        self.digimouse_path = digimouse_path
        
    def create_volume(self,
                      cfg: dict, 
                      y_idx : int,
                      rotate : int,
                      wavelength_m : int = 700e-9
                      ) -> tuple[np.ndarray, np.ndarray]:
        assert rotate in [0, 1, 2, 3], 'Rotation must be 0, 1, 2 or 3 corresponding to 0, pi/2, pi and 3*pi/2 respectively'
        assert 100 <= y_idx <= 875, 'y_idx must be between 100 and 875' 
        
        # load the digimouse phantom
        with open(self.digimouse_path, 'rb') as f:
            digimouse = np.fromfile(f, dtype=np.int8)
        digimouse = digimouse.reshape(380, 992, 208, order='F')
        
        # spatial dimensions of the digimouse phantom
        dx = 0.0001 # [m]
        
        zoom_factors = [n / o for n, o in zip([308, 992, 380], [308, 992, 208])]
        digimouse = zoom(digimouse, zoom_factors, order=0)
        # interpolate to mcx grid dx
        zoom_factors = [dx / cfg['dx'], dx / cfg['dx'], dx / cfg['dx']]
        digimouse = zoom(digimouse, zoom_factors, order=0)
        digimouse = np.rot90(digimouse, rotate, axes=(0, 2))
        [nx, ny, nz] = cfg['mcx_grid_size']
        # translate digimouse[380//2, y_idx, 380//2] to be at the center of the
        # volume (volume[nx//2, ny//2, nz//2])
        tissue_types = np.zeros(cfg['mcx_grid_size'], dtype=np.int8)
        if y_idx - ny//2 < 0:
            # pad the digimouse phantom with zeros (background/coupling medium)
            digimouse = np.pad(
                digimouse, 
                ((0, 0), (ny//2 - y_idx, 0), (0, 0)),
                mode='constant',
                constant_values=0
            )
            y_idx = ny//2
        if y_idx + ny//2 > digimouse.shape[1]:
            # pad the digimouse phantom with zeros (background/coupling medium)
            digimouse = np.pad(
                digimouse, 
                ((0, 0), (0, y_idx + ny//2 - digimouse.shape[1]), (0, 0)),
                mode='constant',
                constant_values=0
            )
        digimouse = digimouse[:, (y_idx-ny//2):(y_idx+ny//2), :]
        # place the digimouse phantom in the center of the volume
        tissue_types[(nx-digimouse.shape[0])//2:(nx+digimouse.shape[0])//2, :, (nz-digimouse.shape[2])//2:(nz+digimouse.shape[2])//2] = digimouse        
        
        # background mask
        bg_mask = tissue_types[:,ny//2,:] != 0
        
        coupling_medium_mu_a = 0.1 # [m^-1]
        coupling_medium_mu_s = 100 # [m^-1]
        wavelengths_nm = wavelength_m * 1e9 # [m] -> [nm]

        # blood volume fraction S_B, oxygen saturation x, water volume fraction S_W
        # note that the equation in the paper contains a typo, mu_a_HbO2 and mu_a_Hb are the wrong way around
        mu_a = lambda S_B, x, S_W : (S_B*(x*self.HbO2['mu_a'][0]+(1-x)*self.Hb['mu_a'][0]) + S_W*self.H2O['mu_a'][0]) # alexandrakis eta al. (2005)
        # power law function for reduced scattering coefficient
        mu_s_alex = lambda a, b : (a * (wavelengths_nm**(-b))) * 1e3 # alexandrakis eta al. (2005)
        mu_s_jac = lambda a, b : (a * ((wavelengths_nm/500)**(-b))) * 1e3 # Jacques & Stevens (2013) 
        
        absorption_coefficients = np.array([
        coupling_medium_mu_a, # 0 --> background
        mu_a(0.0033, 0.7, 0.5), # 1 --> skin --> adipose, alexandrakis eta al. (2005)
        mu_a(0.049, 0.8, 0.15), # 2 --> skeleton, alexandrakis eta al. (2005)
        mu_a(0.0033, 0.7, 0.5), # 3 --> eye --> adipose, alexandrakis eta al. (2005)
        mu_a(0.03, 0.6, 0.75), # 4 --> medulla --> Rat brain cortex, Jacques & Stevens (2013)
        mu_a(0.03, 0.6, 0.75), # 5 --> cerebellum --> Rat brain cortex, Jacques & Stevens (2013)
        mu_a(0.03, 0.6, 0.75), # 6 --> olfactory bulbs --> Rat brain cortex, Jacques & Stevens (2013)
        mu_a(0.03, 0.6, 0.75), # 7 --> external cerebrum --> Rat brain cortex, Jacques & Stevens (2013)
        mu_a(0.03, 0.6, 0.75), # 8 --> striatum --> Rat brain cortex, Jacques & Stevens (2013)
        mu_a(0.05, 0.75, 0.5), # 9 --> heart, alexandrakis eta al. (2005)
        mu_a(0.03, 0.6, 0.75), # 10 --> rest of the brain --> Rat brain cortex, Jacques & Stevens (2013)
        mu_a(0.07, 0.8, 0.5), # 11 --> masseter muscles, alexandrakis eta al. (2005)
        mu_a(0.0033, 0.7, 0.5), # 12 --> lachrymal glands --> adipose, alexandrakis eta al. (2005)
        self.H2O['mu_s'][0], # 13 --> bladder --> water, Hendrik Buiteveld (1994)
        mu_a(0.07, 0.8, 0.5), # 14 --> testis --> muscle, alexandrakis eta al. (2005)
        mu_a(0.01, 0.7, 0.8), # 15 --> stomach, alexandrakis eta al. (2005)
        mu_a(0.3, 0.75, 0.7), # 16 --> spleen, alexandrakis eta al. (2005)
        mu_a(0.3, 0.75, 0.7), # 17 --> pancreas --> liver & spleen, alexandrakis eta al. (2005)
        mu_a(0.3, 0.75, 0.7), # 18 --> liver, alexandrakis eta al. (2005)
        mu_a(0.056, 0.75, 0.8),  # 19 --> kidneys, alexandrakis eta al. (2005)
        mu_a(0.07, 0.8, 0.5), # 20 --> adrenal glands --> muscle, alexandrakis eta al. (2005)
        mu_a(0.15, 0.85, 0.85) # 21 --> lungs, alexandrakis eta al. (2005)
        ]) # [m^-1]

        scattering_coefficients = np.array([
        0.0, # 0 --> background
        mu_s_alex(38, 0.53), # 1 --> skin --> adipose, alexandrakis eta al. (2005)
        mu_s_alex(35600, 1.47), # 2 --> skeleton, alexandrakis eta al. (2005)
        mu_s_alex(38, 0.53), # 3 --> eye --> adipose, alexandrakis eta al. (2005)
        mu_s_jac(2.14, 1.2), # 4 --> medulla --> brain, Jacques & Stevens (2013)
        mu_s_jac(2.14, 1.2), # 5 --> cerebellum --> brain, Jacques & Stevens (2013)
        mu_s_jac(2.14, 1.2), # 6 --> olfactory bulbs --> brain, Jacques & Stevens (2013)
        mu_s_jac(2.14, 1.2), # 7 --> external cerebrum --> brain, Jacques & Stevens (2013)
        mu_s_jac(2.14, 1.2), # 8 --> striatum --> brain, Jacques & Stevens (2013)
        mu_s_alex(10600, 1.43), # 9 --> heart, alexandrakis eta al. (2005)
        mu_s_jac(2.14, 1.2), # 10 --> rest of the brain --> brain, Jacques & Stevens (2013)
        mu_s_alex(4e7, 2.82), # 11 --> masseter muscles, alexandrakis eta al. (2005)
        mu_s_alex(38, 0.53), # 12 --> lachrymal glands --> adipose, alexandrakis eta al. (2005)
        self.H2O['mu_s'][0], # 13 --> bladder --> water, Hendrik Buiteveld (1994)
        mu_s_alex(4e7, 2.82), # 14 --> testis --> muscle, alexandrakis eta al. (2005)
        mu_s_alex(792, 0.97), # 15 --> stomach, alexandrakis eta al. (2005)
        mu_s_alex(629, 1.05), # 16 --> spleen, alexandrakis eta al. (2005)
        mu_s_alex(629, 1.05), # 17 --> pancreas --> liver and spleen, alexandrakis eta al. (2005)
        mu_s_alex(629, 1.05), # 18 --> liver, alexandrakis eta al. (2005)
        mu_s_alex(41700, 1.51), # 19 --> kidneys, alexandrakis eta al. (2005)
        mu_s_alex(4e7, 2.82), # 20 --> adrenal glands --> muscle, alexandrakis eta al. (2005)
        mu_s_alex(68.4, 0.53) # 21 --> lungs, alexandrakis eta al. (2005)
        ]) # [m^-1]
        scattering_coefficients /= (1 - 0.9) # reduced scattering -> scattering, g = 0.9
        scattering_coefficients[0] = coupling_medium_mu_s

        # assign optical properties to the volume
        volume = np.zeros(([2, nx, ny, nz]), dtype=np.float32)
        
        volume[0] = absorption_coefficients[tissue_types] # [m^-1]
        volume[1] = scattering_coefficients[tissue_types] # [m^-1]
        
        return (volume, bg_mask)
