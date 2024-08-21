import numpy as np
from phantoms.phantom import phantom
from skimage.transform import resize
from typing import Union


class digimouse_phantom(phantom):
    
    def __init__(self, digimouse_path : str, *args, **kwargs):
        super(digimouse_phantom, self).__init__(*args, **kwargs)
        self.digimouse_path = digimouse_path
    
    
    def linear_mixing_law_mu_a(self, S_B, x, S_W) -> np.ndarray:
        # blood volume fraction S_B, oxygen saturation x, water volume fraction S_W.
        # note that the equation in the paper contains a typo, mu_a_HbO2 and mu_a_Hb are the wrong way around.
        
        # I didn't include other molecular absorbers (e.g. fat and melanin) in the equation as it was 
        # couldn't find data on them for most of the tissues in the digimouse phantom.
        
        # This model obviously underestimates the absorption of most tissues, for example bone, which is not vascularized.
        # Using this, all I could see was the liver and lungs as they contain a lot of blood,
        # thier predicted absorption coefficients are ~5 times higher than the other tissues!
        
        # I decided to map S_B via the following
        S_B -= 0.5 * (S_B-0.1)
        # this is arbitrary but it reduces the contrast between tissue types,
        # so the simulated images more closely resemble real images of mice taken using the MSOT.
        return S_B*(x*np.asarray(self.HbO2['mu_a'])+(1-x)*np.asarray(self.Hb['mu_a'])) + S_W*np.asarray(self.H2O['mu_a'])
    
    def power_law_alex_mu_s(self, a, b) -> np.ndarray:
        # power law function for reduced scattering coefficient
        # alexandrakis eta al. (2005)
        # assume anisotropy factor g = 0.9
        g = 0.9
        return a * (self.wavelengths_nm**(-b)) * 1e3 / (1-g) # [m^-1]
    
    def power_law_jac_mu_s(self, a, b) -> np.ndarray:
        # power law function for reduced scattering coefficient
        # Steven L Jacques (2013)
        # assume anisotropy factor g = 0.9
        g = 0.9
        return a * ((self.wavelengths_nm/500)**(-b)) * 1e3 / (1-g)
        
    def calculate_tissue_absorption_coefficients(self) -> np.ndarray:
        # I didn't include other molecular absorbers (e.g. fat and melanin) in the equation as it was 
        # couldn't find data on them for most of the tissues in the digimouse phantom.
        
        # These are the fractions of blood and water, and oxygen saturation compiled from the literature
        return np.array([
            np.asarray(self.H2O['mu_a']), # 0 --> background
            self.linear_mixing_law_mu_a(0.0033, 0.7, 0.5), # 1 --> skin --> adipose, alexandrakis eta al. (2005)
            self.linear_mixing_law_mu_a(0.049, 0.8, 0.15), # 2 --> skeleton, alexandrakis eta al. (2005)
            self.linear_mixing_law_mu_a(0.0033, 0.7, 0.5), # 3 --> eye --> adipose, alexandrakis eta al. (2005)
            self.linear_mixing_law_mu_a(0.03, 0.6, 0.75), # 4 --> medulla --> Rat brain cortex, Steven L Jacques (2013)
            self.linear_mixing_law_mu_a(0.03, 0.6, 0.75), # 5 --> cerebellum --> Rat brain cortex, Steven L Jacques (2013)
            self.linear_mixing_law_mu_a(0.03, 0.6, 0.75), # 6 --> olfactory bulbs --> Rat brain cortex, Steven L Jacques (2013)
            self.linear_mixing_law_mu_a(0.03, 0.6, 0.75), # 7 --> external cerebrum --> Rat brain cortex, Steven L Jacques (2013)
            self.linear_mixing_law_mu_a(0.03, 0.6, 0.75), # 8 --> striatum --> Rat brain cortex, Steven L Jacques (2013)
            self.linear_mixing_law_mu_a(0.05, 0.75, 0.5), # 9 --> heart, alexandrakis eta al. (2005)
            self.linear_mixing_law_mu_a(0.03, 0.6, 0.75), # 10 --> rest of the brain --> Rat brain cortex, Steven L Jacques (2013)
            self.linear_mixing_law_mu_a(0.07, 0.8, 0.5), # 11 --> masseter muscles, alexandrakis eta al. (2005)
            self.linear_mixing_law_mu_a(0.0033, 0.7, 0.5), # 12 --> lachrymal glands --> adipose, alexandrakis eta al. (2005)
            np.asarray(self.H2O['mu_a']), # 13 --> bladder --> water, Hendrik Buiteveld (1994)
            self.linear_mixing_law_mu_a(0.07, 0.8, 0.5), # 14 --> testis --> muscle, alexandrakis eta al. (2005)
            self.linear_mixing_law_mu_a(0.01, 0.7, 0.8), # 15 --> stomach, alexandrakis eta al. (2005)
            self.linear_mixing_law_mu_a(0.3, 0.75, 0.7), # 16 --> spleen, alexandrakis eta al. (2005)
            self.linear_mixing_law_mu_a(0.3, 0.75, 0.7), # 17 --> pancreas --> liver & spleen, alexandrakis eta al. (2005)
            self.linear_mixing_law_mu_a(0.3, 0.75, 0.7), # 18 --> liver, alexandrakis eta al. (2005)
            self.linear_mixing_law_mu_a(0.056, 0.75, 0.8),  # 19 --> kidneys, alexandrakis eta al. (2005)
            self.linear_mixing_law_mu_a(0.07, 0.8, 0.5), # 20 --> adrenal glands --> muscle, alexandrakis eta al. (2005)
            self.linear_mixing_law_mu_a(0.15, 0.85, 0.85) # 21 --> lungs, alexandrakis eta al. (2005)
        ]) # [m^-1]
               
    def calculate_tissue_scattering_coefficients(self) -> np.ndarray:
        return np.array([
            np.asarray(self.H2O['mu_s']), # 0 --> background
            self.power_law_alex_mu_s(38, 0.53), # 1 --> skin --> adipose, alexandrakis eta al. (2005)
            self.power_law_alex_mu_s(35600, 1.47), # 2 --> skeleton, alexandrakis eta al. (2005)
            self.power_law_alex_mu_s(38, 0.53), # 3 --> eye --> adipose, alexandrakis eta al. (2005)
            self.power_law_jac_mu_s(2.14, 1.2), # 4 --> medulla --> brain, Steven L Jacques (2013)
            self.power_law_jac_mu_s(2.14, 1.2), # 5 --> cerebellum --> brain, Steven L Jacques (2013)
            self.power_law_jac_mu_s(2.14, 1.2), # 6 --> olfactory bulbs --> brain, Steven L Jacques (2013)
            self.power_law_jac_mu_s(2.14, 1.2), # 7 --> external cerebrum --> brain, Steven L Jacques (2013)
            self.power_law_jac_mu_s(2.14, 1.2), # 8 --> striatum --> brain, Steven L Jacques (2013)
            self.power_law_alex_mu_s(10600, 1.43), # 9 --> heart, alexandrakis eta al. (2005)
            self.power_law_alex_mu_s(2.14, 1.2), # 10 --> rest of the brain --> brain, Steven L Jacques (2013)
            self.power_law_alex_mu_s(4e7, 2.82), # 11 --> masseter muscles, alexandrakis eta al. (2005)
            self.power_law_alex_mu_s(38, 0.53), # 12 --> lachrymal glands --> adipose, alexandrakis eta al. (2005)
            np.asarray(self.H2O['mu_s']), # 13 --> bladder --> water, Hendrik Buiteveld (1994)
            self.power_law_alex_mu_s(4e7, 2.82), # 14 --> testis --> muscle, alexandrakis eta al. (2005)
            self.power_law_alex_mu_s(792, 0.97), # 15 --> stomach, alexandrakis eta al. (2005)
            self.power_law_alex_mu_s(629, 1.05), # 16 --> spleen, alexandrakis eta al. (2005)
            self.power_law_alex_mu_s(629, 1.05), # 17 --> pancreas --> liver and spleen, alexandrakis eta al. (2005)
            self.power_law_alex_mu_s(629, 1.05), # 18 --> liver, alexandrakis eta al. (2005)
            self.power_law_alex_mu_s(41700, 1.51), # 19 --> kidneys, alexandrakis eta al. (2005)
            self.power_law_alex_mu_s(4e7, 2.82), # 20 --> adrenal glands --> muscle, alexandrakis eta al. (2005)
            self.power_law_alex_mu_s(68.4, 0.53) # 21 --> lungs, alexandrakis eta al. (2005)
        ]) # [m^-1]
        
    def create_volume(self,
                      cfg: dict, 
                      y_idx : int,
                      rotate : int,
                      axisymmetric : bool = False
                      ) -> tuple[np.ndarray, np.ndarray]:
        assert rotate in [0, 1, 2, 3], 'Rotation must be 0, 1, 2 or 3 corresponding to 0, pi/2, pi and 3*pi/2 respectively'
        assert 100 <= y_idx <= 875, 'y_idx must be between 100 and 875' 
        
        # load the digimouse phantom
        with open(self.digimouse_path, 'rb') as f:
            digimouse = np.fromfile(f, dtype=np.int8)
        digimouse = digimouse.reshape(380, 992, 208, order='F')
        
        # spatial dimensions of the digimouse phantom
        dx = 0.0001 # [m]
        digimouse = resize(
            digimouse, (208, 992, 208), order=0, preserve_range=True, anti_aliasing=False
        )
        # interpolate to mcx grid dx
        new_shape = (np.array(digimouse.shape) * dx / cfg['dx']).astype(int)
        digimouse = resize(
            digimouse, new_shape, order=0, preserve_range=True, anti_aliasing=False
        )
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
        
        if axisymmetric:
            tissue_types = np.repeat(tissue_types[:, [(ny//2)-1], :], ny, axis=1)

        # background mask
        bg_mask = tissue_types[:,(ny//2)-1,:] != 0

        absorption_coefficients = self.calculate_tissue_absorption_coefficients()
        scattering_coefficients = self.calculate_tissue_scattering_coefficients()

        # assign optical properties to the volume
        volume = np.zeros(([2, nx, ny, nz]), dtype=np.float32)
        
        volume[0] = absorption_coefficients[tissue_types, 0] # [m^-1]
        volume[1] = scattering_coefficients[tissue_types, 0] # [m^-1]
        
        return (volume, bg_mask)
    
    
    def get_tissue_types_dict(self) -> dict:
        # This function returns a dictionary of the absorption and scattering 
        # coefficients of each tissue type as function of wavelength 
        
        tissue_types_dict = {'background': {}, 'adipose': {}, 'skeleton': {}, 'eye': {}, 'medulla': {}, 
                             'cerebellum': {}, 'olfactory bulbs': {}, 'external cerebrum': {}, 
                             'striatum': {}, 'heart': {}, 'rest of the brain': {}, 'masseter muscles': {}, 
                             'lachrymal glands': {}, 'bladder': {}, 'testis': {}, 'stomach': {}, 
                             'spleen': {}, 'pancreas': {}, 'liver': {}, 'kidneys': {}, 
                             'adrenal glands': {}, 'lungs': {}}
        keys = list(tissue_types_dict.keys())
        for i, tissue_type in enumerate(self.calculate_tissue_absorption_coefficients()):
            tissue_types_dict[keys[i]]['mu_a'] = tissue_type
        for i, tissue_type in enumerate(self.calculate_tissue_scattering_coefficients()):
            tissue_types_dict[keys[i]]['mu_s'] = tissue_type
        
        return tissue_types_dict
        