import numpy as np
from abc import abstractmethod
import os, logging
from typing import Union


class phantom:
    
    def __init__(self, wavelengths_m : Union[list, np.ndarray, tuple]):
        # gets directory of the phantom file, used to load chromophores
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.wavelengths_m = np.asarray(wavelengths_m)
        self.wavelengths_interp = np.asarray(wavelengths_m) * 1e9 # [m] -> [nm]
        self.wavelengths_nm = self.wavelengths_interp
    
    @abstractmethod
    def create_volume(self, cfg : dict):
        pass

    # methods to define chromophores needed for the phantom
    
    # depricated
    def define_water(self) -> dict:
        logging.info('WARNING: define_water() is depricated. Use define_H2O() instead.')
        # temperature coefficients https://pubs.acs.org/doi/10.1021/jp010093m
        temp = 34 # [celcius]
        # absorption 380nm-700nm https://opg.optica.org/ao/viewmedia.cfm?uri=ao-38-7-1216&seq=0
        # absorption and scattering 300nm-800nm Optical properties of pure water Hendrik Buiteveld
        # https://www.spiedigitallibrary.org/conference-proceedings-of-spie/2258/0000/Optical-properties-of-pure-water/10.1117/12.190060.full
        self.H2O = {
            'mu_a' : [ 
                0.4318 + (temp - 20) * (-5e-5), # [m^-1]
                2.7542 + (temp - 20) * (-152e-5) # [m^-1]
            ],
            'mu_s' : [0.0006, 0.0003], # [m^-1],
            'n' : [1.33, 1.33], # refractive index
            'g' : [0.9, 0.9] # anisotropy
        }
        return self.H2O

    @classmethod
    def interp_property(cls, property, wavelengths, wavelengths_interp) -> list:
        # sort to wavelength ascending order
        sort_index = wavelengths.argsort()
        wavelengths = wavelengths[sort_index]
        property = property[sort_index]
        # interpolate to wavelengths_interp
        return np.interp(wavelengths_interp, wavelengths, property).reshape(-1).tolist()


    def define_ReBphP_PCM(self) -> dict:
        # (m^2 mol^-1) = (mm^-1 M^-1) = (mm^-1 mol^-1 dm^3) = (mm^-1 mol^-1 L^3)
        
        # ignore first line, load both columns into numpy array
        with open(self.path+'/Chromophores/epsilon_a_ReBphP_PCM_Pr.txt', 'r') as f:
            data = np.genfromtxt(f, skip_header=1, dtype=np.float32, delimiter=', ')
        wavelengths_Pr = data[:,0] # [nm]
        epsilon_a_Pr = data[:,1] * 1e4 # [1e5 M^-1 cm^-1] -> [M^-1 mm^-1]
        
        with open(self.path+'/Chromophores/epsilon_a_ReBphP_PCM_Pfr.txt', 'r') as f:
            data = np.genfromtxt(f, skip_header=1, dtype=np.float32, delimiter=', ')
        wavelengths_Pfr = data[:,0] # [nm]
        epsilon_a_Pfr = data[:,1] * 1e4 # [1e5 M^-1 cm^-1] -> [M^-1 mm^-1]        
        # properties of the bacterial phytochrome
        self.ReBphP_PCM = {
            'Pr' : { # Red absorbing form
                'epsilon_a': self.interp_property(
                    epsilon_a_Pr, wavelengths_Pr, self.wavelengths_interp
                ), # molar absorption coefficient [M^-1 cm^-1]=[m^2 mol^-1]
                'eta' : [0.004, 0.0] # photoisomerisation quantum yield (dimensionless)
                },
            'Pfr' : { # Far-red absorbing form
                'epsilon_a': self.interp_property(
                    epsilon_a_Pfr, wavelengths_Pfr, self.wavelengths_interp
                ), # molar absorption coefficient [M^-1 cm^-1]=[m^2 mol^-1]
                'eta' : [0.0, 0.002] # photoisomerisation quantum yield (dimensionless)
            }   
        }
        return self.ReBphP_PCM
    
    
    def define_water89_gelatin1_intralipid10(self) -> dict:
        # Hanna Jonasson et al. 2022. Water and hemoglobin modulated gelatin-based
        # phantoms to spectrally mimic inflamed tissue in the
        # validation of biomedical techniques and the modeling
        # of microdialysis data

        with open(self.path+'/Chromophores/mu_a_water89_gelatin1_intralipid10.txt', 'r') as f:
            data = np.genfromtxt(f, skip_header=1, dtype=np.float32, delimiter=', ')
        wavelengths_mu_a = data[:,0] # [nm]
        mu_a = data[:,1] * 1e3 # [mm^-1] -> [m^-1]

        with open(self.path+'/Chromophores/mu_s_water89_gelatin1_intralipid10.txt', 'r') as f:
            data = np.genfromtxt(f, skip_header=1, dtype=np.float32, delimiter=', ')
        wavelengths_mu_s = data[:,0] # [nm]
        mu_s = data[:,1] * 1e3 # [mm^-1] -> [m^-1]

        self.water89_gelatin1_intralipid10 = {
            'mu_a' : self.interp_property(
                mu_a, wavelengths_mu_a, self.wavelengths_interp
            ), # [m^-1]
            'mu_s' : self.interp_property(
                mu_s, wavelengths_mu_s, self.wavelengths_interp
            ), # [m^-1]
            'n' : [1.33, 1.33], # refractive index
            'g' : [0.9, 0.9] # anisotropy
        }
        logging.info(f'water89_gelatin1_intralipid10 optical properties: {self.water89_gelatin1_intralipid10}')
        return self.water89_gelatin1_intralipid10

    # not in use
    def define_agarose_gel(self) -> dict:
        # TODO: find optical properties of agarose gel
        # Afrina Mustari et al. 2018. Agarose-based Tissue Mimicking Optical
        # Phantoms for Diffuse Reflectance Spectroscopy
        self.agarose_gel = {
            'mu_a' : [0.0, 0.0], # [m^-1]
            'mu_s' : [0.0, 0.0], # [m^-1]
            'n' : [1.33, 1.33], # refractive index
            'g' : [0.9, 0.9] # anisotropy
        }
        return self.agarose_gel    
    
    # not in use
    def define_intralipid_10(self)-> dict:
        # Light scattering in Intralipid-10% in the wavelength range of 400-1100 nm
        # Hugo J. van Staveren. 1991
        # https://opg.optica.org/ao/fulltext.cfm?uri=ao-30-31-4507&id=39328
        # TODO: find mu_a and make sense of the units [mL L^-1 mm^-1]
        wavelengths_um = np.asarray(self.wavelengths_m) * 1e6 # [m] -> [um]
        self.intralipid10 = {
            'mu_a' : [0.0, 0.0], # [m^-1]
            'mu_s' : (0.016 * (wavelengths_um**(-2.4)) * 1e-6).tolist(), # [mL L^-1 mm^-1]
            'n' : 1.33 * np.ones_like(wavelengths_um), # refractive index
            'g' : (1.1 - (0.58*wavelengths_um)).tolist() # anisotropy
        }
        return self.intralipid10
    
    def define_Hb(self) -> dict:
        # Optical properties of biological tissues: a review Steven L Jacques, 2013
        # https://iopscience.iop.org/article/10.1088/0031-9155/58/11/R37/meta
        
        with open(self.path+'/Chromophores/mu_a_HbO2.txt', 'r') as f:
            data_HbO2 = np.genfromtxt(f, skip_header=1, dtype=np.float32, delimiter=', ')
        wavelengths_HbO2 = data_HbO2[:,0] # [nm]
        mu_a_HbO2 = data_HbO2[:,1] * 1e2 # [cm^-1] -> [m^-1]
        
        with open(self.path+'/Chromophores/mu_a_Hb.txt', 'r') as f:
            data_Hb = np.genfromtxt(f, skip_header=1, dtype=np.float32, delimiter=', ')
        wavelengths_Hb = data_Hb[:,0] # [nm]
        mu_a_Hb = data_Hb[:,1] * 1e2

        with open(self.path+'/Chromophores/blood_litcomp_dat.txt', 'r') as f:
            data_mu_s = np.genfromtxt(f, skip_header=1, dtype=np.float32, delimiter=' ')
        wavelengths_mu_s = data_mu_s[:,0] # [nm]
        mu_s = data_mu_s[:,3] * 1e3 # [mm^-1] -> [m^-1]
        g = data_mu_s[:,4] # anisotropy
        
        self.Hb = {
            'mu_a' : self.interp_property(
                mu_a_Hb, wavelengths_Hb, self.wavelengths_interp
            ), # [m^-1]
            'mu_s' : self.interp_property(
                mu_s, wavelengths_mu_s, self.wavelengths_interp
            ), # [m^-1]
            'n' : 1.33, # refractive index
            'g' : self.interp_property(
                g, wavelengths_mu_s, self.wavelengths_interp
            ) # anisotropy
        }
        self.HbO2 = {
            'mu_a' : self.interp_property(
                mu_a_HbO2, wavelengths_HbO2, self.wavelengths_interp
            ), # [m^-1]
            'mu_s' : self.interp_property(
                mu_s, wavelengths_mu_s, self.wavelengths_interp
            ), # [m^-1]
            'n' : 1.33, # refractive index
            'g' : self.interp_property(
                g, wavelengths_mu_s, self.wavelengths_interp
            ) # anisotropy
        }
        return self.Hb, self.HbO2
    
    
    def define_H2O(self) -> dict:
        # (Hale and Querry 1973 Optical Constants of Water in the 200-nm to 200-mu m Wavelength Region)
        # https://opg.optica.org/ao/fulltext.cfm?uri=ao-12-3-555&id=17737
        # this data is only for H2O at 25 degrees celcius
        # this does not include scattering data, although scattering of water is
        # very low in this wavelength range (mu_s < 1e-3 m^-1 for wavelengths > 600nm)
        # see Hendrik Buiteveld 1998 for scattering
        
        if any(self.wavelengths_interp < 400) or any(self.wavelengths_interp > 1300):
            logging.info(f'Warning: H2O optical data is only from 400nm to 1300nm, wavelengths requested:{self.wavelengths_interp}')
            
        with open(self.path+'/Chromophores/hale_and_querry_h20.txt', 'r') as f:
            data = np.genfromtxt(f, skip_header=1, dtype=np.float32, delimiter='\t')
        wavelengths_nm = data[:,0]
        mu_a = data[:,1] * 1e2 # [cm^-1] -> [m^-1]
        
        self.H2O = {
            'mu_a' : self.interp_property(
                mu_a, wavelengths_nm, self.wavelengths_interp 
            ), # [m^-1]
            'mu_s' : np.zeros(len(self.wavelengths_interp), dtype=np.float32).tolist(), # [m^-1]
            'n' : 1.33, # refractive index
            'g' : 0.9 # anisotropy
        }
        return self.H2O
    
    '''
    # depricated in favor of hale and querry 1973 due to greater wavelength range
    def define_H2O(self,
                   wavelengths_m : Union[list, np.ndarray],
                   temp : float = 34 # [celcius]
                   ) -> dict:
        wavelengths_interp = np.asarray(wavelengths_m) * 1e9 # [m] -> [nm]
        if any(wavelengths_interp > 800e-9) or any(wavelengths_interp < 300e-9):
            logging.info('Warning: H2O optical data is only 300nm and 800nm')
        
        # absorption and scattering 300nm-800nm Optical properties of pure water Hendrik Buiteveld
        # https://www.spiedigitallibrary.org/conference-proceedings-of-spie/2258/0000/Optical-properties-of-pure-water/10.1117/12.190060.full
        with open(self.path+'/Chromophores/coefficients_H2O.txt', 'r', encoding='latin1') as f:
            data = np.genfromtxt(f, skip_header=1, dtype=np.float32, delimiter=', ', comments=None)
        data[np.isnan(data)] = 0.0
        wavelengths_nm = data[:,0]
        mu_a = data[:,1] + (temp - 20.1) * data[:,2] # (a + A(T-20.1)) [m^-1]
        mu_s = data[:,3] # [m^-1]
        
        self.H2O = {
            'mu_a' : self.interp_property(
                mu_a, wavelengths_nm, wavelengths_interp 
            ), # [m^-1]
            'mu_s' : self.interp_property(
                mu_s, wavelengths_nm, wavelengths_interp
            ), # [m^-1]
            'n' : 1.33, # refractive index
            'g' : 0.9 # anisotropy
        }
        return self.H2O
    '''