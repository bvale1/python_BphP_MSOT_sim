import numpy as np
from abc import ABC, abstractmethod
import os


class phantom:
    def __init__(self):
        self.path = os.path.dirname(os.path.realpath(__file__))
    
    
    @abstractmethod
    def create_volume(self, cfg : dict):
        pass


    # call any chromophore's needed for the phantom
    def define_water(self) -> dict:
        # temperature coefficients https://pubs.acs.org/doi/10.1021/jp010093m
        temp = 34; # [celcius]
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
        return np.interp(wavelengths_interp, wavelengths, property).tolist()


    def define_ReBphP_PCM(self, wavelengths_interp: (list, np.ndarray)) -> dict:
        # (m^2 mol^-1) = (mm^-1 M^-1) = (mm^-1 mol^-1 dm^3) = (mm^-1 mol^-1 L^3)
        wavelengths_interp = np.asarray(wavelengths_interp) * 1e9 # [m] -> [nm]
        
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
                    epsilon_a_Pr, wavelengths_Pr, wavelengths_interp
                ), # molar absorption coefficient [M^-1 cm^-1]=[m^2 mol^-1]
                'eta' : [0.0045, 0.0] # photoisomerisation quantum yield (dimensionless)
                },
            'Pfr' : { # Far-red absorbing form
                'epsilon_a': self.interp_property(
                    epsilon_a_Pfr, wavelengths_Pfr, wavelengths_interp
                ), # molar absorption coefficient [M^-1 cm^-1]=[m^2 mol^-1]
                'eta' : [0.0, 0.00195] # photoisomerisation quantum yield (dimensionless)
            }   
        }
        return self.ReBphP_PCM
    
    
    def define_water89_gelatin1_intralipid10(self, wavelengths_interp : (list, np.ndarray)) -> dict:
        # Hanna Jonasson et al. 2022. Water and hemoglobin modulated gelatin-based
        # phantoms to spectrally mimic inflamed tissue in the
        # validation of biomedical techniques and the modeling
        # of microdialysis data
        wavelengths_interp = np.asarray(wavelengths_interp) * 1e9 # [m] -> [nm]

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
                mu_a, wavelengths_mu_a, wavelengths_interp
            ), # [m^-1]
            'mu_s' : self.interp_property(
                mu_s, wavelengths_mu_s, wavelengths_interp
            ), # [m^-1]
            'n' : [1.33, 1.33], # refractive index
            'g' : [0.9, 0.9] # anisotropy
        }
        return self.water89_gelatin1_intralipid10


    def define_agarose_gel(self, wavelengths : (list, np.ndarray)) -> dict:
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
    
    
    def define_intralipid_10(self, wavelengths : (list, np.ndarray)) -> dict:
        # Light scattering in Intralipid-10% in the wavelength range of 400-1100 nm
        # Hugo J. van Staveren. 1991
        # https://opg.optica.org/ao/fulltext.cfm?uri=ao-30-31-4507&id=39328
        # TODO: find mu_a and make sense of the units [mL L^-1 mm^-1]
        wavelengths = np.asarray(wavelengths) * 1e6 # [m] -> [um]
        self.intralipid10 = {
            'mu_a' : [0.0, 0.0], # [m^-1]
            'mu_s' : (0.016 * (wavelengths**(-2.4)) * 1e-6).tolist(), # [mL L^-1 mm^-1]
            'n' : 1.33 * np.ones_like(wavelengths), # refractive index
            'g' : (1.1 - (0.58*wavelengths)).tolist() # anisotropy
        }
        return self.intralipid10
    
    
    #TODO: define other chromophores