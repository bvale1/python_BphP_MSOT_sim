import numpy as np
from abc import ABC, abstractmethod


class phantom:
    def __init__(self):
        pass
    
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
    
    def define_ReBphP_PCM(self) -> dict:
        # properties of the bacterial phytochrome
        self.ReBphP_PCM = {
            'Pr' : { # Red absorbing form
                'epsilon_a': [0.8e4, 0.05e4], # molar absorption coefficient [M^-1 cm^-1]=[m^2 mol^-1]
                'eta' : [0.01, 0.0] # photoisomerisation quantum yield (dimensionless)
                },
            'Pfr' : { # Far-red absorbing form
                'epsilon_a': [0.6e4, 0.8e4], # molar absorption coefficient [M^-1 cm^-1]=[m^2 mol^-1]
                'eta' : [0.015, 0.0] # photoisomerisation quantum yield (dimensionless)
            }   
        }
        return self.ReBphP_PCM
    
    #TODO: define other chromophores