import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Union


def define_ReBphP_PCM(phantoms_path, wavelengths_interp: Union[list, np.ndarray]) -> dict:
    # (m^2 mol^-1) = (mm^-1 M^-1) = (mm^-1 mol^-1 dm^3) = (mm^-1 mol^-1 L^3)
    wavelengths_interp = np.asarray(wavelengths_interp) * 1e9 # [m] -> [nm]
    # ignore first line, load both columns into numpy array
    with open(phantoms_path+'/Chromophores/epsilon_a_ReBphP_PCM_Pr.txt', 'r') as f:
        data = np.genfromtxt(f, skip_header=1, dtype=np.float32, delimiter=', ')
    wavelengths_Pr = data[:,0] # [nm]
    epsilon_a_Pr = data[:,1] * 1e4 # [1e5 M^-1 cm^-1] -> [M^-1 mm^-1]
    # sort to wavelength descending order
    sort_index = wavelengths_Pr.argsort()
    wavelengths_Pr = wavelengths_Pr[sort_index]
    epsilon_a_Pr = epsilon_a_Pr[sort_index]    
    
    with open(phantoms_path+'/Chromophores/epsilon_a_ReBphP_PCM_Pfr.txt', 'r') as f:
        data = np.genfromtxt(f, skip_header=1, dtype=np.float32, delimiter=', ')
    wavelengths_Pfr = data[:,0] # [nm]
    epsilon_a_Pfr = data[:,1] * 1e4 # [1e5 M^-1 cm^-1] -> [M^-1 mm^-1]
    # sort to wavelength descending order
    sort_index = wavelengths_Pfr.argsort()
    wavelengths_Pfr = wavelengths_Pfr[sort_index]
    epsilon_a_Pfr = epsilon_a_Pfr[sort_index]
    
        
    # properties of the bacterial phytochrome
    ReBphP_PCM = {
        'Pr' : { # Red absorbing form
            'epsilon_a': np.interp(
                wavelengths_interp, wavelengths_Pr, epsilon_a_Pr
            ).tolist(), # molar absorption coefficient [M^-1 cm^-1]=[m^2 mol^-1]
            'eta' : [0.03, 0.0] # photoisomerisation quantum yield (dimensionless)
            },
        'Pfr' : { # Far-red absorbing form
            'epsilon_a': np.interp(
                wavelengths_interp, wavelengths_Pfr, epsilon_a_Pfr
            ).tolist(), # molar absorption coefficient [M^-1 cm^-1]=[m^2 mol^-1]
            'eta' : [0.0, 0.005] # photoisomerisation quantum yield (dimensionless)
        }   
    }
    return ReBphP_PCM

def load_raw_ReBphP_PCM(phantoms_path):
    # ignore first line, load both columns into numpy array
    with open(phantoms_path+'/Chromophores/epsilon_a_ReBphP_PCM_Pr.txt', 'r') as f:
        data_Pr = np.genfromtxt(f, skip_header=1, dtype=np.float32, delimiter=', ')
    data_Pr[:,1] = data_Pr[:,1] * 1e4 # [1e5 M^-1 cm^-1] -> [M^-1 mm^-1]
    
    with open(phantoms_path+'/Chromophores/epsilon_a_ReBphP_PCM_Pfr.txt', 'r') as f:
        data_Pfr = np.genfromtxt(f, skip_header=1, dtype=np.float32, delimiter=', ')
    data_Pfr[:,1] = data_Pfr[:,1] * 1e4 # [1e5 M^-1 cm^-1] -> [M^-1 mm^-1]
    
    return [data_Pr, data_Pfr]


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    #font = {'size'   : 5}
    #plt.rc('font', **font)
    
    wavelengths = np.linspace(550e-9, 800e-9, num=2000)
    ReBphP_PCM = define_ReBphP_PCM(os.path.dirname(os.path.realpath(__file__)), wavelengths)
    
    for Pr_ratio in range(0, 11, 2):
        Pr_ratio *= 0.1
        print(Pr_ratio)
        epsilon_a = (np.asarray(ReBphP_PCM['Pr']['epsilon_a']) * Pr_ratio +
                     np.asarray(ReBphP_PCM['Pfr']['epsilon_a']) * (1-Pr_ratio))
        label = str(f'{round(5*Pr_ratio)}:{round(5*(1-Pr_ratio))}')
        molar_concentration = 1e-8 # [M] assume molar concentration of 1e-8 M
        mu_a = epsilon_a * molar_concentration # [M^-1 mm^-1 * M -> mm^-1]
        mu_a *= 10 # [mm^-1 -> cm^-1]
        ax.plot(
            wavelengths * 1e9, # [m -> nm]
            mu_a * 1e3, # [cm^-1 -> 1e-3 cm^-1]
            color=(
                (26/256)*Pr_ratio + (212/256)*(1-Pr_ratio), # r [0.0 to 1.0]
                (133/256)*Pr_ratio + (17/256)*(1-Pr_ratio), # g [0.0 to 1.0]
                (255/256)*Pr_ratio + (89/256)*(1-Pr_ratio)  # b [0.0 to 1.0]
            ),
            label=label
        )
    
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(title=r'ratio (Pr:Pfr)')
    #fig.suptitle('ReBphP PCM', fontsize='xx-large')
    ax.set_ylabel(r'$\mu_{\mathdefault{a}}^{(\mathdefault{p})}$ ($10^{-3}$ cm$^{^-1})$')
    ax.set_xlabel(r'$\lambda$ (nm)')
    fig.tight_layout()
    plt.savefig('ReBphP_PCM_photoswitching.svg', format='svg', dpi=300)
    