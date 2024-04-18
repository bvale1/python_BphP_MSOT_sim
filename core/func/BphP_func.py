import numpy as np

# Please refer to research notes for documentation on the equations used here

def steady_state_BphP(Pr, Pfr, wavelength_idx):
    # return the ratio of Pr to Pfr at steady state given the wavelength
    # such that Pr_frac + Pfr_frac = 1
    a = (Pr['epsilon_a'][wavelength_idx] * Pr['eta'][wavelength_idx] +
        Pfr['epsilon_a'][wavelength_idx] * Pfr['eta'][wavelength_idx])

    Pr_frac = Pfr['epsilon_a'][wavelength_idx] * Pfr['eta'][wavelength_idx] / a
    Pfr_frac = Pr['epsilon_a'][wavelength_idx] * Pr['eta'][wavelength_idx] / a
    
    return [Pr_frac, Pfr_frac]


def switch_BphP(Pr, Pfr, Pr_c, Pfr_c, Phi, wavelengths, wavelength_idx):
    
    c_tot = Pr_c + Pfr_c
    
    a = (Pr['epsilon_a'][wavelength_idx] * Pr['eta'][wavelength_idx] +
        Pfr['epsilon_a'][wavelength_idx] * Pfr['eta'][wavelength_idx])
    
    # molar fluence [mols m^-2] = lambda * Phi / (h * c * N_A)
    Phi_mols = 8.3593472259 * wavelengths[wavelength_idx] * Phi
    
    # dimensionless exponential term
    exponential = np.exp(-a * Phi_mols)
    
    k_Pr = Pfr['epsilon_a'][wavelength_idx] * Pfr['eta'][wavelength_idx] / a
    Pr_c = k_Pr * c_tot * (1 - exponential) + Pr_c * exponential
    
    k_Pfr = Pr['epsilon_a'][wavelength_idx] * Pr['eta'][wavelength_idx] / a
    Pfr_c = k_Pfr * c_tot * (1 - exponential) + Pfr_c * exponential
    
    return [Pr_c, Pfr_c]
