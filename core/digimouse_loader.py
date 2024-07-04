import numpy as np
import matplotlib.pyplot as plt
import func.plot_func as pf
from scipy.ndimage import zoom
from phantoms.phantom import phantom

# this is a test script I used to implement and visualise the digimouse phantom

path = '/home/wv00017/digimouse_atlas/atlas_380x992x208.img'

# load the img file
with open(path, 'rb') as f:
    digimouse = np.fromfile(f, dtype=np.int8)

print(f'shape {digimouse.shape}')

digimouse = digimouse.reshape(380, 992, 208, order='F')

if __name__ == '__main__':
    
    dx = 0.0001
    domain = np.array([380, 992, 208]) * dx
    new_domain = np.array([308, 992, 380]) * dx
    
    grid = np.array([308, 992, 208])
    new_grid = np.array([308, 992, 380])
    zoom_factors = [n / o for n, o in zip(new_grid, grid)]
    digimouse = zoom(digimouse, zoom_factors, order=0)
    
    cross_sections = np.transpose(digimouse[:,200:875:25,:], axes=(1,2,0))
    pf.heatmap(cross_sections, dx=dx, sharescale=True, title='tissue types', cmap='tab20')
    
    # translate digimouse so the centre of mass along the xz plane
    # at y_ydx is in the centre
    #x = np.arange(digimouse.shape[0]) - digimouse.shape[0]//2
    #z = np.arange(digimouse.shape[2])[np.newaxis,:] - digimouse.shape[2]//2
    #for cross_section in cross_sections:
    #    COM_x = np.sum(digimouse[:,:]!=0)*x / (digimouse.shape[0]*digimouse.shape[2])
    #    COM_z = np.sum(digimouse[:,:]!=0)*z / (digimouse.shape[0]*digimouse.shape[2])
    
    
    bg_masks = cross_sections != 0
    
    coupling_medium_mu_a = 0.1 # [m^-1]
    coupling_medium_mu_s = 100 # [m^-1]
    
    wavelength_nm = 700 # [nm]
    wavelength_m = wavelength_nm * 1e-9 # [m]
    
    # The following are optical properties recommended for use by the authors of digimouse
    # @article{dogdas2007digimouse,
    # title={Digimouse: a 3D whole body mouse atlas from CT and cryosection data},
    # author={Dogdas, Belma and Stout, David and Chatziioannou, Arion F and Leahy, Richard M},
    # journal={Physics in Medicine \& Biology},
    # volume={52},
    # number={3},
    # pages={577},
    # year={2007},
    # publisher={IOP Publishing}
    # }
    
    # These values were collected from 
    # @article{alexandrakis2005tomographic,
    # title={Tomographic bioluminescence imaging by use of a combined optical-PET (OPET) system: a computer simulation feasibility study},
    # author={Alexandrakis, George and Rannou, Fernando R and Chatziioannou, Arion F},
    # journal={Physics in Medicine \& Biology},
    # volume={50},
    # number={17},
    # pages={4225},
    # year={2005},
    # publisher={IOP Publishing}
    # }
    # They are wavelength dependant but unfortunately several tissue types are missing
    
    chromophores_obj = phantom()    
    (Hb, HbO2) = chromophores_obj.define_Hb(wavelength_m)
    mu_a_Hb = Hb['mu_a'][0] * 1e-3 # [m^-1] -> [mm^-1]    
    mu_a_HbO2 = HbO2['mu_a'][0] * 1e-3 # [m^-1] -> [mm^-1]
    mu_a_H2O = chromophores_obj.define_H2O(wavelength_m)['mu_a'][0] * 1e-3 # [m^-1] -> [mm^-1]
    
    # optical absorption coefficient at nm [m^-1]
    mu_a = lambda S_B, x, S_W : (S_B*(x*mu_a_Hb+(1-x)*mu_a_HbO2) + S_W*mu_a_H2O) * 1e3 # [mm^-1] -> [m^-1]
    absorption_coefficients = np.array([
        coupling_medium_mu_a, # 0 --> background
        mu_a(0.06, 0.75, 0.5), # 1 --> skin
        mu_a(0.049, 0.8, 0.15), # 2 --> skeleton
        coupling_medium_mu_a, # 3 --> eye
        coupling_medium_mu_a, # 4 --> medulla --> whole brain 
        coupling_medium_mu_a, # 5 --> cerebellum --> whole brain 
        coupling_medium_mu_a, # 6 --> olfactory bulbs --> whole brain 
        coupling_medium_mu_a, # 7 --> external cerebrum --> whole brain 
        coupling_medium_mu_a, # 8 --> striatum --> whole brain 
        mu_a(0.05, 0.75, 0.5), # 9 --> heart
        coupling_medium_mu_a, # 10 --> rest of the brain --> whole brain 
        mu_a(0.07, 0.8, 0.5), # 11 --> masseter muscles
        coupling_medium_mu_a, # 12 --> lachrymal glands
        coupling_medium_mu_a, # 13 --> bladder
        coupling_medium_mu_a, # 14 --> testis
        mu_a(0.01, 0.7, 0.8), # 15 --> stomach
        mu_a(0.3, 0.75, 0.7), # 16 --> spleen
        coupling_medium_mu_a, # 17 --> pancreas
        mu_a(0.3, 0.75, 0.7), # 18 --> liver
        mu_a(0.056, 0.75, 0.8),  # 19 --> kidneys
        coupling_medium_mu_a, # 20 --> adrenal glands
        mu_a(0.15, 0.85, 0.85) # 21 --> lungs
    ])
    
    
    mu_s = lambda a, b : (a * (wavelength_nm**-b) * 1e3) # [mm^-1] -> [m^-1]
    scattering_coefficients = np.array([
        coupling_medium_mu_s, # 0 --> background
        mu_s(2850, 1.1), # 1 --> skin
        mu_s(35600, 1.47), # 2 --> skeleton
        coupling_medium_mu_s, # 3 --> eye
        coupling_medium_mu_s, # 4 --> medulla --> whole brain 
        coupling_medium_mu_s, # 5 --> cerebellum --> whole brain 
        coupling_medium_mu_s, # 6 --> olfactory bulbs --> whole brain 
        coupling_medium_mu_s, # 7 --> external cerebrum --> whole brain 
        coupling_medium_mu_s, # 8 --> striatum --> whole brain 
        mu_s(10600, 1.43), # 9 --> heart
        coupling_medium_mu_s, # 10 --> rest of the brain --> whole brain 
        mu_s(4e7, 2.82), # 11 --> masseter muscles
        coupling_medium_mu_s, # 12 --> lachrymal glands
        coupling_medium_mu_s, # 13 --> bladder
        coupling_medium_mu_s, # 14 --> testis
        mu_s(792, 0.97), # 15 --> stomach
        mu_s(629, 1.05), # 16 --> spleen
        coupling_medium_mu_s, # 17 --> pancreas
        mu_s(629, 1.05), # 18 --> liver
        mu_s(41700, 1.51), # 19 --> kidneys
        coupling_medium_mu_s, # 20 --> adrenal glands
        mu_s(68.4, 0.53) # 21 --> lungs
    ])
    cross_sections_mu_a = absorption_coefficients[cross_sections]
    cross_sections_mu_s = scattering_coefficients[cross_sections]
    
    # The following optical properties were collected for digimouse by Qianqian
    # Fang, the author of MCX.    
    # He cites the following two papers for the optical properties he uses:
    # @article{cheong1990review,
    # title={A review of the optical properties of biological tissues},
    # author={Cheong, Wai-Fung and Prahl, Scott A and Welch, Ashley J},
    # journal={IEEE journal of quantum electronics},
    # volume={26},
    # number={12},
    # pages={2166--2185},
    # year={1990},
    # publisher={IEEE}
    # }
    
    # @article{strangman2003factors,
    # title={Factors affecting the accuracy of near-infrared spectroscopy concentration calculations for focal changes in oxygenation parameters},
    # author={Strangman, Gary and Franceschini, Maria Angela and Boas, David A},
    # journal={Neuroimage},
    # volume={18},
    # number={4},
    # pages={865--879},
    # year={2003},
    # publisher={Elsevier}
    # }
    
    # Details at https://mcx.space/wiki/index.cgi?MMC/DigimouseMesh 
    # properties from Strangman et al. (2003) are at 830nm.
    # he seems to have used the values for adipose tissue as for skin, which 
    # likely overestimates the absorption and scattering coefficients of adipose
    prop=np.array([
        [0, coupling_medium_mu_a*1e-3, coupling_medium_mu_s*1e-3, 0.9, 1.37], # 0 --> background
        [1, 0.0191, 6.6, 0.9, 1.37], # 1 --> skin --> scalp, Strangman et al. (2003), 830nm
        [2, 0.0136, 8.6, 0.9, 1.37], # 2 --> skeleton --> skull, Strangman et al. (2003), 830nm
        [3, 0.0026, 0.01, 0.9, 1.37], # 3 --> eye --> cerebrospinal fluid, Strangman et al. (2003), 830nm
        [4, 0.0186, 11.1, 0.9, 1.37], # 4 --> medulla --> brain, Strangman et al. (2003), 830nm
        [5, 0.0186, 11.1, 0.9, 1.37], # 5 --> cerebellum --> brain, Strangman et al. (2003), 830nm
        [6, 0.0186, 11.1, 0.9, 1.37], # 6 --> olfactory bulbs --> brain, Strangman et al. (2003), 830nm
        [7, 0.0186, 11.1, 0.9, 1.37], # 7 --> external cerebrum --> brain, Strangman et al. (2003), 830nm
        [8, 0.0186, 11.1, 0.9, 1.37], # 8 --> striatum --> brain, Strangman et al. (2003), 830nm
        [9, 0.0240, 8.9, 0.9, 1.37], # 9 --> heart --> muscle,
        [10, 0.0026, 0.01, 0.9, 1.37], # 10 --> rest of the brain --> cerebrospinal fluid, Strangman et al. (2003), 830nm
        [11, 0.0240, 8.9, 0.9, 1.37], # 11 --> masseter muscles --> muscle,
        [12, 0.0240, 8.9, 0.9, 1.37], # 12 --> lachrymal glands --> muscle,
        [13, 0.0240, 8.9, 0.9, 1.37], # 13 --> bladder --> muscle,
        [14, 0.0240, 8.9, 0.9, 1.37], # 14 --> testis --> muscle,
        [15, 0.0240, 8.9, 0.9, 1.37], # 15 --> stomach --> muscle,
        [16, 0.072, 5.6, 0.9, 1.37], # 16 --> spleen --> liver, Cheong et al. (1990)
        [17, 0.072, 5.6, 0.9, 1.37], # 17 --> pancreas
        [18, 0.072, 5.6, 0.9, 1.37], # 18 --> liver, Cheong et al. (1990)
        [19, 0.050, 5.4, 0.9, 1.37], # 19 --> kidneys --> cow kidney, Cheong et al. (1990), 789nm
        [20, 0.024, 8.9, 0.9, 1.37], # 20 --> adrenal glands --> muscle,
        [21, 0.076, 10.9, 0.9, 1.37] # 21 --> lungs --> pig lung, Cheong et al. (1990), 850nm
    ])
    prop[:, 1] *= 1e3 # [mm^-1] -> [m^-1]
    prop[:, 2] *= 1e3 # [mm^-1] -> [m^-1]
    cross_sections_mu_a = prop[cross_sections,1]
    cross_sections_mu_s = prop[cross_sections,2]
    
    pf.heatmap(cross_sections_mu_a, dx=dx, sharescale=False, title=r'$\mu_{a}$ (m$^{-1}$)')
    pf.heatmap(cross_sections_mu_s, dx=dx, sharescale=False, title=r'$\mu_{s}$ (m$^{-1}$)')
    