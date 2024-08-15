import numpy as np
import matplotlib.pyplot as plt
import func.plot_func as pf
from skimage.transform import resize
from phantoms.phantom import phantom
from phantoms.digimouse_phantom import digimouse_phantom

# this is a test script I used to implement and visualise the digimouse phantom

path = '/home/wv00017/digimouse_atlas/atlas_380x992x208.img'

# load the img file
with open(path, 'rb') as f:
    digimouse = np.fromfile(f, dtype=np.int8)

print(f'shape {digimouse.shape}')

digimouse = digimouse.reshape(380, 992, 208, order='F')

# original digimouse phantom publication
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

if __name__ == '__main__':
    
    dx = 0.0001
    
    grid = np.array([308, 992, 208])
    new_grid = np.array([208, 992, 208])
    digimouse = resize(digimouse, (208, 992, 208), order=0, preserve_range=True, anti_aliasing=False)
    
    cross_sections = np.transpose(digimouse[:,200:875:25,:], axes=(1,2,0))
    # for visualisation, combine all the brain parts into one
    #cross_sections_vis = cross_sections.copy()
    #cross_sections_vis[cross_sections_vis==4] = 10
    #cross_sections_vis[cross_sections_vis==5] = 10
    #cross_sections_vis[cross_sections_vis==6] = 10
    #cross_sections_vis[cross_sections_vis==7] = 10
    #cross_sections_vis[cross_sections_vis==8] = 10
    #cross_sections_vis[cross_sections_vis>=9] -= 5
    #pf.heatmap(cross_sections_vis, dx=dx, sharescale=True, title='tissue types', cmap='tab20')
    
    # translate digimouse so the centre of mass along the xz plane
    # at y_ydx is in the centre
    #x = np.arange(digimouse.shape[0]) - digimouse.shape[0]//2
    #z = np.arange(digimouse.shape[2])[np.newaxis,:] - digimouse.shape[2]//2
    #for cross_section in cross_sections:
    #    COM_x = np.sum(digimouse[:,:]!=0)*x / (digimouse.shape[0]*digimouse.shape[2])
    #    COM_z = np.sum(digimouse[:,:]!=0)*z / (digimouse.shape[0]*digimouse.shape[2])
    
    wavelengths_nm = np.arange(680, 805, 5) # [nm]
    wavelengths_m = wavelengths_nm * 1e-9 # [nm] -> [m]
    coupling_medium_mu_a = 0.1 # [m^-1]
    coupling_medium_mu_s = 100 # [m^-1]
    
    chromophores_obj = phantom(wavelengths_m)    
    (Hb, HbO2) = chromophores_obj.define_Hb()
    H2O = chromophores_obj.define_H2O()
    mu_a_Hb = np.asarray(Hb['mu_a']) # [m^-1]
    mu_a_HbO2 = np.asarray(HbO2['mu_a']) # [m^-1]
    mu_a_H2O = np.asarray(H2O['mu_a']) # [m^-1]
    mu_s_H2O = np.asarray(H2O['mu_s']) # [m^-1]
    
    # blood volume fraction S_B, oxygen saturation x, water volume fraction S_W
    # note that the equation in the paper contains a typo, mu_a_HbO2 and mu_a_Hb are the wrong way around
    mu_a = lambda S_B, x, S_W : (S_B*(x*mu_a_HbO2 + (1-x)*mu_a_Hb) + S_W*mu_a_H2O) # alexandrakis eta al. (2005)
    # power law function for reduced scattering coefficient
    mu_s_alex = lambda a, b : (a * (wavelengths_nm**(-b))) * 1e3 # [m^-1] alexandrakis eta al. (2005)
    mu_s_jac = lambda a, b : (a * ((wavelengths_nm/500)**(-b))) * 1e3 # [m^-1] Steven L Jacques (2013) 
    
    # The following are optical properties recommended for use by the authors of digimouse
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
    # These are wavelength dependant but several tissue types are missing
    # so I will use Steven L Jacques (2013) values for the missing tissue types
    # @article{jacques2013optical,
    # title={Optical properties of biological tissues: a review},
    # author={Jacques, Steven L},
    # journal={Physics in Medicine \& Biology},
    # volume={58},
    # number={11},
    # pages={R37},
    # year={2013},
    # publisher={IOP Publishing}
    # }
    # The optical properties of water are from Hendrik Buiteveld (1994)
    # 
    
    # blood volume fraction S_B, oxygen saturation x, water volume fraction S_W
    absorption_coefficients = np.array([
        np.zeros_like(wavelengths_m), # 0 --> background
        mu_a(0.0033, 0.7, 0.5), # 1 --> skin --> adipose, alexandrakis eta al. (2005)
        mu_a(0.049, 0.8, 0.15), # 2 --> skeleton, alexandrakis eta al. (2005)
        mu_a(0.0033, 0.7, 0.5), # 3 --> eye --> adipose, alexandrakis eta al. (2005)
        mu_a(0.03, 0.6, 0.75), # 4 --> medulla --> Rat brain cortex, Steven L Jacques (2013)
        mu_a(0.03, 0.6, 0.75), # 5 --> cerebellum --> Rat brain cortex, Steven L Jacques (2013)
        mu_a(0.03, 0.6, 0.75), # 6 --> olfactory bulbs --> Rat brain cortex, Steven L Jacques (2013)
        mu_a(0.03, 0.6, 0.75), # 7 --> external cerebrum --> Rat brain cortex, Steven L Jacques (2013)
        mu_a(0.03, 0.6, 0.75), # 8 --> striatum --> Rat brain cortex, Steven L Jacques (2013)
        mu_a(0.05, 0.75, 0.5), # 9 --> heart, alexandrakis eta al. (2005)
        mu_a(0.03, 0.6, 0.75), # 10 --> rest of the brain --> Rat brain cortex, Steven L Jacques (2013)
        mu_a(0.07, 0.8, 0.5), # 11 --> masseter muscles, alexandrakis eta al. (2005)
        mu_a(0.0033, 0.7, 0.5), # 12 --> lachrymal glands --> adipose, alexandrakis eta al. (2005)
        mu_a_H2O, # 13 --> bladder --> water, Hendrik Buiteveld (1994)
        mu_a(0.07, 0.8, 0.5), # 14 --> testis --> muscle, alexandrakis eta al. (2005)
        mu_a(0.01, 0.7, 0.8), # 15 --> stomach, alexandrakis eta al. (2005)
        mu_a(0.3, 0.75, 0.7), # 16 --> spleen, alexandrakis eta al. (2005)
        mu_a(0.3, 0.75, 0.7), # 17 --> pancreas --> liver & spleen, alexandrakis eta al. (2005)
        mu_a(0.3, 0.75, 0.7), # 18 --> liver, alexandrakis eta al. (2005)
        mu_a(0.056, 0.75, 0.8),  # 19 --> kidneys, alexandrakis eta al. (2005)
        mu_a(0.07, 0.8, 0.5), # 20 --> adrenal glands --> muscle, alexandrakis eta al. (2005)
        mu_a(0.15, 0.85, 0.85) # 21 --> lungs, alexandrakis eta al. (2005)
    ]) # [m^-1]
    absorption_coefficients[0,:] = coupling_medium_mu_a
    
    
    # liver absorption is very high but this is validated by
    # @article{parsa1989optical,
    # title={Optical properties of rat liver between 350 and 2200 nm},
    # author={Parsa, Parwane and Jacques, Steven L and Nishioka, Norman S},
    # journal={Applied optics},
    # volume={28},
    # number={12},
    # pages={2325--2330},
    # year={1989},
    # publisher={Optica Publishing Group}
    # }
    
    scattering_coefficients = np.array([
        np.zeros_like(wavelengths_m), # 0 --> background
        mu_s_alex(38, 0.53), # 1 --> skin --> adipose, alexandrakis eta al. (2005)
        mu_s_alex(35600, 1.47), # 2 --> skeleton, alexandrakis eta al. (2005)
        mu_s_alex(38, 0.53), # 3 --> eye --> adipose, alexandrakis eta al. (2005)
        mu_s_jac(2.14, 1.2), # 4 --> medulla --> brain, Steven L Jacques (2013)
        mu_s_jac(2.14, 1.2), # 5 --> cerebellum --> brain, Steven L Jacques (2013)
        mu_s_jac(2.14, 1.2), # 6 --> olfactory bulbs --> brain, Steven L Jacques (2013)
        mu_s_jac(2.14, 1.2), # 7 --> external cerebrum --> brain, Steven L Jacques (2013)
        mu_s_jac(2.14, 1.2), # 8 --> striatum --> brain, Steven L Jacques (2013)
        mu_s_alex(10600, 1.43), # 9 --> heart, alexandrakis eta al. (2005)
        mu_s_jac(2.14, 1.2), # 10 --> rest of the brain --> brain, Steven L Jacques (2013)
        mu_s_alex(4e7, 2.82), # 11 --> masseter muscles, alexandrakis eta al. (2005)
        mu_s_alex(38, 0.53), # 12 --> lachrymal glands --> adipose, alexandrakis eta al. (2005)
        np.zeros_like(wavelengths_m), # 13 --> bladder --> water, Hendrik Buiteveld (1994)
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
    scattering_coefficients[0,:] = coupling_medium_mu_s
    scattering_coefficients[13,:] = mu_s_H2O
    
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
    #prop=np.array([
    #    [0, coupling_medium_mu_a, coupling_medium_mu_s, 0.9, 1.37], # 0 --> background
    #    [1, 0.0191, 6.6, 0.9, 1.37], # 1 --> skin --> scalp, Strangman et al. (2003), 830nm
    #    [2, 0.0136, 8.6, 0.9, 1.37], # 2 --> skeleton --> skull, Strangman et al. (2003), 830nm
    #    [3, 0.0026, 0.01, 0.9, 1.37], # 3 --> eye --> cerebrospinal fluid, Strangman et al. (2003), 830nm
    #    [4, 0.0186, 11.1, 0.9, 1.37], # 4 --> medulla --> brain, Strangman et al. (2003), 830nm
    #    [5, 0.0186, 11.1, 0.9, 1.37], # 5 --> cerebellum --> brain, Strangman et al. (2003), 830nm
    #    [6, 0.0186, 11.1, 0.9, 1.37], # 6 --> olfactory bulbs --> brain, Strangman et al. (2003), 830nm
    #    [7, 0.0186, 11.1, 0.9, 1.37], # 7 --> external cerebrum --> brain, Strangman et al. (2003), 830nm
    #    [8, 0.0186, 11.1, 0.9, 1.37], # 8 --> striatum --> brain, Strangman et al. (2003), 830nm
    #    [9, 0.0240, 8.9, 0.9, 1.37], # 9 --> heart --> muscle,
    #    [10, 0.0026, 0.01, 0.9, 1.37], # 10 --> rest of the brain --> cerebrospinal fluid, Strangman et al. (2003), 830nm
    #    [11, 0.0240, 8.9, 0.9, 1.37], # 11 --> masseter muscles --> muscle,
    #    [12, 0.0240, 8.9, 0.9, 1.37], # 12 --> lachrymal glands --> muscle,
    #    [13, 0.0240, 8.9, 0.9, 1.37], # 13 --> bladder --> muscle,
    #    [14, 0.0240, 8.9, 0.9, 1.37], # 14 --> testis --> muscle,
    #    [15, 0.0240, 8.9, 0.9, 1.37], # 15 --> stomach --> muscle,
    #    [16, 0.072, 5.6, 0.9, 1.37], # 16 --> spleen --> liver, Cheong et al. (1990)
    #    [17, 0.072, 5.6, 0.9, 1.37], # 17 --> pancreas
    #    [18, 0.072, 5.6, 0.9, 1.37], # 18 --> liver, Cheong et al. (1990)
    #    [19, 0.050, 5.4, 0.9, 1.37], # 19 --> kidneys --> cow kidney, Cheong et al. (1990), 789nm
    #    [20, 0.024, 8.9, 0.9, 1.37], # 20 --> adrenal glands --> muscle,
    #    [21, 0.076, 10.9, 0.9, 1.37] # 21 --> lungs --> pig lung, Cheong et al. (1990), 850nm
    #])
    #prop[:, 1] *= 1e3 # [mm^-1] -> [m^-1]
    #prop[:, 2] *= 1e3 # [mm^-1] -> [m^-1]
    #cross_sections_mu_a = prop[cross_sections,1]
    #cross_sections_mu_s = prop[cross_sections,2]
    
    # 770 nm = idx 17
    cross_sections_mu_a = absorption_coefficients[cross_sections, 17]
    cross_sections_mu_s = scattering_coefficients[cross_sections, 17]
    
    pf.heatmap(cross_sections_mu_a, dx=dx, sharescale=True, title=r'$\mu_{a}$ (m$^{-1}$)')
    pf.heatmap(cross_sections_mu_s, dx=dx, sharescale=True, title=r'$\mu_{s}$ (m$^{-1}$)')
    
    tissue_labels = ['adipose', 'skeleton', 'eye', 'brain', 'heart', 'muscle', 'water', 'stomach', 'liver & spleen', 'kidneys', 'lungs']
    labels_idx = np.array([1, 2, 3, 4, 9, 11, 13, 15, 16, 19, 21])
    fig, ax = plt.subplots(1, 1, figsize=(6,8))
    for i in range(len(tissue_labels)):
        ax.plot(wavelengths_nm, absorption_coefficients[labels_idx[i]], label=tissue_labels[i])
    ax.plot(wavelengths_nm, mu_a_Hb, label='Hb')
    ax.plot(wavelengths_nm, mu_a_HbO2, label='HbO2')
    ax.legend()
    ax.set_xlim(680, 800)
    ax.set_yscale('log')
    ax.set_ylabel(r'$\mu_{a}$ $($m$^{-1})$')
    ax.set_xlabel('wavelength (nm)')
    ax.grid(True)
    ax.set_axisbelow(True)
    
    
    '''
    cfg = {
     'wavelengths': [7e-07],
     'mcx_domain_size': [0.082, 0.025, 0.082],
     'kwave_domain_size': [0.082, 0.025, 0.082],
     'mcx_grid_size': [748, 236, 748],
     'kwave_grid_size': [748, 236, 748],
     'points_per_wavelength': 2,
     'dx': 0.00010962566844919787,
     'phantom': 'digimouse_phantom',
     'crop_size': 256
    }
    phantom_obj = digimouse_phantom('\\\\wsl$\\Ubuntu-22.04\\home\\wv00017\\digimouse_atlas\\atlas_380x992x208.img')
    H2O = phantom_obj.define_H2O(700e-9)
    (Hb, HbO2) = phantom_obj.define_Hb(700e-9)
    (volume, bg_mask) = phantom_obj.create_volume(cfg, 500, rotate=3)
    shape = volume.shape[1:]
    volume = volume[:,(shape[0]-cfg['crop_size'])//2:(shape[0]+cfg['crop_size'])//2,:,(shape[2]-cfg['crop_size'])//2:(shape[2]+cfg['crop_size'])//2]
    bg_mask = bg_mask[(shape[0]-cfg['crop_size'])//2:(shape[0]+cfg['crop_size'])//2,  (shape[2]-cfg['crop_size'])//2:(shape[2]+cfg['crop_size'])//2]
    pf.heatmap(volume[0,:,volume.shape[2]//2,:], dx=cfg['dx'], title=r'$\mu_{a}$ (m$^{-1}$)')
    pf.heatmap(volume[1,:,volume.shape[2]//2,:], dx=cfg['dx'], title=r'$\mu_{s}$ (m$^{-1}$)')
    pf.heatmap(bg_mask, dx=cfg['dx'], title='mask')
    '''
    
    # plot wavelength dependant optical properties of tissue types
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    tissue_labels = ['adipose', 'skeleton', 'brain', 'heart', 'muscle', 'stomach', 'liver & spleen', 'kidneys', 'lungs']
    colours = ['black', 'grey', 'tab:blue', 'tab:purple', 'green', 'tab:red', 'tab:orange', 'tab:pink', 'tab:cyan']
    linestyles = ['solid', (0,(1,1)), 'dashdot', (0,(5,5)), (0,(3,1,1,1,1,1)), (0,(4,2,4,2,1,2)), (0,(10,1,10,1,10,3)), (0,(10,3)), (0,(4,1))]
    labels_idx = np.array([1, 2, 4, 9, 11, 13, 15, 16, 19, 21])
    
    wavelengths_nm = np.linspace(500, 1200, num=1000)
    wavelengths_m = wavelengths_nm * 1e-9
    phantom_obj = digimouse_phantom('\\\\wsl$\\Ubuntu-22.04\\home\\wv00017\\digimouse_atlas\\atlas_380x992x208.img', wavelengths_m)
    H2O = phantom_obj.define_H2O()
    (Hb, HbO2) = phantom_obj.define_Hb()
    tissue_types_dict = phantom_obj.get_tissue_types_dict()
    tissue_types_dict['brain'] = tissue_types_dict['cerebellum']
    tissue_types_dict['water'] = tissue_types_dict['bladder']
    tissue_types_dict['muscle'] = tissue_types_dict['masseter muscles']
    tissue_types_dict['liver & spleen'] = tissue_types_dict['liver']
    
    for i in range(len(tissue_labels)):
        # absorption coefficient
        axes[0].plot(
            wavelengths_nm, 
            tissue_types_dict[tissue_labels[i]]['mu_a'],
            linestyle=linestyles[i], 
            linewidth=2, 
            color=colours[i], 
            label=tissue_labels[i]
        )
        # Scattering coefficient
        axes[1].plot(
            wavelengths_nm,
            tissue_types_dict[tissue_labels[i]]['mu_s'],
            linestyle=linestyles[i],
            linewidth=2,
            color=colours[i],
            label=tissue_labels[i]
        )
    
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Absorption coefficient ($\mathrm{m}^{-1}$)")
    axes[0].set_yscale("log")
    #axes[0].set_ylim(0, 1000)
    axes[0].grid()
    axes[0].text(0.1, 0.95, 'A', transform=axes[0].transAxes, fontsize=13, fontweight='bold', va='top')
    axes[0].legend(bbox_to_anchor=(0.47, 0.6, 0.45, 0.38))
    
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Scattering coefficient ($\mathrm{m}^{-1}$)")
    #axes[1].set_yscale("log")
    axes[1].grid()
    #axes[1].set_yticks([1E02,1E01])
    axes[1].ticklabel_format(useMathText=True)
    axes[1].ticklabel_format(style='plain')
    axes[1].text(0.1, 0.95, 'B', transform=axes[1].transAxes, fontsize=13, fontweight='bold', va='top')
    axes[1].legend(loc="best")

    