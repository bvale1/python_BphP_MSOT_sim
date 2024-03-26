import numpy as np
from phantoms.phantom import phantom
import geometry_func as gf
import BphP_func as bf


class Clara_experiment_phantom(phantom):
    
    def create_volume(self, cfg : dict, bg_mu_s=0.0) -> tuple:
        # phantom is made from 1.5% agarose 3.5% intralipid emulsion
        
        # initialise proteins, Pr to Pfr ratio is the steady state
        Pr_frac, Pfr_frac = bf.steady_state_BphP(
            self.ReBphP_PCM['Pr'],
            self.ReBphP_PCM['Pfr'],
            wavelength_idx=1
        )
        # [M] = [mol L^-3] = [mol/mm^3]
        c_tot = 0.0005 * gf.cylinder_mask(
            cfg['dx'],
            cfg['mcx_grid_size'],
            1.5e-3,
            [(cfg['mcx_domain_size'][0]/2)-2e-3, 0.0, (cfg['mcx_domain_size'][2]/2)-4e-3]
        )
        
        ReBphP_PCM_Pfr_c = c_tot * Pfr_frac
        ReBphP_PCM_Pr_c = c_tot * Pr_frac
        
        # define volume scattering and absorption coefficients
        # index as [1, ..., lambda]->[mu_a, mu_s]->[x]->[y]->[z]
        volume = np.zeros(
            (
                len(cfg['wavelengths']),
                2, 
                cfg['mcx_grid_size'][0], 
                cfg['mcx_grid_size'][1], 
                cfg['mcx_grid_size'][2]            
            ), dtype=np.float32
        )
        volume[:,1,:,:,:] = bg_mu_s
        mask = gf.cylinder_mask(
            cfg['dx'],
            cfg['mcx_grid_size'],
            0.01,
            [(cfg['mcx_domain_size'][0]/2), 0.0, (cfg['mcx_domain_size'][2]/2)]
        )
        for i in range(len(cfg['wavelengths'])):
            volume[i,0,:,:,:] = 1 * mask #self.water89_gelatin1_intralipid10['mu_a'][i] * mask
            volume[i,1,:,:,:] = 1000 * mask #self.water89_gelatin1_intralipid10['mu_s'][i] * mask
        bg_mask = mask[:,0,:]
        
        return (volume, ReBphP_PCM_Pr_c, ReBphP_PCM_Pfr_c, bg_mask)