import numpy as np
from scipy.ndimage import rotate
from phantoms.phantom import phantom
from PIL import Image
import func.geometry_func as gf
import func.utility_func as uf


class ImageNet_phantom(phantom):
    
    def __init__(self, rng : np.random.Generator, *args, **kwargs):
        super(ImageNet_phantom, self).__init__(*args, **kwargs)
        self.rng = rng
        
    def create_volume(self, cfg: dict, image_file : str):
        ppw = cfg['points_per_wavelength']
        # image must be a path to a JPEG
        with Image.open(image_file) as f:
            image = f.resize((256*ppw, 256*ppw))
            image = image.convert('RGB')
            image = np.array(image, dtype=np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
        
        mu_s_min = self.rng.uniform(2000, 8000) # [m^-1]
        mu_s_max = self.rng.uniform(10000, 20000) # [m^-1]
        mu_a_min = self.rng.uniform(5, 20) # [m^-1]
        mu_a_max = self.rng.uniform(25, 100) # [m^-1]
        coupling_medium_mu_a = self.H2O['mu_a'][0] # [m^-1]
        coupling_medium_mu_s = self.H2O['mu_s'][0] # [m^-1]
        
        # volume[0] = absorption coefficient [m^-1]
        # volume[1] = scattering coefficient [m^-1]
        volume = np.zeros((
            2, cfg['mcx_grid_size'][0], cfg['mcx_grid_size'][1], cfg['mcx_grid_size'][2]            
        ), dtype=np.float32)      
        
        image = rotate(
            image, np.random.randint(-180, 180), axes=(1, 2), reshape=False
        )
        
        (mu_a_channel, mu_s_channel) = np.random.choice([0, 1, 2], 2, replace=False)
        bg_mask = gf.random_spline_mask(
            self.rng, R_min=25*ppw, R_max=55*ppw, crop_size=cfg['crop_size']
        )
        bg_mask = uf.square_centre_pad(bg_mask, cfg['mcx_grid_size'][0])
        bg_mask = bg_mask[:,np.newaxis,:]
        image = uf.square_centre_pad(image, cfg['mcx_grid_size'][0])
        image = image[:,:,np.newaxis,:]
        
        volume[0] += (mu_a_min + (mu_a_max - mu_a_min) * image[mu_a_channel]) * bg_mask
        volume[1] += (mu_s_min + (mu_s_max - mu_s_min) * image[mu_s_channel]) * bg_mask
        volume[0] += coupling_medium_mu_a * (~bg_mask)
        volume[1] += coupling_medium_mu_s * (~bg_mask)
        
        return (volume, bg_mask)