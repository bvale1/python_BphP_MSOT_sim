import numpy as np

def grid_xyz(dx, grid_size):
    return np.meshgrid(
        np.arange(grid_size[0], dtype=np.float32) * dx,
        np.arange(grid_size[1], dtype=np.float32) * dx,
        np.arange(grid_size[2], dtype=np.float32) * dx,
        indexing='ij'
    )

def sphere_mask(dx, grid_size, radius, origin):
    
    [X,Y,Z] = grid_xyz(dx, grid_size)
    
    distances = np.sqrt(
        (X - origin[1])**2 + 
        (Y - origin[1])**2 +
        (Z - origin[0])**2
    )
    
    return distances <= radius

def cylinder_mask(dx : (int, float),
                  grid_size : (list, tuple, np.ndarray),
                  radius : (int, float), 
                  origin : (list, tuple, np.ndarray)) -> np.ndarray:
    # Note origin and radius are in mm
    # Note origin is in the form [x, y, z]
    # Note the cylinder is aligned along the y-axis
    [X,Y,Z] = grid_xyz(dx, grid_size)
    
    distances = np.sqrt(
        (X - origin[0])**2 + 
        (Z - origin[2])**2
    )
    
    return distances <= radius

def quadratic_profile_tumor(dx, grid_size, radius, origin):
    
    [X,Y,Z] = grid_xyz(dx, grid_size)
    
    distances = np.sqrt(
        (X - origin[0])**2 + 
        (Y - origin[1])**2 +
        (Z - origin[2])**2
    )
    
    absoption_profile = 1 - ((distances / radius)**2)
    # zero outside the radius
    absoption_profile *= (distances <= radius)
    
    return absoption_profile

def get_optical_grid_size(domain_size=[0.082, 0.025, 0.082],
                          c0_min=1500,
                          points_per_wavelength=2,
                          f_max=6e6,
                          pml_size=10):
    
    # calculate grid size from powers of 2 and 3
    dx_min = c0_min / (points_per_wavelength * f_max)
    nx = 2**(np.ceil(np.log2(domain_size[0] / dx_min))-2) * 3
    
    # subtract pml from each edge
    nx = int(nx - 2 * pml_size)
    dx = domain_size[0] / nx
    ny = int(2**np.round(np.log2(domain_size[1] / dx))) - 2 * pml_size
    nz = int(2**np.round(np.log2(domain_size[2] / dx)-2) * 3) - 2 * pml_size
    
    return [nx, ny, nz], dx
    
    
def get_acoustic_grid_size(dx, domain_size=[0.082, 0.01025, 0.082], pml_size=10):
        
        nx = int(2**np.round(np.log2(domain_size[0] / dx))) - 2 * pml_size
        ny = int(2**np.round(np.log2(domain_size[1] / dx))) - 2 * pml_size
        nz = int(2**np.round(np.log2(domain_size[2] / dx))) - 2 * pml_size
        
        return [nx, ny, nz], dx
                           