import numpy as np
from typing import Union
from scipy import interpolate


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
        (X - origin[0])**2 + 
        (Y - origin[1])**2 +
        (Z - origin[2])**2
    )
    
    return distances <= radius

def cylinder_mask(dx: Union[int, float],
                  grid_size: Union[list, tuple, np.ndarray],
                  radius: Union[int, float], 
                  origin: Union[list, tuple, np.ndarray]) -> np.ndarray:
    # Note origin and radius are in mm
    # Note origin is in the form [x, y, z]
    # Note the cylinder is aligned along the y-axis
    [X,Y,Z] = grid_xyz(dx, np.asarray(grid_size))
    
    distances = np.sqrt(
        (X - origin[0])**2 + 
        (Z - origin[2])**2
    )
    
    return distances <= radius

def quadratic_profile_tumor(dx : Union[int, float],
                            grid_size : Union[list, tuple, np.ndarray],
                            radius : Union[int, float], 
                            origin : Union[list, tuple, np.ndarray]) -> np.ndarray:
    
    [X,Y,Z] = grid_xyz(dx, np.asarray(grid_size))
    
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
    
    
def get_acoustic_grid_size(dx, domain_size=[0.082, 0.025, 0.082], pml_size=10):
        
    nx = int(2**(np.round(np.log2(domain_size[0] / dx))-2) * 3) - 2 * pml_size
    ny = int(2**(np.round(np.log2(domain_size[1] / dx)))) - 2 * pml_size
    nz = int(2**(np.round(np.log2(domain_size[2] / dx))-2) * 3) - 2 * pml_size
    
    return [nx, ny, nz], dx
                           
                           
def random_spline_mask(rng : np.random.Generator,
                       R_min=85,
                       R_max=125,
                       n_min=6,
                       n_max=12):
    n_points = int(rng.uniform(n_min, n_max))
    # define the boundary with a coarse set of random points
    R_coarse = rng.uniform(R_min, R_max, n_points).astype(np.float32)
    theta_coarse = np.linspace(0, 2*np.pi, n_points, dtype=np.float32)
    # smooth the boundary with spline interpolation
    spline = interpolate.splrep(theta_coarse, R_coarse, k=3, per=True)
    theta = np.linspace(0, 2*np.pi, 1000, np.float32)
    R = interpolate.splev(theta, spline)
    R[R>R_max] = R_max
    # to cartesian coordinates
    x = R * np.sin(theta)
    y = R * np.cos(theta)
    # create the binary mask within the boundary using 2D linear interpolation
    [X, Y] = np.meshgrid(np.arange(256)-128, np.arange(256)-128)
    interp = interpolate.LinearNDInterpolator(
        list(zip(x, y)), np.ones(1000), fill_value=0
    )
    mask = interp(X, Y).astype(bool)

    return mask