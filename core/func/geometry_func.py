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


def get_sim_grid_size(domain_size=[0.082, 0.025, 0.082],
                      c0_min=1500,
                      points_per_wavelength=2,
                      f_max=7e6,
                      pml_size=10):
    
    # calculate grid size from prime factors 2, 3 and 5
    dx_min = c0_min / (points_per_wavelength * f_max)
    domain_size = np.asarray(domain_size) + 2 * pml_size * dx_min
    grid_size = np.zeros(3, dtype=int)
    for dim in range(3):
        grid_size[dim] = min([i for i in [
            2**(np.ceil(np.log2((domain_size[dim] / dx_min)))) - 2 * pml_size,
            (2**(np.ceil(np.log2((domain_size[dim] / dx_min)))-2) * 3) - 2 * pml_size,
            (2**(np.ceil(np.log2((domain_size[dim] / dx_min)))-3) * 5) - 2 * pml_size
        ] if i * dx_min >= domain_size[dim] - 2 * pml_size * dx_min])
    
    domain_size -= 2 * pml_size * dx_min
    dx = np.max(domain_size / grid_size)
    
    return grid_size.tolist(), dx

                           
def random_spline_mask(rng : np.random.Generator,
                       R_min=50,
                       R_max=110,
                       n_min=7,
                       n_max=14,
                       crop_size=256):
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
    [X, Y] = np.meshgrid(np.arange(crop_size)-crop_size//2, np.arange(crop_size)-crop_size//2)
    interp = interpolate.LinearNDInterpolator(
        list(zip(x, y)), np.ones(1000), fill_value=0
    )
    mask = interp(X, Y).astype(bool)

    return mask