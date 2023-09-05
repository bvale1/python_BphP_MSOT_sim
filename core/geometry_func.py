import numpy as np

def sphere_mask(dx, grid_size, radius, origin):
    
    [X,Y,Z] = np.meshgrid(
        np.arange(grid_size[0], dtype=np.float32) * dx, 
        np.arange(grid_size[1], dtype=np.float32) * dx, 
        np.arange(grid_size[2], dtype=np.float32) * dx,
        indexing='ij'
    )
    
    distances = np.sqrt(
        (X - origin[1])**2 + 
        (Y - origin[1])**2 +
        (Z - origin[0])**2
    )
    
    return distances <= radius


def cylinder_mask(dx, grid_size, radius, origin):
    
    # Note the cylinder is aligned along the y-axis
    [X,Y,Z] = np.meshgrid(
        np.arange(grid_size[0], dtype=np.float32) * dx, 
        np.arange(grid_size[1], dtype=np.float32) * dx, 
        np.arange(grid_size[2], dtype=np.float32) * dx,
        indexing='ij'
    )
    
    distances = np.sqrt(
        (X - origin[0])**2 + 
        (Z - origin[2])**2
    )
    
    return distances <= radius


def get_optical_grid_size(domain_size=[0.082, 0.01, 0.082],
                          c0_min=1500,
                          points_per_wavelength=2,
                          f_max=6e6,
                          pml_size=10):
    
    # calculate grid size from powers of 2
    dx_min = c0_min / (points_per_wavelength * f_max)
    nx = 2**(np.ceil(np.log2(domain_size[0] / dx_min)))
    
    # subtract pml from each edge
    nx = int(nx - 2 * pml_size)
    dx = domain_size[0] / nx
    ny = int(np.ceil(domain_size[1] / dx))
    if ny % 2 == 1:
        ny += 1
    nz = int(np.ceil(domain_size[2] / dx))
    
    return [nx, ny, nz], dx
    
    