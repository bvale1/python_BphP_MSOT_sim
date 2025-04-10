import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import h5py
import os
from phantoms.digimouse_phantom import digimouse_phantom

def meanpool(X, f=2):
    I1, J1, K1 = X.shape
    I2 = I1 // f
    J2 = J1 // f
    K2 = K1 // f
    return X[:I2*f,:J2*f,:K2*f].reshape(I2,f,J2,f,K2,f).mean(axis=(1, 3, 5))

# wsl
#water_sim_path = '/mnt/f/cluster_MSOT_simulations/20240104_water_calibration.c144702.p0/temp.h5'
#digimouse_atlas_path = '/home/wv00017/digimouse_atlas/atlas_380x992x208.img'
# win
water_sim_path = r'F:\cluster_MSOT_simulations\20240104_water_calibration.c144702.p0\temp.h5'
digimouse_atlas_path = r'F:\digimouse_atlas\atlas_380x992x208.img'

# config
cfg = {
    'mcx_grid_size' : [748, 236, 748],
    'dx' : 0.00010962566844919787 # 109.62566844919787 microns
}

# create 3d plot
ax = plt.figure().add_subplot(projection='3d')
'''
# create digimouse phantom
print('creating digimouse phantom')
phantom = digimouse_phantom(digimouse_atlas_path, wavelengths_m=[750e-9])
H2O = phantom.define_H2O()
(Hb, HbO2) = phantom.define_Hb()
#absorption_coefficients = phantom.calculate_tissue_absorption_coefficients()
(volume, bg_mask) = phantom.create_volume(
    cfg, 500, rotate=2, extrusion=False, bg_mask_2d=False
)
volume = volume[0]
# downsample
volume = meanpool(volume)
bg_mask = meanpool(bg_mask.astype(np.float32))
bg_mask = np.round(bg_mask).astype(bool)
# define voxels
X, Y, Z = np.indices(np.asarray(volume.shape) + np.array([1,1,1]), dtype=np.float32)
X -= (cfg['mcx_grid_size'][0]/2)+0.5
Y -= (cfg['mcx_grid_size'][1]/2)+0.5
Z -= (cfg['mcx_grid_size'][2]/2)+0.5
X *= 2 * cfg['dx'] * 1e3 # convert to mm
Y *= -2 * cfg['dx'] * 1e3 # convert to mm
Z *= 2 * cfg['dx'] * 1e3 # convert to mm
# define colors
colors = np.zeros(volume.shape, dtype=np.float32)
volume /= np.max(volume) # normalize to 1
colors[bg_mask == 0] = 0.0 # transparent
colors[bg_mask == 1] = volume[bg_mask==1] # greyscale
colors = np.repeat(colors[...,np.newaxis], 3, axis=-1)
#colors = matplotlib.colors.hsv_to_rgb(colors)
# downsample for faster plotting testing
#bg_mask = bg_mask[::10, ::10, ::10]
#colors = colors[::10, ::10, ::10]
# plot the digimouse phantom
print('plotting digimouse voxels')
ax.voxels(X, Y, Z, bg_mask, facecolors=colors)

# transducer array geometry provided by Janek Gröhl https://github.com/jgroehl
number_detector_elements = 256
radius_mm = 40.5
radius_2_mm = 37
pitch_mm = 0.74
element_size = 0.635
element_length = 15
num_interpolation_points_x = 7
num_interpolation_points_y = 101
x_range = (
    np.linspace(
        -num_interpolation_points_x / 2,
        num_interpolation_points_x / 2,
        num_interpolation_points_x
    )
)
y_range = (
    np.linspace(
        -num_interpolation_points_y / 2,
        num_interpolation_points_y / 2,
        num_interpolation_points_y
    )
)

pitch_angle = pitch_mm / radius_mm
x_increment_angle = element_size / radius_mm / num_interpolation_points_x
y_increment_angle = element_length / radius_2_mm / num_interpolation_points_y
det_elements = np.linspace(
    number_detector_elements / 2,
    -number_detector_elements / 2,
    number_detector_elements
)

rotation_angle = - np.pi / 2
Ry = np.array([
    [np.cos(rotation_angle), 0, np.sin(rotation_angle)],
    [0, 1, 0],
    [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]
]) # euclidian rotation matrix

theta = np.pi/2
print('computing detector positions')
for det_idx in range(len(det_elements)):
    detector_positions = np.zeros(
        (3, num_interpolation_points_x * num_interpolation_points_y)
    )
    for x_idx in range(len(x_range)):
        x_inc = x_range[x_idx] * x_increment_angle
        for y_idx in range(len(y_range)):
            y_inc = y_range[y_idx] * y_increment_angle
            idx = (x_idx-1) * num_interpolation_points_y + y_idx
            detector_positions[0, idx] = np.sin(np.pi/2 + pitch_angle * det_elements[det_idx] + x_inc) * np.sin(theta + y_inc) * (radius_mm - 0.5 * element_size)
            detector_positions[1, idx] = np.cos(theta + y_inc) * (radius_mm - 0.5 * element_size)
            detector_positions[2, idx] = np.cos(np.pi/2 + pitch_angle * det_elements[det_idx] + x_inc) * np.sin(theta + y_inc) * (radius_mm - 0.5 * element_size)
    detector_positions = np.matmul(Ry, detector_positions)
    ax.scatter(
        detector_positions[0, :],
        -detector_positions[1, :],
        detector_positions[2, :],
        s=0.5
    )

'''
# load p0_3d data
print('loading p0_3d data')
with h5py.File(water_sim_path, 'r') as f:
    p0_3d = f['p0_3D'][0,1,0]
p0_3d = meanpool(p0_3d)
X, Y, Z = np.indices(np.asarray(p0_3d.shape), dtype=np.float32)
# threshold p0_3d to remove values above 90% and below 10% of the max
p0_mask = (p0_3d > 0.001 * np.max(p0_3d)) & (p0_3d < 0.9 * np.max(p0_3d))
p0_3d = p0_3d[p0_mask]
# normalize p0_3d
p0_3d = (p0_3d - np.min(p0_3d)) / (np.max(p0_3d) - np.min(p0_3d))
# apply gamma correction to p0_3d
p0_3d = p0_3d**3
# mask coordinates
X = X[p0_mask]
Y = Y[p0_mask]
Z = Z[p0_mask]
X -= (cfg['mcx_grid_size'][0]/2)
Y -= (cfg['mcx_grid_size'][1]/2)
Z -= (cfg['mcx_grid_size'][2]/2)
X *= 2 * cfg['dx'] * 1e3 # convert to mm
Y *= 2 * cfg['dx'] * 1e3 # convert to mm
Z *= 2 * cfg['dx'] * 1e3 # convert to mm
# plot p0_3d
ax.scatter(X, Y, Z, s=0.5, color='red', alpha=p0_3d)

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
print('show plot')
plt.show()