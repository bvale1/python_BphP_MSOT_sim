import numpy as np
import matplotlib.pyplot as plt
import logging
from mpl_toolkits.axes_grid1 import make_axes_locatable

# plots 4D data as a 3D scatter plot
def plot4D(data,
           cmap="Reds",
           dpi=500,
           save_name=False,
           logartihmic=False):
    
    print(f'(max, min)=({np.min(data)}, {np.max(data)})')
    data += np.abs(np.min(data))
    if logartihmic:
        data = np.log10(data+1e-3)
        print(f'(log10(max), log10(min))=({np.min(data)}, {np.max(data)})')
                
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    mask = data# > 0.01
    idx = np.arange(int(np.prod(data.shape)))
    x, y, z = np.unravel_index(idx, data.shape)
    ax.scatter(
        x, y, z, c=data.flatten(),
        s=3.0 * mask, 
        edgecolor="face",
        alpha=0.2, marker="o",
        cmap=cmap,
        linewidth=0
    )
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=dpi)
    plt.close(fig)
    

def heatmap(img, 
            title='', 
            cmap='binary_r', 
            vmax=None,
            vmin=None,
            dx=0.0001, 
            rowmax=6,
            labels=None,
            sharescale=False,
            cbar_label=None):
    # TODO: heatmap should use a list to plot images of different resolution
    logging.basicConfig(level=logging.INFO)    
    # use cmap = 'cool' for feature extraction
    # use cmap = 'binary_r' for raw data
    dx = dx * 1e3 # [m] -> [mm]
    
    frames = []
        
    shape = np.shape(img)
    if sharescale or len(shape) == 2:
        mask = np.logical_not(np.isnan(img))
        if not vmin:
            vmin = np.min(img[mask])
        if not vmax:
            vmax = np.max(img[mask])
    
    extent = [-dx*shape[-2]/2, dx*shape[-2]/2, -dx*shape[-1]/2, dx*shape[-1]/2]
    
    if len(shape) == 2: # one pulse
        nframes = 1
        fig, ax = plt.subplots(nrows=1, ncols=nframes, figsize=(6,8))
        ax = np.array([ax])
        ax[0].set_xlabel('x (mm)')
        ax[0].set_ylabel('z (mm)')
        frames.append(ax[0].imshow(
            img,
            cmap=cmap, 
            vmin=vmin, 
            vmax=vmax,
            extent=extent,
            origin='lower'
        ))
        divider = make_axes_locatable(ax[0])
        cbar_ax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(frames[0], cax=cbar_ax, orientation='vertical')        
        
    else: # multiple pulses
        nframes = shape[0]
        nrows = int(np.ceil(nframes/rowmax))
        rowmax = nframes if nframes < rowmax else rowmax
        fig, ax = plt.subplots(nrows=nrows, ncols=rowmax, figsize=(16, 12))
        ax = np.asarray(ax)
        if len(np.shape(ax)) == 1:
            ax = ax.reshape(1, rowmax)
        for row in range(nrows):
            ax[row, 0].set_ylabel('z (mm)')
        for col in range(rowmax):
            ax[-1, col].set_xlabel('x (mm)')
        ax = ax.ravel()
        
        for frame in range(nframes): 
            if not sharescale:
                mask = np.logical_not(np.isnan(img[frame]))
                vmin = np.min(img[frame][mask])
                vmax = np.max(img[frame][mask])
            frames.append(ax[frame].imshow(
                img[frame],
                cmap=cmap, 
                vmin=vmin, 
                vmax=vmax,
                extent=extent,
                origin='lower'
            ))
            ax[frame].set_xlabel('x (mm)')
            if labels:
                ax[frame].set(title=labels[frame])
            if not sharescale:
                divider = make_axes_locatable(ax[frame])
                cbar_ax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(frames[frame], cax=cbar_ax, orientation='vertical')
                #cbar = plt.colorbar(frames[frame], ax=ax[frame])
                if cbar_label:
                    cbar.set_label=cbar_label

    fig.subplots_adjust(right=0.8)
    
    if sharescale:
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(frames[0], cax=cbar_ax)
        if cbar_label:
            cbar.set_label=cbar_label
    else:
        fig.tight_layout()
            
    fig.suptitle(title, fontsize='xx-large')
    
    return (fig, ax, frames)


if __name__ == '__main__':
    import h5py
    
    load_path = '202307020_python_Clara_phantom_ReBphP_0p001/data.h5'
    
    f = h5py.File(load_path, 'r')
    data = np.array(f['norm_fluence'])
    
    # discard top ten points
    shape = np.shape(data)
    data = data.flatten()
    for i in range(10):
        data[np.argmax(data)] = 0.0
    data = np.reshape(data, shape)
    
    # plot the results
    plot4D(
        data,
        save_name='202307020_python_Clara_phantom_ReBphP_0p001/Janek_InVision_MCX_source.png',
    )