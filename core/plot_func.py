import numpy as np
import matplotlib.pyplot as plt


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