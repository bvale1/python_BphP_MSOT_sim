import numpy as np
from phantoms.ImageNet_phantom import fluence_correction_phantom
import json, h5py, os, timeit, logging, argparse, gc
import func.geometry_func as gf
import func.utility_func as uf
import optical_simulation
import acoustic_forward_simulation
import acoustic_inverse_simulation


# same function as in https://github.com/bvale1/MSOT_Diffusion.git
def load_sim(path : str, args='all', verbose=False) -> list:
    data = {}
    with h5py.File(os.path.join(path, 'data.h5'), 'r') as f:
        images = list(f.keys())
        if verbose:
            print(f'images found {images}')
        if args == 'all':
            args = f[images[0]].keys()
            print(f'args found in images[0] {args}')
        for image in images:
            data[image] = {}
            for arg in args:
                if arg not in f[image].keys():
                    print(f'arg {arg} not found in {image}')
                    pass
                # include 90 deg anticlockwise rotation
                elif arg != 'sensor_data':
                    data[image][arg] = np.rot90(
                        np.array(f[image][arg][()]), k=1, axes=(-2,-1)
                    ).copy()
                else:
                    data[image][arg] = np.array(f[image][arg][()])
            
    with open(path+'/config.json', 'r') as f:
        cfg = json.load(f)
        
    return [data, cfg]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Iterative model-based fluence correction')
    parser.add_argument(
        '--mcx_bin_path', type=str,
        default='/home/wv00017/mcx/bin/mcx',
        action='store',
        help='path to MCX CUDA binary'
    )
    parser.add_argument(
        '--save_dir', type=str,
        default='unnamed_sim',
        action='store',
        help='directory to save simulation data'
    )
    parser.add_argument('--dataset', type=str, help='path to dataset')
    parser.add_argument('--niter', type=int, help='Number of iterations', default=10)
    parser.add_argument('--crop_size', type=int, default=256, action='store')
    parser.add_argument('--sim_git_hash', type=str, default=None, action='store')
    parser.add_argument('--recon_iterations', type=int, default=5, action='store')
    parser.add_argument('--recon_alpha', type=float, default=1.0, action='store')
    parser.add_argument('--forward_model', type=str, default='invision', action='store')
    parser.add_argument('--inverse_model', type=str, default='invision', action='store')
    parser.add_argument('--crop_p0_3d_size', type=int, default=512, action='store')
    parser.add_argument('--delete_p0_3d', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('-v', type=str, help='verbose level', default='INFO')
    parser.add_argument('--Gamma', type=float, default=1.0, action='store', help='Gruneisen parameter')
    
    args = parser.parse_args()
    
    if args.v == 'INFO':
        logging.basicConfig(level=logging.INFO)
    elif args.v == 'DEBUG':
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.info(f'{args.v} not a recognised verbose level, using INFO instead')
    
    if (os.path.exists(args.save_dir+'config.json') 
        and os.path.exists(args.save_dir+'data.h5')):
        
        with open(args.save_dir+'config.json', 'r') as f:
            cfg = json.load(f)
        logging.info(f'dataset config found {cfg}')
        
    