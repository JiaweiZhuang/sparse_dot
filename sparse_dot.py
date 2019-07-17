import numpy as np
import xarray as xr
from scipy.sparse import coo_matrix

from timeit import default_timer as timer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='coo', help='coo, csc or csr')
parser.add_argument('--order', type=str, default='F', help='F or C')
args = parser.parse_args()


class Timer:    
    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, *args):
        self.end = timer()
        print('time: {} seconds'.format(self.end - self.start)) 

        
def get_weights():
    ds = xr.load_dataset("weights.nc")

    n_s = ds.dims['n_s']
    col = ds['col'].values - 1
    row = ds['row'].values - 1
    S = ds['S'].values

    A = coo_matrix((S, (row, col))) 
    
    if args.type == 'csc':
        A = A.tocsc()
    
    elif args.type == 'csr':
        A = A.tocsr()
        
    elif args.type == 'coo':
        pass
    
    else:
        raise ValueError
    
    return A


def get_data(A):
    data = np.random.rand(500, A.shape[1])
    
    if args.order == 'F':
        data_T = data.T
        
    elif args.order == 'C':
        data_T = data.T.copy()
        
    else:
        raise ValueError        
        
    return data_T


def main():
    A = get_weights()
    data = get_data(A)
    
    with Timer():
        for _ in range(5):
            A.dot(data)
        
        
if __name__ == "__main__":
    main()