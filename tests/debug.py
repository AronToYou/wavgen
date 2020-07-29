import h5py

with h5py.File('/nothere.h5', 'w') as f:
    f.create_group('duh')

print('done')
