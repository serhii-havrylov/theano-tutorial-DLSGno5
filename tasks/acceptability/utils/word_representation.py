import h5py


with h5py.File('data/truncated_glove_model_300d.h5', 'r') as h5_file:
    word_to_idx = dict((word.decode('utf-8'), i)
                       for i, word in enumerate(h5_file['idx_to_word'][...]))
    idx_to_vector = h5_file['idx_to_vector'][...]