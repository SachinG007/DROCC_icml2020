import numpy as np

train_f = np.load('train_seven.npz')['features']
others_f = np.load('other_seven.npz')['features']
cn_f = np.load('cn_seven.npz')['features']

np.random.shuffle(train_f)
np.random.shuffle(others_f)
np.random.shuffle(cn_f)

len_t = 1000

test_data = np.concatenate((train_f[:len_t], others_f[:len_t]), axis=0)
labels = [1] * len_t + [0] * len_t

np.save('train_data.npy', test_data)
np.save('train_labels.npy', labels)

test_data = np.concatenate((train_f[len_t:len_t+200], others_f[len_t:len_t+200]), axis=0)
labels = [1] * 200 + [0] * 200
np.save('test_others_data.npy', test_data)
np.save('test_others_labels.npy', labels)

test_data = np.concatenate((train_f[len_t:len_t+200], cn_f[len_t:len_t+200]), axis=0)
labels = [1] * 200 + [0] * 200
np.save('test_cn_data.npy', test_data)
np.save('test_cn_labels.npy', labels)
