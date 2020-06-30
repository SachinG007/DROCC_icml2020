import pandas as pd
import numpy as np

data = pd.read_csv('abalone.data', header=None, sep=',')
data.head()

data = data.rename(columns={8: 'y'})

data['y'].replace([8, 9, 10], -1, inplace=True)
data['y'].replace([3, 21], 0, inplace=True)
data.iloc[:, 0].replace('M', 0, inplace=True)
data.iloc[:, 0].replace('F', 1, inplace=True)
data.iloc[:, 0].replace('I', 2, inplace=True)

test = data[data['y'] == 0]

normal = data[data['y'] == -1].sample(frac=1)
# np.random.shuffle(normal)
test_data = np.concatenate((test.drop('y', axis=1), normal[:29].drop('y', axis=1)), axis=0)
train = normal[29:]
train_data = train.drop('y', axis=1).values
train_labels = train['y'].replace(-1, 1)
# test_data = test.drop('y', axis=1)
test_labels = np.concatenate((test['y'], normal[:29]['y'].replace(-1, 1)), axis=0)

normal = data[data[:,-1] == 0]
np.random.shuffle(normal)
test = np.concatenate((test, normal[:66]), axis=0)
train = normal[66:]
train_data = train[:,:-1]
train_labels = train[:,-1]
test_data = test[:,:-1]
test_labels = test[:, -1]

np.save('train_data.npy', train_data)
np.save('train_labels.npy', train_labels)
np.save('test_data.npy', test_data)
np.save('test_labels.npy', test_labels)


