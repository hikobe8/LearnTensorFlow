from scipy.io import loadmat as load

train_data = load("../data/train_32x32.mat")
test_data = load("../data/test_32x32.mat")
extra_data = load("../data/extra_32x32.mat")

print('Train Data Samples Shape: ',  train_data['X'].shape)
print('Train Data     Labels Shape: ',  train_data['y'].shape)

print('Test Data Samples Shape: ',  test_data['X'].shape)
print('Test Data     Labels Shape: ',  test_data['y'].shape)

print('Extra Data Samples Shape: ',  extra_data['X'].shape)
print('Extra Data     Labels Shape: ',  extra_data['y'].shape)