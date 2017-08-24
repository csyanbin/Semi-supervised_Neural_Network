import sys
print(sys.argv[1])
label = sys.argv[1]
rand  = sys.argv[2]
my_parameter = {
    'batch':20,
    #'batch':100,
    'epochs':100,
    #'learning_rate':0.002,
    'learning_rate':0.003,
    #'noise_std':0.3,
    'noise_std':0.2,
    'seed':1,
    #'unsupervised_cost_lambda': [0.1, 0.1, 0.1, 0.1, 0.1, 10., 20000.],
    'unsupervised_cost_lambda': [0.1, 0.1, 0.1, 0.1, 0.1, 20., 2000.],
    'cuda':True,
    'decay_epoch':20,
    'encoder_activations':['relu', 'relu', 'relu', 'relu', 'relu', 'softmax'],
    #'encoder_sizes': [1000, 500, 250, 250, 250, 20],
    #'decoder_sizes': [250, 250, 250, 500, 1000, 1024],
    #'encoder_sizes': [500, 250, 125, 125, 125, 20],
    #'decoder_sizes': [125, 125, 125, 250, 500, 1024],
    'encoder_sizes': [200, 100, 50, 50, 50, 20],
    'decoder_sizes': [50, 50, 50, 100, 200, 1024],
    'data_dir':'../data/coil-20-data/label'+label+'_'+rand+'/',
    'encoder_train_bn_scaling': [True, True, True, True, True, True]
}
