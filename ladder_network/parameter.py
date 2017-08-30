import sys
print(sys.argv[1])
label = sys.argv[1]
rand  = sys.argv[2]

ratio = 1.0
dataset = 'coil'
if len(sys.argv)>=4:
    ratio = float(sys.argv[3])
if len(sys.argv)==5:
    dataset = sys.argv[4]

if dataset=='coil':
    folder = '../data/coil-20-data/label'+label+'_'+rand+'/'
    n_class= 20
    n_input=1024
    batch = 20
    lr = 0.002
elif dataset=="hela":
    folder = '../data/hela-10-data/label'+label+'_'+rand+'/'
    n_class= 10
    n_input=1024
    batch = 10
    lr = 0.001
elif dataset=="pollen":
    folder = '../data/pollen-7-data/label'+label+'_'+rand+'/'
    n_class= 7
    n_input=625
    batch = 7
    lr = 0.001


enc_layers = [1000, 500, 250, 250, 250, n_class/ratio]
dec_layers = [250, 250, 250, 500, 1000, n_input/ratio]
my_parameter = {
        # coil:20, hela:10, pollen:7
    'batch':batch,
    'epochs':100,
    # coil:0.002, hela:0.001, pollen:0.001
    'learning_rate':lr,
    #'noise_std':0.3,
    'noise_std':0.2,
    'seed':1,
    #'unsupervised_cost_lambda': [0.1, 0.1, 0.1, 0.1, 0.1, 10., 20000.],
    'unsupervised_cost_lambda': [0.1, 0.1, 0.1, 0.1, 0.1, 20., 2000.],
    'cuda':True,
    'decay_epoch':20,
    'encoder_activations':['relu', 'relu', 'relu', 'relu', 'relu', 'softmax'],
    'encoder_sizes': [int(l*ratio) for l in enc_layers],
    'decoder_sizes': [int(l*ratio) for l in dec_layers],
    'data_dir': folder,
    'encoder_train_bn_scaling': [True, True, True, True, True, True]
}
