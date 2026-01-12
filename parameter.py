

Inputfile_name = 'Marm.bin'
Inputfile_path = './data/'
fwi_model_path = './data/deterministic_fwi_result.npy'


num_particles       = 50
nepoch              = 100
nz                  = 221
nx                  = 601
num_dims            = 2
weight_decay        = 0
learning_rate       = 0.005            # learning rate of pretraining
pretrain_num_iter   = 2000             # number of iteration for pretraining
dip_pretrain_loss   = 'L1'             # loss type for pretraining
OPTIMIZER           = 'adam'           # optimizer
k_channels          = 128              # number of channels of input
dip_pretrain_loss   = 'L1'             # loss type for pretraining

cnnmodel_path ='./Pretrained-network/Dip_pretrain_small_model_iter'
Figurepath    = './Conditional-Unet-output/'
