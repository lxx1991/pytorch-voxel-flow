# VoxelFlow model
model = dict(
    name="VoxelFlow",
    syn_type="extra",
    bn_param=dict(momentum=0.9997),
    bn_training=False,
    bn_parallel=False,
    mult_conv_w=[1, 1],  # lr, decay
    mult_conv_b=[2, 0],  # lr, decay
    mult_bn=[1, 1],  # lr, decay
)

device = [0, 1, 2, 3, 4, 5, 6, 7]
resume = 'outputs/voxelflow_model_best.pth.tar'
weight = ''
dataset = 'UCF101'

# Training strategry
train = dict(
    batch_size=64,
    optimizer=dict(
        algorithm='ADAM',
        args=dict(
            base_lr=0.0001,
            # momentum=0.9,
            weight_decay=1e-4,
            # policy='poly_epoch',
            # learning_power=0.9,
            # policy='step',
            # rate_decay_factor=0.1,
            # rate_decay_step=400,
            max_epoch=400)),
    data_list='train_motion',
    step=3,
    syn_type=model['syn_type'],
    crop_size=[256, 256],
    rotation=[-10, 10],
    crop_policy='random',
    flip=True,
    scale_factor=[1.07, 1.5])

# Testing strategry
test = dict(
    batch_size=64,
    data_list='test_motion',
    step=3,
    syn_type=model['syn_type'],
    crop_size=[256, 256],
    crop_policy='center',
    scale_factor=[1.07])

# Logging
output_dir = 'outputs'
snapshot_pref = 'voxelflow_finetune'
logging = dict(log_dir='', print_freq=50, eval_freq=1)
