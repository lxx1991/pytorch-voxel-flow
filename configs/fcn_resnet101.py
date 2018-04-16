# FCN model
model = dict(
    name="FCN",
    backbone=dict(
        type='sense_resnet101',
        bn_param=dict(momentum=0.95),
        bn_training=True,
        bn_parallel=True,
        layers_stride=[1, 2, 1, 1],
        layers_dilation=[1, 1, 2, 4],
        mult_conv_w=[1, 1],  # lr, decay
        mult_conv_b=[2, 0],  # lr, decay
        mult_bn=[1, 1],  # lr, decay
    ),
    classifier=dict(
        bn_param=dict(momentum=0.95),
        bn_training=True,
        bn_parallel=True,
        num_class=21,
        dropout=0.1,
        mult_conv_w=[10, 1],  # lr, decay
        mult_conv_b=[20, 0],  # lr, decay
        mult_bn=[10, 1],  # lr, decay
    ))

device = [0, 1, 2, 3, 4, 5, 6, 7]
resume = ''
weight = ''
dataset = 'VOCAug'

# Training strategry
train = dict(
    batch_size=16,
    optimizer=dict(
        algorithm='SGD',
        args=dict(
            base_lr=0.01,
            momentum=0.9,
            weight_decay=1e-4,
            policy='poly_epoch',
            learning_power=0.9,
            # policy='step',
            # rate_decay_factor=0.1,
            # rate_decay_step=2000,
            max_epoch=48)),
    data_list='train',
    crop_size=[473, 473],
    rotation=[-10, 10],
    blur=True,
    crop_policy='random',
    flip=True,
    scale_factor=[0.5, 2.0])

# Testing strategry
test = dict(
    batch_size=16,
    data_list='val',
    crop_size=[513, 513],
    crop_policy='center',
    scale_factor=[1.0])

# Logging
output_dir = 'outputs'
snapshot_pref = 'fcn'
logging = dict(log_dir='', print_freq=50, eval_freq=1)
