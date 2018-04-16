# DeepLab model
model = dict(
    name="DeepLab",
    backbone=dict(
        type='sense_resnet101',
        bn_param=dict(momentum=0.9997),
        bn_training=True,
        bn_parallel=True,
        layers_stride=[1, 2, 2, 1],
        layers_dilation=[1, 1, 1, [i * 2 for i in [1, 2, 4]]],
        mult_conv_w=[1, 1],  # lr, decay
        mult_conv_b=[2, 0],  # lr, decay
        mult_bn=[1, 1],  # lr, decay
    ),
    aspp=dict(
        channels=256,
        kernel_size=3,
        dilation_series=[6, 12, 18],
        image_pooling=True,
        bn_param=dict(momentum=0.9997),
        bn_training=True,
        bn_parallel=True,
        mult_conv_w=[1, 1],  # lr, decay
        mult_conv_b=[2, 0],  # lr, decay
        mult_bn=[1, 1],  # lr, decay
    ),
    classifier=dict(
        bn_param=dict(momentum=0.9997),
        bn_training=True,
        bn_parallel=True,
        num_class=22,
        dropout=0.1,
        mult_conv_w=[1, 1],  # lr, decay
        mult_conv_b=[2, 0],  # lr, decay
        mult_bn=[1, 1],  # lr, decay
    ))

device = [0, 1, 2, 3, 4, 5, 6, 7]
resume = ''
weight = 'outputs/deeplab_pretrain_apollo_model_best.pth.tar'
dataset = 'APOLLO'

# Training strategry
train = dict(
    batch_size=16,
    optimizer=dict(
        algorithm='SGD',
        args=dict(
            base_lr=0.007,
            momentum=0.9,
            weight_decay=0.00004,
            policy='poly',
            learning_power=0.9,
            # policy='step',
            # rate_decay_factor=0.1,
            # rate_decay_step=2000,
            max_epoch=24)),
    data_list='train',
    crop_size=[513, 513],
    rotation=[-10, 10],
    blur=True,
    crop_policy='random',
    flip=True,
    scale_factor=[0.5, 2.0])

# Testing strategry
test = dict(
    batch_size=8,
    data_list='val',
    crop_size=[721, 961],
    crop_policy='center',
    scale_factor=[1.0])

# Logging
output_dir = ''
snapshot_pref = 'deeplab_pretrain_apollo'
logging = dict(log_dir='', print_freq=50, eval_freq=1)
