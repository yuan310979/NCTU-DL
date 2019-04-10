import os

activation = ['ReLU', 'LeakyReLU', 'ELU']
batch_size = [64, 128, 256, 512, 1024, 2048]
lr = [1e-2, 1e-3, 1e-4]
model = ['EEGNet', 'DeepConvNet']

for _m in model:
    for _a in activation:
        for _b in batch_size:
            for _lr in lr:
                filename = f"{_m}_{_a}_{_b}_{_lr}"
                print(filename)
                os.system(f'python train.py -a {_a} -b {_b} --lr {_lr} -m {_m} --epochs 2000 --checkpoint {filename}')
