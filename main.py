import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from model.coke import CoKE
# import pytorch_lightning as pl
from reader.coke_reader import KBCDataReader
from init_args import create_args
from init_config import init_coke_net_config, init_train_config
from trainer import Trainer
import math
args, logger = create_args()


def main():
    print(torch.cuda.is_available())

    # ------------
    # data
    # ------------
    train_dataset = KBCDataReader(vocab_path='./data/FB15K/vocab.txt',
                 data_path='./data/FB15K/train.coke.txt',
                 max_seq_len=args.max_seq_len,
                 batch_size=args.batch_size,
                 is_training=True,
                 shuffle=True,
                 dev_count=1,
                 epoch=args.epoch,
                 vocab_size=args.vocab_size)

    val_dataset = KBCDataReader(vocab_path='./data/FB15K/vocab.txt',
                 data_path='./data/FB15K/valid.coke.txt',
                 max_seq_len=args.max_seq_len,
                 batch_size=args.batch_size,
                 is_training=False,
                 shuffle=False,
                 dev_count=1,
                 epoch=args.epoch,
                 vocab_size=args.vocab_size)

    # validation_split = 0.2
    # dataset_size = len(train_dataset)
    # indices = list(range(dataset_size))
    # split = int(np.floor(validation_split * dataset_size))
    # shuffle_dataset = True
    # if shuffle_dataset:
    #     np.random.shuffle(indices)
    #
    # train_indices, val_indices = indices[split:], indices[:split]
    # train_sampler = SubsetRandomSampler(train_indices)
    # validation_sampler = SubsetRandomSampler(val_indices)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    # val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=validation_sampler)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    device = torch.device('cuda' if args.use_cuda else 'cpu')
    coke_config = init_coke_net_config(args, logger, print_config=True)
    model = CoKE(config=coke_config)
    model._init_parameters()
    if torch.cuda.device_count() >= 1:
        model = nn.DataParallel(model, device_ids=[0,1,2,6])
    model.to(device=device)

    # ------------
    # training
    # ------------
    train_config = init_train_config(args, logger, print_config=True)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.module.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # for name, parms in model.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data))

    # warm_up_with_cosine_lr
    t = args.warm_up_epochs  # warmup
    T = args.epoch
    n_t = 0.5
    lambda1 = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if n_t * (
                1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1 else n_t * (
                1 + math.cos(math.pi * (epoch - t) / (T - t)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    coke_trainer = Trainer(model, train_config)
    coke_trainer.load_train_data_loader(train_loader)
    coke_trainer.load_val_data_loader(val_loader)
    coke_trainer.set_loss_function(loss_function)
    coke_trainer.set_optimizer(optimizer)
    coke_trainer.set_lr_scheduler(scheduler)

    total_acc, total_loss = coke_trainer.train()

    print (total_acc, total_loss)


if __name__ == '__main__':
    main()