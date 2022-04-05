import logging
import argparse
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import pytorch_lightning as pl
from reader.coke_reader import KBCDataReader
from utils.args import ArgumentGroup
from init_args import create_args
from init_config import init_coke_net_config, init_train_config

class CoKEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._max_seq_len = config['max_seq_len']
        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._voc_size = config['vocab_size']
        self._n_relation = config['num_relations']
        self._max_position_seq_len = config['max_position_embeddings']
        self._dropout = config['dropout']
        self._attention_dropout = config['attention_dropout']
        self._activation_dropout = config['activation_dropout']
        self._intermediate_size = config['intermediate_size']
        self._initializer_range = config['initializer_range']
        self._hidden_activation = config['hidden_act']
        self._soft_label = config['soft_label']
        self._weight_sharing = config['weight_sharing']

        # embedding layer
        self.embedding_layer = nn.ModuleDict({
            'word_embedding': nn.Embedding(num_embeddings=self._voc_size, embedding_dim=self._emb_size),
            'position_embedding': nn.Embedding(num_embeddings=self._max_position_seq_len, embedding_dim=self._emb_size)
        })

        # transformer block
        self.transformer_block = nn.ModuleDict({
            'transformer_encoder':
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=self._emb_size,
                        nhead=self._n_head,
                        dim_feedforward=self._intermediate_size,
                        layer_norm_eps=1e-12,
                        dropout=self._attention_dropout,
                        activation=self._hidden_activation),
                    num_layers=self._n_layer)
        })

        # classification
        self.classification_head = nn.ModuleList([
            nn.Linear(in_features=self._emb_size, out_features=self._emb_size),
            nn.GELU(),
            nn.LayerNorm(self._emb_size, eps=1e-12),
            nn.Linear(in_features=self._emb_size, out_features=self._voc_size)
        ])

        self.sub_layers = nn.ModuleDict({
            'layer_norm': nn.LayerNorm(self._emb_size, eps=1e-12),
            'dropout': nn.Dropout(p=self._dropout)
        })

    def forward(self, input_map):
        # tensors, B x seq_len x 1
        src_ids = input_map['src_ids']
        position_ids = input_map['position_ids']
        mask_pos = input_map['mask_pos']
        input_mask = input_map['input_mask']

        emb_out = self.embedding_layer["word_embedding"](src_ids.squeeze()) + \
                  self.embedding_layer["position_embedding"](position_ids.squeeze())
        emb_out = self.sub_layers["layer_norm"](emb_out)
        emb_out = self.sub_layers["dropout"](emb_out)
        emb_out = emb_out.permute(1, 0, 2)  # -> seq_len x B x E

        batch_size = src_ids.size(0)
        # attn_mask = nn.Transformer.generate_square_subsequent_mask(self._max_seq_len).cuda()
        # attn_mask = attn_mask.expand(size=[batch_size, 3, 3])
        # attn_mask = attn_mask + input_mask.squeeze() * -10000
        # attn_mask = torch.cat((attn_mask, attn_mask, attn_mask, attn_mask), dim=0)

        # attn_mask = torch.stack(tensors=[input_mask]*self._n_head, dim=1).reshape(
        #     shape=[batch_size * self._n_head, self._max_seq_len, self._max_seq_len])

        with torch.no_grad():
            self_attn_mask = torch.bmm(input_mask.squeeze(dim=1), input_mask.squeeze(dim=1).permute(0, 2, 1))
            attn_mask = torch.stack(tensors=[self_attn_mask] * self._n_head, dim=1)
            attn_mask = attn_mask.squeeze().reshape(shape=[batch_size * self._n_head, -1, self._max_seq_len])

        enc_out = self.transformer_block["transformer_encoder"](emb_out, mask=attn_mask.squeeze())
        enc_out = enc_out.reshape(shape=[-1, self._emb_size])
        logits = torch.index_select(input=enc_out, dim=0,
                                    index=mask_pos.squeeze())

        for idx, layer in enumerate(self.classification_head):
            # if idx == 3 and self._weight_sharing:
            #     layer.weight = nn.Parameter(self.embedding_layer["word_embedding"].weight)
            logits = layer(logits)

        return logits

class EntityClassifier(pl.LightningModule):
    def __init__(self, CoKEModel, learning_rate=5e-4):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = CoKEModel
        self.learning_rate = learning_rate
    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding
    # soft label
    def training_step(self, batch, batch_idx):
        src_ids, position_ids, input_mask, mask_pos, mask_label = batch
        input_x = {
            'src_ids': src_ids,
            'position_ids': position_ids,
            'input_mask': input_mask,
            'mask_pos': mask_pos,
            'mask_label': mask_label
        }
        y_hat = self.backbone(input_x)
        loss = F.cross_entropy(
            input=y_hat,
            target=mask_label.squeeze(),
            label_smoothing=0.8
        )
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src_ids, position_ids, input_mask, mask_pos, mask_label = batch
        input_x = {
            'src_ids': src_ids,
            'position_ids': position_ids,
            'input_mask': input_mask,
            'mask_pos': mask_pos,
            'mask_label': mask_label
        }
        y_hat = self.backbone(input_x)
        loss = F.cross_entropy(
            input=y_hat,
            target=mask_label.squeeze(),
            label_smoothing=0.8
        )
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        src_ids, position_ids, input_mask, mask_pos, mask_label = batch
        input_x = {
            'src_ids': src_ids,
            'position_ids': position_ids,
            'input_mask': input_mask,
            'mask_pos': mask_pos,
            'mask_label': mask_label
        }
        y_hat = self.backbone(input_x)
        loss = F.cross_entropy(
            input=y_hat,
            target=mask_label.squeeze(),
            label_smoothing=0.8
        )
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        # return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # train_g.add_arg("learning_rate", float, 5e-5, "Learning rate used to train with warmup.")
        # train_g.add_arg("lr_scheduler", str, "linear_warmup_decay", "scheduler of learning rate.",
        #                 choices=['linear_warmup_decay', 'noam_decay'])
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def main():
    pl.seed_everything(1234)
    print(torch.cuda.is_available())
    args, logger = create_args()
    # ------------
    # args
    # ------------
    coke_config = init_coke_net_config(args, logger, print_config=True)
    # ------------
    # data
    # ------------
    train_dataset = KBCDataReader(vocab_path='./data/FB15K/vocab.txt',
                 data_path='./data/FB15K/train.coke.txt',
                 max_seq_len=3,
                 batch_size=args.batch_size,
                 is_training=True,
                 shuffle=True,
                 dev_count=1,
                 epoch=args.epoch,
                 vocab_size=16396)
    val_dataset = KBCDataReader(vocab_path='./data/FB15K/vocab.txt',
                                  data_path='./data/FB15K/valid.coke.txt',
                                  max_seq_len=3,
                                  batch_size=args.batch_size,
                                  is_training=True,
                                  shuffle=True,
                                  dev_count=1,
                                  epoch=args.epoch,
                                  vocab_size=16396)
    test_dataset = KBCDataReader(vocab_path='./data/FB15K/vocab.txt',
                 data_path='./data/FB15K/test.coke.txt',
                 max_seq_len=3,
                 batch_size=args.batch_size,
                 is_training=False,
                 shuffle=True,
                 dev_count=1,
                 epoch=10,
                 vocab_size=16396)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    # ------------
    # model
    # ------------

    model = EntityClassifier(CoKEModel(config=coke_config), args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)
    print(result.shape)


if __name__ == '__main__':
    main()