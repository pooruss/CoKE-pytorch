import torch
import torch.nn as nn
import torch.nn.functional as F

class CoKE(nn.Module):
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
            'word_embedding':nn.Embedding(num_embeddings=self._voc_size, embedding_dim=self._emb_size),
            'position_embedding':nn.Embedding(num_embeddings=self._max_position_seq_len, embedding_dim=self._emb_size)
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
            'layer_norm':nn.LayerNorm(self._emb_size, eps=1e-12),
            'dropout':nn.Dropout(p=self._dropout)
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

        attn_mask = torch.stack(tensors=[input_mask]*self._n_head, dim=1).reshape(
            shape=[batch_size * self._n_head, self._max_seq_len, self._max_seq_len])

        enc_out = self.transformer_block["transformer_encoder"](emb_out, mask=attn_mask.squeeze())
        enc_out = enc_out.reshape(shape=[-1, self._emb_size])
        logits = torch.index_select(input=enc_out, dim=0,
                                       index=mask_pos.squeeze())

        for idx,layer in enumerate(self.classification_head):
            if idx == 3 and self._weight_sharing:
                layer.weight = nn.Parameter(self.embedding_layer["word_embedding"].weight)
            logits = layer(logits)

        return logits

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, self._initializer_range)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, self._initializer_range)
            # elif isinstance(m, nn.LayerNorm):
            #     nn.init.constant_(m.weight, 1.0)
            #     nn.init.constant_(m.bias, 0.0)
                # nn.init.constant_(m.bias, 0)

    def compute_loss(self, logits, targets):
        loss = F.cross_entropy(
            input=logits,
            target=targets,
            label_smoothing=self._soft_label
        )
        return loss
