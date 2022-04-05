#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" data reader for CoKE
"""

from __future__ import print_function
from __future__ import division
import argparse
from utils.args import ArgumentGroup
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import six
import collections
import logging
import torch
from reader.batching import prepare_batch_data

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())

RawExample = collections.namedtuple("RawExample", ["token_ids", "mask_type"])

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")
#def printable_text(text):
#    """Returns text encoded in a way suitable for print or `tf.logging`."""
#
#    # These functions want `str` for both Python2 and Python3, but in one case
#    # it's a Unicode string and in the other it's a byte string.
#    if six.PY3:
#        if isinstance(text, str):
#            return text
#        elif isinstance(text, bytes):
#            return text.decode("utf-8", "ignore")
#        else:
#            raise ValueError("Unsupported string type: %s" % (type(text)))
#    elif six.PY2:
#        if isinstance(text, str):
#            return text
#        elif isinstance(text, unicode):
#            return text.encode("utf-8")
#        else:
#            raise ValueError("Unsupported string type: %s" % (type(text)))
#    else:
#        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    fin = open(vocab_file)
    for num, line in enumerate(fin):
        items = line.strip().split("\t")
        if len(items) > 2:
            break
        token = items[0]
        index = items[1] if len(items) == 2 else num
        token = token.strip()
        vocab[token] = int(index)
    return vocab


#def convert_by_vocab(vocab, items):
#    """Converts a sequence of [tokens|ids] using the vocab."""
#    output = []
#    for item in items:
#        output.append(vocab[item])
#    return output


def convert_tokens_to_ids(vocab, tokens):
    """Converts a sequence of tokens into ids using the vocab."""
    output = []
    for item in tokens:
        output.append(vocab[item])
    return output


class KBCDataReader(Dataset):
    """ DataReader
    """

    def __init__(self,
                 vocab_path,
                 data_path,
                 max_seq_len=3,
                 batch_size=4096,
                 is_training=True,
                 shuffle=True,
                 dev_count=1,
                 epoch=10,
                 vocab_size=-1):
        self.vocab = load_vocab(vocab_path)
        if vocab_size > 0:
            assert len(self.vocab) == vocab_size, \
                "Assert Error! Input vocab_size(%d) is not consistant with voab_file(%d)" % \
                (vocab_size, len(self.vocab))
        self.pad_id = self.vocab["[PAD]"]
        self.mask_id = self.vocab["[MASK]"]

        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        self.is_training = is_training
        self.shuffle = shuffle
        self.dev_count = dev_count
        self.epoch = epoch
        if not is_training:
            self.shuffle = False
            self.dev_count = 1
            self.epoch = 1

        self.examples = self.read_example(data_path)
        self.total_instance = len(self.examples)

        self.current_epoch = -1
        self.current_instance_index = -1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        token_ids, mask_type = example[0],example[1]
        example_out = [token_ids] + [mask_type]
        example_data = prepare_batch_data(
            [example_out],
            max_len=self.max_seq_len,
            pad_id=self.pad_id,
            mask_id=self.mask_id)
        src_id, pos_id, input_mask, mask_pos, mask_label = example_data[0], example_data[1], example_data[2], example_data[3], example_data[4]

        return torch.tensor(src_id, dtype=torch.long), \
               torch.tensor(pos_id, dtype=torch.long), \
               torch.tensor(input_mask, dtype=torch.float16), \
               torch.tensor(mask_pos, dtype=torch.long), \
                torch.tensor(mask_label, dtype=torch.long)



    def get_progress(self):
        """return current progress of traning data
        """
        return self.current_instance_index, self.current_epoch

    def line2tokens(self, line):
        tokens = line.split("\t")
        return tokens

    def read_example(self, input_file):
        """Reads the input file into a list of examples."""
        examples = []
        with open(input_file, "r") as f:
            for line in f.readlines():
                line = convert_to_unicode(line.strip())
                tokens = self.line2tokens(line)
                assert len(tokens) <= (self.max_seq_len + 1), \
                    "Expecting at most [max_seq_len + 1]=%d tokens each line, current tokens %d" \
                    % (self.max_seq_len + 1, len(tokens))
                token_ids = convert_tokens_to_ids(self.vocab, tokens[:-1])
                if len(token_ids) <= 0:
                    continue
                examples.append(
                    RawExample(
                        token_ids=token_ids, mask_type=tokens[-1]))
                # if len(examples) <= 10:
                #     logger.info("*** Example ***")
                #     logger.info("tokens: %s" % " ".join([printable_text(x) for x in tokens]))
                #     logger.info("token_ids: %s" % " ".join([str(x) for x in token_ids]))
        return examples

    def data_generator(self):
        """ wrap the batch data generator
        """
        range_list = [i for i in range(self.total_instance)]

        def wrapper():
            """ wrapper batch data
            """

            def reader():
                for epoch_index in range(self.epoch):
                    self.current_epoch = epoch_index
                    if self.shuffle is True:
                        np.random.shuffle(range_list)
                    for idx, sample in enumerate(range_list):
                        self.current_instance_index = idx
                        yield self.examples[sample]

            def batch_reader(reader, batch_size):
                """reader generator for batches of examples
                :param reader: reader generator for one example
                :param batch_size: int batch size
                :return: a list of examples for batch data
                """
                batch = []
                for example in reader():
                    token_ids = example.token_ids
                    mask_type = example.mask_type
                    example_out = [token_ids] + [mask_type]
                    to_append = len(batch) < batch_size
                    if to_append is False:
                        yield batch
                        batch = [example_out]
                    else:
                        batch.append(example_out)
                if len(batch) > 0:
                    yield batch

            all_device_batches = []
            for batch_data in batch_reader(reader, self.batch_size):
                batch_data = prepare_batch_data(
                    batch_data,
                    max_len=self.max_seq_len,
                    pad_id=self.pad_id,
                    mask_id=self.mask_id)
                if len(all_device_batches) < self.dev_count:
                    all_device_batches.append(batch_data)

                if len(all_device_batches) == self.dev_count:
                    for batch in all_device_batches:
                        yield batch
                    all_device_batches = []

        return wrapper


class PathqueryDataReader(KBCDataReader):
    def __init__(self,
                 vocab_path,
                 data_path,
                 max_seq_len=3,
                 batch_size=4096,
                 is_training=True,
                 shuffle=True,
                 dev_count=1,
                 epoch=10,
                 vocab_size=-1):

        KBCDataReader.__init__(self, vocab_path, data_path, max_seq_len,
                               batch_size, is_training, shuffle, dev_count,
                               epoch, vocab_size)

    def line2tokens(self, line):
        tokens = []
        s, path, o, mask_type = line.split("\t")
        path_tokens = path.split(",")
        tokens.append(s)
        tokens.extend(path_tokens)
        tokens.append(o)
        tokens.append(mask_type)
        return tokens

# below is for forward test

class CoKEModel(nn.Module):
    def __init__(self, config, soft_label=0.9):
        super().__init__()
        self.batch_size = config['batch_size']
        self._max_seq_len = config['max_seq_len']
        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._voc_size = config['vocab_size']
        self._n_relation = config['num_relations']
        self._max_position_seq_len = config['max_position_embeddings']
        self._attention_dropout = config['attention_probs_dropout_prob']
        self._intermediate_size = config['intermediate_size']
        self._soft_label = soft_label

        self.layer_norm = nn.LayerNorm(self._emb_size)
        self.dropout = nn.Dropout(p=self._attention_dropout)
        self.word_embedding = nn.Embedding(num_embeddings=self._voc_size,embedding_dim=self._emb_size)
        # 此处需确认sparse
        self.posititon_embedding = nn.Embedding(num_embeddings=self._max_position_seq_len,embedding_dim=self._emb_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._emb_size,
            nhead=self._n_head,
            dim_feedforward=self._intermediate_size,
            layer_norm_eps=1e-12,
            dropout=self._attention_dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,num_layers=self._n_layer)
        self.feed_forward = nn.Linear(in_features=self._emb_size, out_features=self._emb_size)
        self.classification = nn.Linear(in_features=self._emb_size, out_features=self._voc_size, bias=True)
        # self._build_model(src_ids, position_ids, input_mask)

    def forward(self, src_ids, position_ids, input_mask, mask_pos):
        # mask 1
        # batch_size = src_ids.size(0)
        # attn_mask = torch.stack(tensors=[input_mask] * self._n_head, dim=1).reshape(
        #     shape=[batch_size * self._n_head, self._max_seq_len, self._max_seq_len])
        # mask 2
        self_attn_mask = torch.bmm(input_mask.squeeze(dim=1), input_mask.squeeze(dim=1).permute(0, 2, 1))
        attn_mask = torch.stack(tensors=[self_attn_mask] * self._n_head, dim=1)
        attn_mask = attn_mask.squeeze().reshape(shape=[self.batch_size * self._n_head, -1, self._max_seq_len])

        word_emb_out = self.word_embedding(src_ids.squeeze())
        position_emb_out = self.posititon_embedding(position_ids.squeeze())
        emb_out = word_emb_out + position_emb_out
        emb_out = self.layer_norm(emb_out)
        emb_out = self.dropout(emb_out)
        emb_out = emb_out.permute(1, 0, 2)  # -> seq_len x B x E
        enc_out = self.transformer_encoder(emb_out, mask=attn_mask.float())
        enc_out = torch.reshape(enc_out, (-1, self._emb_size))
        mask_feat = torch.index_select(input=enc_out, dim=0, index=mask_pos.squeeze())
        out = self.feed_forward(mask_feat)
        out = self.layer_norm(out)
        out = self.classification(out)
        return out

def init_coke_net_config(args, print_config = True):
    config = dict()
    config["batch_size"] = args.batch_size
    config["max_seq_len"] = args.max_seq_len
    config["hidden_size"] = args.hidden_size
    config["num_hidden_layers"] = args.num_hidden_layers
    config["num_attention_heads"] = args.num_attention_heads
    config["vocab_size"] = args.vocab_size
    config["num_relations"] = args.num_relations
    config["max_position_embeddings"] = args.max_position_embeddings
    config["hidden_act"] = args.hidden_act
    config["hidden_dropout_prob"] = args.hidden_dropout_prob
    config["attention_probs_dropout_prob"] = args.attention_probs_dropout_prob
    config["initializer_range"] = args.initializer_range
    config["intermediate_size"] = args.intermediate_size
    config["gpus"] = 1
    if print_config is True:
        logger.info('----------- CoKE Network Configuration -------------')
        for arg, value in config.items():
            logger.info('%s: %s' % (arg, value))
        logger.info('------------------------------------------------')
    return config


# for forward test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
    model_g.add_arg("hidden_size", int, 256, "CoKE model config: hidden size, default 256")
    model_g.add_arg("num_hidden_layers", int, 6, "CoKE model config: num_hidden_layers, default 6")
    model_g.add_arg("num_attention_heads", int, 4, "CoKE model config: num_attention_heads, default 4")
    model_g.add_arg("vocab_size", int, 16396, "CoKE model config: vocab_size")
    model_g.add_arg("num_relations", int, None, "CoKE model config: vocab_size")
    model_g.add_arg("max_position_embeddings", int, 10, "CoKE model config: max_position_embeddings")
    model_g.add_arg("hidden_act", str, "gelu", "CoKE model config: hidden_ac, default gelu")
    model_g.add_arg("hidden_dropout_prob", float, 0.1, "CoKE model config: attention_probs_dropout_prob, default 0.1")
    model_g.add_arg("attention_probs_dropout_prob", float, 0.1,
                    "CoKE model config: attention_probs_dropout_prob, default 0.1")
    model_g.add_arg("initializer_range", int, 0.02, "CoKE model config: initializer_range")
    model_g.add_arg("intermediate_size", int, 512, "CoKE model config: intermediate_size, default 512")

    model_g.add_arg("init_checkpoint", str, None, "Init checkpoint to resume training from, or for prediction only")
    model_g.add_arg("init_pretraining_params", str, None,
                    "Init pre-training params which preforms fine-tuning from. If the "
                    "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
    model_g.add_arg("checkpoints", str, "checkpoints", "Path to save checkpoints.")
    model_g.add_arg("weight_sharing", bool, True, "If set, share weights between word embedding and masked lm.")

    train_g = ArgumentGroup(parser, "training", "training options.")
    train_g.add_arg("epoch", int, 100, "Number of epoches for training.")
    train_g.add_arg("learning_rate", float, 5e-5, "Learning rate used to train with warmup.")
    train_g.add_arg("lr_scheduler", str, "linear_warmup_decay", "scheduler of learning rate.",
                    choices=['linear_warmup_decay', 'noam_decay'])
    train_g.add_arg("soft_label", float, 0.9, "Value of soft labels for loss computation")
    train_g.add_arg("weight_decay", float, 0.01, "Weight decay rate for L2 regularizer.")
    train_g.add_arg("warmup_proportion", float, 0.1,
                    "Proportion of training steps to perform linear learning rate warmup for.")
    train_g.add_arg("use_ema", bool, True, "Whether to use ema.")
    train_g.add_arg("ema_decay", float, 0.9999, "Decay rate for expoential moving average.")
    train_g.add_arg("use_fp16", bool, False, "Whether to use fp16 mixed precision training.")
    train_g.add_arg("loss_scaling", float, 1.0,
                    "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")

    log_g = ArgumentGroup(parser, "logging", "logging related.")
    log_g.add_arg("skip_steps", int, 1000, "The steps interval to print loss.")
    log_g.add_arg("verbose", bool, False, "Whether to output verbose log.")

    data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
    data_g.add_arg("dataset", str, "", "dataset name")
    data_g.add_arg("train_file", str, None, "Data for training.")
    data_g.add_arg("sen_candli_file", str, None,
                   "sentence_candicate_list file for path query evaluation. Only used for path query datasets")
    data_g.add_arg("sen_trivial_file", str, None,
                   "trivial sentence file for pathquery evaluation. Only used for path query datasets")
    data_g.add_arg("predict_file", str, None, "Data for predictions.")
    data_g.add_arg("vocab_path", str, None, "Path to vocabulary.")
    data_g.add_arg("true_triple_path", str, None, "Path to all true triples. Only used for KBC evaluation.")
    data_g.add_arg("max_seq_len", int, 3, "Number of tokens of the longest sequence.")
    data_g.add_arg("batch_size", int, 64, "Total examples' number in batch for training. see also --in_tokens.")
    data_g.add_arg("in_tokens", bool, False,
                   "If set, the batch size will be the maximum number of tokens in one batch. "
                   "Otherwise, it will be the maximum number of examples in one batch.")

    run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
    run_type_g.add_arg("do_train", bool, False, "Whether to perform training.")
    run_type_g.add_arg("do_predict", bool, False, "Whether to perform prediction.")
    run_type_g.add_arg("use_cuda", bool, True, "If set, use GPU for training, default is True.")
    run_type_g.add_arg("use_fast_executor", bool, False, "If set, use fast parallel executor (in experiment).")
    run_type_g.add_arg("num_iteration_per_drop_scope", int, 1,
                       "Ihe iteration intervals to clean up temporary variables.")
    args = parser.parse_args()
    config = init_coke_net_config(args)

    model = CoKEModel(config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.00001)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer)

    train_dataset = KBCDataReader(vocab_path='../data/FB15K/vocab.txt',
                     data_path='../data/FB15K/train.coke.txt',
                     max_seq_len=3,
                     batch_size=args.batch_size,
                     is_training=True,
                     shuffle=True,
                     dev_count=1,
                     epoch=10,
                     vocab_size=16396)
    # example_length = 5, [src_id, pos_id, input_mask, mask_pos, mask_label]
    iter = train_dataset.data_generator()
    import torch.nn.functional as F
    for idx,data in enumerate(iter()):
        src_id, pos_id, input_mask, mask_pos, mask_label = data
        src_id, pos_id, input_mask, mask_pos, mask_label = \
        torch.tensor(src_id, dtype=torch.long), \
        torch.tensor(pos_id, dtype=torch.long), \
        torch.tensor(input_mask, dtype=torch.float16), \
        torch.tensor(mask_pos, dtype=torch.long), \
        torch.tensor(mask_label, dtype=torch.long)
        # print(input_mask.shape)      # [64, 3, 1]
        output = model(src_id, pos_id, input_mask, mask_pos)
        # output = F.softmax(output)
        # print(mask_label.squeeze().shape)     # [64]
        # print(output)     # [64, 16396]
        print(src_id[0])
        print(mask_pos[0])
        print(mask_label.squeeze())
        loss = F.cross_entropy(input=output, target=mask_label, label_smoothing=0.8)

        print('### {} data ###'.format(str(idx)))
        print (loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # example = train_dataset.__getitem__(0)
    # src_id, pos_id, input_mask, mask_label, mask_pos = example
    # print(src_id.shape)
