import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import (
    LayerNorm,
    SinusoidalPositionalEmbedding,
    TransformerSentenceEncoder
)

from . import (
    FairseqEncoder
)

class MaskedLMEncoder(FairseqEncoder):
    """
    Encoder for Masked Language Modelling.
    """

    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        self.padding_idx = dictionary.pad()
        self.vocab_size = dictionary.__len__()
        self.max_positions = args.max_positions

        self.sentence_encoder = TransformerSentenceEncoder(
            padding_idx=self.padding_idx,
            vocab_size=self.vocab_size,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            max_seq_len=self.max_positions,
            num_segments=args.num_segment,
            use_position_embeddings=not args.no_token_positional_embeddings,
            encoder_normalize_before=args.encoder_normalize_before,
            apply_bert_init=args.apply_bert_init,
            activation_fn=args.activation_fn,
            learned_pos_embedding=args.encoder_learned_pos,
            add_bias_kv=args.bias_kv,
            add_zero_attn=args.zero_attn,
        )

        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None
        self.sentence_projection_layer = None
        self.sentence_out_dim = args.sentence_class_num
        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = not getattr(args, 'remove_head', False)

        self.masked_lm_pooler = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.pooler_activation = utils.get_activation_fn(args.pooler_activation_fn)

        self.lm_head_transform_weight = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)

        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(self.vocab_size))

            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(
                    args.encoder_embed_dim,
                    self.vocab_size,
                    bias=False
                )

            if args.sent_loss:
                self.sentence_projection_layer = nn.Linear(
                    args.encoder_embed_dim,
                    self.sentence_out_dim,
                    bias=False
                )

    def forward(self, src_tokens, segment_labels=None, **unused):
        """
        Forward pass for Masked LM encoder. This first computes the token
        embedding using the token embedding matrix, position embeddings (if
        specified) and segment embeddings (if specified).

        Here we assume that the sentence representation corresponds to the
        output of the classification_token (see bert_task or cross_lingual_lm
        task for more details).
        Args:
            - src_tokens: B x T matrix representing sentences
            - segment_labels: B x T matrix representing segment label for tokens
        Returns:
            - a tuple of the following:
                - logits for predictions in format B x T x C to be used in
                  softmax afterwards
                - a dictionary of additional data, where 'pooled_output' contains
                  the representation for classification_token and 'inner_states'
                  is a list of internal model states used to compute the
                  predictions (similar in ELMO). 'sentence_logits'
                  is the prediction logit for NSP task and is only computed if
                  this is specified in the input arguments.
        """

        inner_states, sentence_rep = self.sentence_encoder(
            src_tokens,
            segment_labels=segment_labels,
        )

        x = inner_states[-1].transpose(0, 1)
        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        pooled_output = self.pooler_activation(self.masked_lm_pooler(sentence_rep))

        # project back to size of vocabulary
        if self.share_input_output_embed \
                and hasattr(self.sentence_encoder.embed_tokens, 'weight'):
            x = F.linear(x, self.sentence_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)

        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias
        sentence_logits = None
        if self.sentence_projection_layer:
            sentence_logits = self.sentence_projection_layer(pooled_output)

        return x, {
            'inner_states': inner_states,
            'pooled_output': pooled_output,
            'sentence_logits': sentence_logits
        }

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        if isinstance(
                self.sentence_encoder.embed_positions,
                SinusoidalPositionalEmbedding
        ):
            state_dict[
                name + '.sentence_encoder.embed_positions._float_tensor'
            ] = torch.FloatTensor(1)
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if (
                    "embed_out.weight" in k or
                    "sentence_projection_layer.weight" in k or
                    "lm_output_learned_bias" in k
                ):
                    del state_dict[k]
        return state_dict
