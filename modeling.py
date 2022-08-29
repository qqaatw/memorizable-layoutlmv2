""" Memorizable-LayoutLMv2 model."""

import math
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, torch_int_div
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_detectron2_available,
    logging,
    replace_return_docstrings,
    requires_backends,
    ModelOutput,
)
from transformers.models.layoutlmv2.configuration_layoutlmv2 import LayoutLMv2Config


# soft dependency
if is_detectron2_available():
    import detectron2
    from detectron2.modeling import META_ARCH_REGISTRY

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LayoutLMv2Config"

LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/layoutlmv2-base-uncased",
    "microsoft/layoutlmv2-large-uncased",
    # See all LayoutLMv2 models at https://huggingface.co/models?filter=layoutlmv2
]

@dataclass
class MemorizableLayoutLMv2ModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor
    pooler_output: torch.FloatTensor = None
    mems: List[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class LayoutLMv2Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(LayoutLMv2Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def _calc_spatial_position_embeddings(self, bbox):
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The `bbox` coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        return spatial_position_embeddings


class MemoryExtendedPositionalEmbedding(nn.Module):
    """Sinusoid encoding. Copied from modeling_transfo_xl.py->PositionalEmbedding
    """
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class LayoutLMv2RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(self, config, r_r_bias=None, r_w_bias=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.config = config
        self.fast_qkv = config.fast_qkv
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if config.fast_qkv:
            self.qkv_linear = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
            self.v_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
        else:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale = 1 / (self.attention_head_size**0.5)

        # memory specific
        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias
        self.r_net = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        # end of memory specific
    
    def _rel_shift(self, x):
        # x = b n q k
        zero_pad_shape = x.size()[0:3] + (1,) # b n q 1
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=3) # b n q (k + 1)

        x_padded_shape = x.size()[:2] + (x.size(3) + 1, x.size(2))
        x_padded = x_padded.view(*x_padded_shape)
        x = x_padded[:, :, 1:, :].view_as(x) # b n q k

        return x

    def compute_qkv(self, hidden_states):
        if self.fast_qkv:
            qkv = self.qkv_linear(hidden_states)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            if q.ndimension() == self.q_bias.ndimension():
                q = q + self.q_bias
                v = v + self.v_bias
            else:
                _sz = (1,) * (q.ndimension() - 1) + (-1,)
                q = q + self.q_bias.view(*_sz)
                v = v + self.v_bias.view(*_sz)
        else:
            q = self.query(hidden_states)
            k = self.key(hidden_states)
            v = self.value(hidden_states)
        
        return q, k, v

    def forward(
        self,
        hidden_states,
        mem_pos_embedding,
        attention_mask,
        head_mask=None,
        mems=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        qlen, rlen, bsz = hidden_states.size(1), mem_pos_embedding.size(0), hidden_states.size(0)
        mlen = 0
        vlen = 49
        if mems is not None:
            mlen =  mems.size(1)
            vlen = qlen - (attention_mask.size(1) - mlen)
            cat = torch.cat([mems, hidden_states], 1) # concat along the length axis
            q, k, v = self.compute_qkv(cat)
            q = q[:,-qlen:] # strip memory for query
        else:
            q, k, v = self.compute_qkv(cat)
        

        # [bsz, seq_len, hidden_size] -> [bsz, seq_len, num_attn_heads, attn_size]
        query_layer = q.view(q.size()[:-1] + (self.num_attention_heads, self.attention_head_size))
        # W_(k,E) E_(x, j) <- content-based key
        key_layer = k.view(k.size()[:-1] + (self.num_attention_heads, self.attention_head_size))
        value_layer = v.view(v.size()[:-1] + (self.num_attention_heads, self.attention_head_size))
        
        # W_(k,R) R_(i-j) <- location-based key
        r_head_k = self.r_net(mem_pos_embedding)
        r_head_k = r_head_k.view(rlen, self.num_attention_heads, self.attention_head_size)

        # q^n_(t,i) + u
        rw_head_q = query_layer + self.r_w_bias
        # term a + c = q^n_(t,i) k^n_(t,j) + u k^n_(t,j)
        AC = torch.einsum("bind,bjnd->bnij", (rw_head_q, key_layer))  
        # q^n_(t,i) + v
        rr_head_q = query_layer + self.r_r_bias
        # term b + d = q^n_(t,i) W_(k,R) R_(i-j) + v W_(k,R) R_(i-j)
        BD = torch.einsum("bind,jnd->bnij", (rr_head_q, r_head_k))
        # A linear approach to compute W_(k,R) R_(i-j) for all (i, j)
        # See Appendix B of https://arxiv.org/abs/1901.02860
        BD = self._rel_shift(BD)

        attention_scores = (AC + BD)
        attention_scores = attention_scores * self.scale

        if self.has_relative_attention_bias:
            attention_scores[:, :, :, -qlen:] += rel_pos
        if self.has_spatial_attention_bias:
            attention_scores[:, :, :, -qlen:] += rel_2d_pos

        # 2d attention head for q & k
        assert attention_mask.dim() == 2

        mask_value = torch.finfo(attention_scores.dtype).min
        if attention_mask is not None and torch.sum(attention_mask).item():
            attention_mask = attention_mask == 1  # Switch to bool
            attention_scores = attention_scores.float().masked_fill(attention_mask, mask_value).type_as(attention_scores)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).type_as(value_layer)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.einsum("bnij,bjnd->bind", attention_probs, value_layer).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class LayoutLMv2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = LayoutLMv2RelPartialLearnableMultiHeadAttn(config)
        self.output = LayoutLMv2SelfOutput(config)

    def forward(
        self,
        hidden_states,
        mem_pos_embedding,
        attention_mask=None,
        head_mask=None,
        mems=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        self_outputs = self.self(
            hidden_states,
            mem_pos_embedding, # relative position embedding
            attention_mask,
            head_mask,
            mems,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class LayoutLMv2SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->LayoutLMv2
class LayoutLMv2Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->LayoutLM
class LayoutLMv2Output(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayoutLMv2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LayoutLMv2Attention(config)
        self.intermediate = LayoutLMv2Intermediate(config)
        self.output = LayoutLMv2Output(config)

    def forward(
        self,
        hidden_states,
        mem_pos_embedding,
        attention_mask=None,
        head_mask=None,
        mems=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            mem_pos_embedding,
            attention_mask,
            head_mask,
            mems,
            output_attentions=output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for small
    absolute relative_position and larger buckets for larger absolute relative_positions. All relative positions
    >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket. This should
    allow for more graceful generalization to longer sequences than the model has been trained on.

    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """

    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret


class LayoutLMv2Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([LayoutLMv2Layer(config) for _ in range(config.num_hidden_layers)])

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_onehot_size = config.rel_pos_bins
            self.rel_pos_bias = nn.Linear(self.rel_pos_onehot_size, config.num_attention_heads, bias=False)

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_2d_pos_onehot_size = config.rel_2d_pos_bins
            self.rel_pos_x_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)
            self.rel_pos_y_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)

        self.gradient_checkpointing = False

    def _calculate_1d_position_embeddings(self, hidden_states, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        rel_pos = nn.functional.one_hot(rel_pos, num_classes=self.rel_pos_onehot_size).type_as(hidden_states)
        rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        rel_pos = rel_pos.contiguous()
        return rel_pos

    def _calculate_2d_position_embeddings(self, hidden_states, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_x = nn.functional.one_hot(rel_pos_x, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_y = nn.functional.one_hot(rel_pos_y, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_x = self.rel_pos_x_bias(rel_pos_x).permute(0, 3, 1, 2)
        rel_pos_y = self.rel_pos_y_bias(rel_pos_y).permute(0, 3, 1, 2)
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def forward(
        self,
        hidden_states,
        mem_pos_embedding,
        attention_mask=None,
        head_mask=None,
        mems=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        bbox=None,
        position_ids=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        rel_pos = (
            self._calculate_1d_position_embeddings(hidden_states, position_ids)
            if self.has_relative_attention_bias
            else None
        )
        rel_2d_pos = (
            self._calculate_2d_position_embeddings(hidden_states, bbox) if self.has_spatial_attention_bias else None
        )

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    mem_pos_embedding,
                    attention_mask,
                    layer_head_mask,
                    mems[i],
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    mem_pos_embedding,
                    attention_mask,
                    layer_head_mask,
                    mems[i],
                    output_attentions,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class LayoutLMv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LayoutLMv2Config
    pretrained_model_archive_map = LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "layoutlmv2"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LayoutLMv2Encoder):
            module.gradient_checkpointing = value


def my_convert_sync_batchnorm(module, process_group=None):
    # same as `nn.modules.SyncBatchNorm.convert_sync_batchnorm` but allowing converting from `detectron2.layers.FrozenBatchNorm2d`
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        return nn.modules.SyncBatchNorm.convert_sync_batchnorm(module, process_group)
    module_output = module
    if isinstance(module, detectron2.layers.FrozenBatchNorm2d):
        module_output = torch.nn.SyncBatchNorm(
            num_features=module.num_features,
            eps=module.eps,
            affine=True,
            track_running_stats=True,
            process_group=process_group,
        )
        module_output.weight = torch.nn.Parameter(module.weight)
        module_output.bias = torch.nn.Parameter(module.bias)
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=module.running_mean.device)
    for name, child in module.named_children():
        module_output.add_module(name, my_convert_sync_batchnorm(child, process_group))
    del module
    return module_output


class LayoutLMv2VisualBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config.get_detectron2_config()
        meta_arch = self.cfg.MODEL.META_ARCHITECTURE
        model = META_ARCH_REGISTRY.get(meta_arch)(self.cfg)
        assert isinstance(model.backbone, detectron2.modeling.backbone.FPN)
        self.backbone = model.backbone

        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        self.register_buffer(
            "pixel_mean",
            torch.Tensor(self.cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1),
        )
        self.register_buffer("pixel_std", torch.Tensor(self.cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1))
        self.out_feature_key = "p2"
        if torch.are_deterministic_algorithms_enabled():
            logger.warning("using `AvgPool2d` instead of `AdaptiveAvgPool2d`")
            input_shape = (224, 224)
            backbone_stride = self.backbone.output_shape()[self.out_feature_key].stride
            self.pool = nn.AvgPool2d(
                (
                    math.ceil(math.ceil(input_shape[0] / backbone_stride) / config.image_feature_pool_shape[0]),
                    math.ceil(math.ceil(input_shape[1] / backbone_stride) / config.image_feature_pool_shape[1]),
                )
            )
        else:
            self.pool = nn.AdaptiveAvgPool2d(config.image_feature_pool_shape[:2])
        if len(config.image_feature_pool_shape) == 2:
            config.image_feature_pool_shape.append(self.backbone.output_shape()[self.out_feature_key].channels)
        assert self.backbone.output_shape()[self.out_feature_key].channels == config.image_feature_pool_shape[2]

    def forward(self, images):
        images_input = ((images if torch.is_tensor(images) else images.tensor) - self.pixel_mean) / self.pixel_std
        features = self.backbone(images_input)
        features = features[self.out_feature_key]
        features = self.pool(features).flatten(start_dim=2).transpose(1, 2).contiguous()
        return features

    def synchronize_batch_norm(self):
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() > -1
        ):
            raise RuntimeError("Make sure torch.distributed is set up properly.")

        self_rank = torch.distributed.get_rank()
        node_size = torch.cuda.device_count()
        world_size = torch.distributed.get_world_size()
        if not (world_size & node_size == 0):
            raise RuntimeError("Make sure the number of processes can be divided by the number of nodes")

        node_global_ranks = [list(range(i * node_size, (i + 1) * node_size)) for i in range(world_size // node_size)]
        sync_bn_groups = [
            torch.distributed.new_group(ranks=node_global_ranks[i]) for i in range(world_size // node_size)
        ]
        node_rank = self_rank // node_size

        self.backbone = my_convert_sync_batchnorm(self.backbone, process_group=sync_bn_groups[node_rank])


LAYOUTLMV2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LayoutLMv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

LAYOUTLMV2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `{0}`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`LayoutLMv2Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

        bbox (`torch.LongTensor` of shape `({0}, 4)`, *optional*):
            Bounding boxes of each input sequence tokens. Selected in the range `[0,
            config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
            format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
            y1) represents the position of the lower right corner.

        image (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `detectron.structures.ImageList` whose `tensors` is of shape `(batch_size, num_channels, height, width)`):
            Batch of document images.

        attention_mask (`torch.FloatTensor` of shape `{0}`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `{0}`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `{0}`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class LayoutLMv2Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@add_start_docstrings(
    "The bare MemorizableLayoutLMv2Model Model transformer outputting raw hidden-states without any specific head on top.",
    LAYOUTLMV2_START_DOCSTRING,
)
class MemorizableLayoutLMv2Model(LayoutLMv2PreTrainedModel):
    def __init__(self, config, mem_len=512, clamp_len=1000, same_length=True):
        requires_backends(self, "detectron2")
        super().__init__(config)
        self.config = config
        
        # memory specific
        self.mem_len = mem_len
        self.clamp_len = clamp_len
        self.same_length = same_length
        self.memory_extended_pos_embedding = MemoryExtendedPositionalEmbedding(config.hidden_size)
        # end of memory specific

        self.has_visual_segment_embedding = config.has_visual_segment_embedding
        self.embeddings = LayoutLMv2Embeddings(config)

        self.visual = LayoutLMv2VisualBackbone(config)
        self.visual_proj = nn.Linear(config.image_feature_pool_shape[-1], config.hidden_size)
        if self.has_visual_segment_embedding:
            self.visual_segment_embedding = nn.Parameter(nn.Embedding(1, config.hidden_size).weight[0])
        self.visual_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encoder = LayoutLMv2Encoder(config)
        self.pooler = LayoutLMv2Pooler(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _calc_text_embeddings(self, input_ids, bbox, position_ids, token_type_ids, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings.word_embeddings(input_ids)
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._calc_spatial_position_embeddings(bbox)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + spatial_position_embeddings + token_type_embeddings
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings

    def _calc_img_embeddings(self, image, bbox, position_ids):
        visual_embeddings = self.visual_proj(self.visual(image))
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._calc_spatial_position_embeddings(bbox)
        embeddings = visual_embeddings + position_embeddings + spatial_position_embeddings
        if self.has_visual_segment_embedding:
            embeddings += self.visual_segment_embedding
        embeddings = self.visual_LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def _calc_visual_bbox(self, image_feature_pool_shape, bbox, device, final_shape):
        visual_bbox_x = torch_int_div(
            torch.arange(
                0,
                1000 * (image_feature_pool_shape[1] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            ),
            self.config.image_feature_pool_shape[1],
        )
        visual_bbox_y = torch_int_div(
            torch.arange(
                0,
                1000 * (self.config.image_feature_pool_shape[0] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            ),
            self.config.image_feature_pool_shape[0],
        )
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(image_feature_pool_shape[0], 1),
                visual_bbox_y[:-1].repeat(image_feature_pool_shape[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(image_feature_pool_shape[0], 1),
                visual_bbox_y[1:].repeat(image_feature_pool_shape[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, bbox.size(-1))

        visual_bbox = visual_bbox.repeat(final_shape[0], 1, 1)

        return visual_bbox

    def init_mems(self, bsz):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.config.num_hidden_layers):
                empty = torch.zeros(bsz, self.mem_len, self.config.hidden_size, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, mlen, qlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        # drop the first hidden state as it isn't 
        assert len(hids) - 1 == len(mems), "len(hids) != len(mems)"

        # There are `mlen + qlen` steps that can be cached into mems
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids) - 1):
                cat = torch.cat([mems[i], hids[i + 1]], dim=1)
                new_mems.append(cat[:, beg_idx:end_idx].detach())

        return new_mems

    @add_start_docstrings_to_model_forward(LAYOUTLMV2_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @replace_return_docstrings(output_type=MemorizableLayoutLMv2ModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        image: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        mems = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Return:

        Examples:

        ```python
        >>> from transformers import LayoutLMv2Processor, LayoutLMv2Model, set_seed
        >>> from PIL import Image
        >>> import torch
        >>> from datasets import load_dataset

        >>> set_seed(88)

        >>> processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
        >>> model = LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased")


        >>> dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
        >>> image_path = dataset["test"][0]["file"]
        >>> image = Image.open(image_path).convert("RGB")

        >>> encoding = processor(image, return_tensors="pt")

        >>> outputs = model(**encoding)
        >>> last_hidden_states = outputs.last_hidden_state

        >>> last_hidden_states.shape
        torch.Size([1, 342, 768])
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        visual_shape = list(input_shape)
        visual_shape[1] = self.config.image_feature_pool_shape[0] * self.config.image_feature_pool_shape[1]
        visual_shape = torch.Size(visual_shape)
        final_shape = list(input_shape)
        final_shape[1] += visual_shape[1]
        final_shape = torch.Size(final_shape)

        visual_bbox = self._calc_visual_bbox(self.config.image_feature_pool_shape, bbox, device, final_shape)
        final_bbox = torch.cat([bbox, visual_bbox], dim=1)



        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if position_ids is None:
            seq_length = input_shape[1]
            position_ids = self.embeddings.position_ids[:, :seq_length]
            position_ids = position_ids.expand(input_shape)

        visual_position_ids = torch.arange(0, visual_shape[1], dtype=torch.long, device=device).repeat(
            input_shape[0], 1
        )
        final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)

        if bbox is None:
            bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

        text_layout_emb = self._calc_text_embeddings(
            input_ids=input_ids,
            bbox=bbox,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        visual_emb = self._calc_img_embeddings(
            image=image,
            bbox=visual_bbox,
            position_ids=visual_position_ids,
        )
        final_emb = torch.cat([text_layout_emb, visual_emb], dim=1)
        
        # memory handling
        bsz, qlen = input_shape
        if mems is None:
            mems = self.init_mems(bsz)        
        mlen = mems[0].size(1) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = text_layout_emb.new_ones((qlen, klen), dtype=torch.uint8)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            memory_attention_mask = (torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -mask_shift_len)) # -1
        else:
            memory_attention_mask = torch.triu(text_layout_emb.new_ones((qlen, klen), dtype=torch.uint8), diagonal=1 + mlen)

        # do not mask visual positions
        final_attention_mask = torch.nn.functional.pad(memory_attention_mask, (0, visual_shape[1], 0, visual_shape[1],), value=0)

        mem_pos_seq = torch.arange(final_attention_mask.size(1) - 1, -1, -1.0, device=text_layout_emb.device, dtype=text_layout_emb.dtype)
        if self.clamp_len > 0:
            mem_pos_seq.clamp_(max=self.clamp_len)
        mem_pos_emb = self.memory_extended_pos_embedding(mem_pos_seq)
        mem_pos_emb = self.dropout(mem_pos_emb)
        # end of memory handling

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            final_emb,
            mem_pos_emb,
            final_attention_mask,
            bbox=final_bbox,
            position_ids=final_position_ids,
            head_mask=head_mask,
            mems=mems,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        new_mems = self._update_mems(encoder_outputs[1], mems, mlen, qlen)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output, new_mems) + encoder_outputs[1:]

        return MemorizableLayoutLMv2ModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            mems=new_mems,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )