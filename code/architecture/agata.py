import math
from typing import Dict, Mapping, Optional, Tuple, Union, cast

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, Module, Sequential
from dataclasses import dataclass
from typing import List, NamedTuple, Sequence

class FCHeadOutput(NamedTuple):
    logits: Tensor
    activations: Tensor

@dataclass
class LinearLayerSpec:
    dim: int
    activation: Module


@dataclass
class FCHeadConfig:
    in_channels: int
    layer_specs: Sequence[LinearLayerSpec]
class FCHead(Module):
    def __init__(self, in_channels: int, layer_specs: Sequence[LinearLayerSpec]) -> None:
        """Creates fully connected layers to be used as a model head.

        Args:
            in_channels: Number of channels of the feature vector provided by the backbone
            layer_specs: Sequence of linear layer specifications
        """
        super().__init__()
        self.in_channels = in_channels
        self.layer_specs = layer_specs

        ops: List[Module] = []

        for layer_spec in self.layer_specs[:-1]:
            ops.append(Linear(in_channels, layer_spec.dim))
            ops.append(layer_spec.activation)

            in_channels = layer_spec.dim

        ops.append(Linear(in_channels, self.layer_specs[-1].dim))

        self.activation = self.layer_specs[-1].activation
        self.fc = Sequential(*ops)

    def __call__(self, __x: Tensor) -> FCHeadOutput:  # type: ignore
        out: FCHeadOutput = super().__call__(__x)
        return out

    def forward(self, x: Tensor) -> FCHeadOutput:  # type: ignore
        """Runs a forward pass."""
        logits = self.fc(x)
        activations = self.activation(logits)
        return FCHeadOutput(logits=logits, activations=activations)
class DotProductAttentionWithLearnedQueries(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_queries: int = 1,
        scaled: bool = False,
        absolute: bool = False,
        reduce: bool = False,
        padding_indicator: float = 1,
        return_normalized: bool = True,
    ) -> None:
        """Performs dot product attention given a set of keys and values with a set of learned
        queries.

        Args:
            in_features: Number of features expected in query embeddings.
            n_queries: Number of learned keys to use. Defaults to 1.
            scaled: Performs scaled dot product attention. Defaults to False.
            absolute: Applies attention to absolute value of value embeddings.
                Defaults to False.
            reduce: Sums the attended values to output a single embeddings for each
                instance in the batch. Defaults to False.
            padding_indicator: Value that indicates padding positions in mask.
                Defaults to 1.
            return_normalized: Returns the attention scores with softmax applied.
        """
        super().__init__()
        # If we think about the generalization of attention being a dot product between a set of
        # queries and a set of keys where the resulting product is applied to a set of values then
        # here we will have learned keys in the form of a weight matrix
        self.queries = nn.Linear(in_features, n_queries, bias=False)

        self.n_queries = n_queries
        self.scaled = scaled
        self.absolute = absolute
        self.reduce = reduce
        self.padding_indicator = padding_indicator
        self.return_normalized = return_normalized
        self.pad_value = float('-inf')

    def _format_padding_mask(
        self, padding_mask: Tensor, expected_shape: Tuple[int, int, int]
    ) -> Tensor:
        """Validates and formats padding masks. Masks are expected to be in the shape of
        (batch, sequence) or (batch, sequence, 1) and must have the same size in the batch and
        sequence dimensions as the query/value tensors. This function validates the format and
        shapes are as expected and repeats masks in a third dimension to enable proper use for
        masking out multiple attention keys.

        Args:
            padding_mask: (B, S) or (B, S, 1) mask to ensure a padding position does not contribute
                to attention.
            expected_shape: Shape to validate and format to.

        Raises:
            RuntimeError: If the size at the batch and sequence dimensions is incorrect.
            RuntimeError: If the format is not what is expected.

        Returns:
            Tensor: Formatted padding mask.
        """
        if padding_mask.shape[:2] != expected_shape[:2]:
            raise RuntimeError(
                f'Expected padding_mask to have shape of {expected_shape[:2]} at first two dimensions, but got shape of {padding_mask.shape[:2]}'
            )
        if len(padding_mask.shape) == 2:
            padding_mask = padding_mask.unsqueeze(-1)

        if len(padding_mask.shape) == 3:
            if padding_mask.shape[2] == 1:
                padding_mask = padding_mask.expand(-1, -1, expected_shape[2])
            elif padding_mask.shape[2] != expected_shape[2]:
                raise RuntimeError(
                    f'Expected padding_mask to have shape of {expected_shape[:2]} or {(expected_shape[0], expected_shape[1], 1)}, but got shape of {padding_mask.shape}'
                )

        return padding_mask

    def forward(  # type: ignore
        self, key: Tensor, value: Tensor, padding_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass to perform dot product attention with learned set of keys.

        Args:
            key: (B, S, F) where B is the batch dimension, S is the sequence length, and F is the
                embedding dimension.
            value: (B, S, F) where B is the batch dimension, S is the sequence length, and F is the
                embedding dimension.
            padding_mask: (B, S) or (B, S, 1) mask to ensure a padding position does not contribute
                to attention. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: Attended values of shape (B, F) if reduce is true and (B, Q, F)
                otherwise. Attention scores of shape (B, Sk, Q).
        """
        if key.size(1) != value.size(1):
            raise RuntimeError('Expected key and value to have same sequence length (dim 1).')
        # (B, Sk, F) x (B, F, Q) -> (B, Sk, Q)
        attn = self.queries(key)

        if padding_mask is not None:
            padding_mask = self._format_padding_mask(
                padding_mask, expected_shape=(key.size(0), key.size(1), attn.size(2))
            )
            # Set attention scores at padding locations to -inf so after softmax they are
            # zero-valued
            attn[padding_mask == self.padding_indicator] = self.pad_value

        if self.scaled:
            d_k = key.size(-1)
            attn /= math.sqrt(d_k)

        normalized_attn = F.softmax(attn, dim=1)

        if self.absolute:
            value = torch.abs(value)

        # (B, Q, Sk) x (B, Sv, F) -> (B, Q, F), where Sv == Sk
        output = torch.bmm(normalized_attn.transpose(-2, -1), value)

        if self.reduce:
            # (B, Q, F) -> (B, F)
            output = output.sum(1)  # should this be mean?

        if self.return_normalized:
            attn = normalized_attn

        return output, attn


class Agata(nn.Module):
    """Defines a two layer, feed forward, 'Aggregator with Attention' model architecture"""

    def __init__(
        self,

        in_features: int = 768,
        layer1_out_features: int = 512,
        layer2_out_features: int = 512,
        label_name_fclayer_head_config: Mapping[str, FCHeadConfig] = {
            'head': FCHeadConfig(
                512, [LinearLayerSpec(dim=2048, activation=nn.ReLU()), LinearLayerSpec(dim=1, activation=nn.ReLU())]
            )
        },
        activation: nn.Module = nn.GELU(),
        scaled_attention: bool = False,
        absolute_attention: bool = False,
        n_attention_queries: int = 16,
        padding_indicator: int = 1,
    ) -> None:
        """Initializes an Agata model.

        Args:
            in_features: Dimension of an input embedding.
            layer1_out_features: Dimension of first linear layer.
            layer2_out_features: Dimension of second linear layer
            activation: Activation to be applied between both linear layers.
            label_name_fclayer_head_config: A mapping specifying the type of head for each label.
            scaled_attention: Normalizes attention values by sqrt(`layer2_out_features`).
            absolute_attention: Takes absolute value of attention.
            n_attention_queries: Number of attention queries.
            padding_indicator: Value that indicates padding positions in mask.
        """
        super().__init__()
        self.linear1 = nn.Linear(in_features, layer1_out_features)
        self.linear2 = nn.Linear(layer1_out_features, layer2_out_features)

        self.attention = DotProductAttentionWithLearnedQueries(
            in_features=layer1_out_features,
            n_queries=n_attention_queries,
            scaled=scaled_attention,
            absolute=absolute_attention,
            reduce=True,
            padding_indicator=padding_indicator,
            return_normalized=False,  # want non-softmaxed attn scores for KL loss calc
        )

        self.activation = activation

        heads = {
            name: FCHead(cfg.in_channels, cfg.layer_specs)
            for name, cfg in label_name_fclayer_head_config.items()
        }
        self.heads = cast(Mapping[str, FCHead], nn.ModuleDict(heads))

    def forward(  # type: ignore
        self, x: Tensor
    ) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        """Runs a forward pass.

        Args:
            x: Batch of embeddings. Expected to be in the shape of (batch, sequence, features
            a.k.a. (b, s, f), because torch.nn.Linear operation requires features to be in the last
            dimension.

            padding_masks: Mask that indicates which indices in a sequence are padding and which are
                valid. Expected to be in the shape of (batch, sequence).

        Returns:
            A dictionary containing output tensors for the aggregated embedding, output
            head logits and activations, and attention scores.
        """
        # x.shape -> (batch, sequence, features) a.k.a (B, S, F)

        x = x.unsqueeze(0)
        x_1, x_2 = self.forward_features(x)

        # x_3.shape -> (B, F), attn.shape -> (B, S, n_attention_queries)
        # x_3, attn = self.attention(key=x_1, value=x_2, padding_mask=padding_masks)
        x_3, attn = self.attention(key=x_1, value=x_2)
        heads_logits, heads_activations = self._forward_output_heads(x_3)

        # return {
        #     'backbone_embedding': x_3,
        #     'heads_logits': heads_logits,
        #     'heads_activations': heads_activations,
        #     'attn_scores': attn,
        # }
        return heads_logits['head']



    def forward_features(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Runs the layers in the network that come before attention.

        .. note::
            this API is named as such in order to achieve consistency with timm architectures,
            which all posess a `forward_features` method.
        """
        x_1 = self.linear1(x)
        x_1 = self.activation(x_1)

        x_2 = self.linear2(x_1)
        x_2 = self.activation(x_2)
        return x_1, x_2

    def _forward_output_heads(
        self, backbone_embedding: Tensor
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Forward pass for output heads.

        Args:
            backbone_embedding: Shared embedding to be distributed to output heads.

        Raises:
            TypeError: Unsupported type is output by a head.

        Returns:
            Dictionaries of label names mapped to head logits/activations.
        """
        heads_logits: Dict[str, Tensor] = {}
        heads_activations: Dict[str, Tensor] = {}
        for name, head in self.heads.items():
            logits, activations = head(backbone_embedding)
            heads_logits[name] = logits
            heads_activations[name] = activations

        return heads_logits, heads_activations