import torch.nn as nn

from modules.transformer import TransformerEncoder
from modules.transformer import TransformerDecoder


class Transformer(nn.Module):

    def __init__(
            self,
            num_heads,
            embedding_size,
            encoder_embedding,
            decoder_embedding,
            ffn_size,
            pad_index,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dropout=0.0,
            attention_dropout=None,
            relu_dropout=None,
            learn_positional_embeddings=False,
            embeddings_scale=False,
            num_positions=1024
    ):
        self.encoder = TransformerEncoder(
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            embedding_size=embedding_size,
            embedding=encoder_embedding,
            ffn_size=ffn_size,
            pad_index=pad_index,
            dropout=dropout,
            attention_dropout=attention_dropout,
            relu_dropout=relu_dropout,
            learn_positional_embeddings=learn_positional_embeddings,
            embeddings_scale=embeddings_scale,
            reduction=False,
            num_positions=num_positions
        )

        self.decoder = TransformerDecoder(
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            embedding_size=embedding_size,
            ffn_size=ffn_size,
            embedding=decoder_embedding,
            dropout=dropout,
            attention_dropout=attention_dropout,
            relu_dropout=relu_dropout,
            embeddings_scale=embeddings_scale,
            learn_positional_embeddings=learn_positional_embeddings,
            num_positions=num_positions,
        )

