#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import workspace as ws

def initialize_latent_vectors(num_cases, latent_size = 128, code_bound = 1.0, code_init_stddev = 1.0):
    lat_vecs = torch.nn.Embedding(num_cases, latent_size, max_norm = code_bound)
    nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        code_init_stddev / math.sqrt(latent_size),
    )
    return lat_vecs

class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dim_out,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        latent_dropout=False,
        L = 10
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        self.L = L

        if self.L is not None:
            dims = [latent_size + self.L * 6] + dims + [dim_out]

            # Precalculate
            L_mult = torch.pow( 2, torch.arange( self.L ) ) * math.pi
            L_mult = L_mult.reshape( 1, 1, -1 )
            self.register_buffer( 'L_mult', L_mult )

        else:
            dims = [latent_size + 3] + dims + [dim_out+1]

        self.num_layers = len(dims)

        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= (3 if self.L is None else 6 * self.L)

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout

    def mapping_inputs(self,xyz):
        if self.L is not None:
            # Augmentations [sin,cos,...]
            xyz = (self.L_mult * xyz.unsqueeze( -1 ))
            xyz = torch.cat( [torch.sin( xyz ), torch.cos( xyz )], 1 )
            xyz = xyz.flatten( 1 )
            return xyz
        else:
            return xyz

    # input: N x (L+3)
    def forward(self, input):

        xyz = input[:, -3:]
        xyz = self.mapping_inputs(xyz)
        latent_vecs = input[:, :-3]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)

        x = torch.cat([latent_vecs, xyz], 1)

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, latent_vecs, xyz], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        return x

def load_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]