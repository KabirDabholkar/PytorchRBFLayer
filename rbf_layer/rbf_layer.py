#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Alessio Russo [alessior@kth.se]. All rights reserved.
#
# This file is part of PytorchRBFLayer.
#
# PytorchRBFLayer is free software: you can redistribute it and/or modify
# it under the terms of the MIT License. You should have received a copy of
# the MIT License along with PytorchRBFLayer.
# If not, see <https://opensource.org/licenses/MIT>.
#

import torch
import torch.nn as nn

from typing import Callable


class RBFLayer(nn.Module):
    """
    Defines a Radial Basis Function Layer

    An RBF is defined by 5 elements:
        1. A radial kernel phi
        2. A positive shape parameter epsilon
        3. The number of kernels N, and their relative
           centers c_i, i=1, ..., N
        4. A norm ||.||
        5. A set of weights w_i, i=1, ..., N

    The output of an RBF is given by
    y(x) = sum_{i=1}^N a_i * phi(eps_i * ||x - c_i||)

    For more information check [1,2]

    [1] https://en.wikipedia.org/wiki/Radial_basis_function
    [2] https://en.wikipedia.org/wiki/Radial_basis_function_network

    Parameters
    ----------
        in_features_dim: int
            Dimensionality of the input features
        num_kernels: int
            Number of kernels to use
        out_features_dim: int
            Dimensionality of the output features
        radial_function: Callable[[torch.Tensor], torch.Tensor]
            A radial basis function that returns a tensor of real values
            given a tensor of real values
        norm_function: Callable[[torch.Tensor], torch.Tensor]
            Normalization function applied on the features
        normalization: bool, optional
            if True applies the normalization trick to the rbf layer
        initial_shape_parameter: torch.Tensor, optional
            Sets the shape parameter to the desired value.
        initial_centers_parameter: torch.Tensor, optional
            Sets the centers to the desired value.
        initial_weights_parameters: torch.Tensor, optional
            Sets the weights parameter to the desired value.
        constant_shape_parameter: bool, optional
            Sets the shapes parameters to a non-learnable constant.
            initial_shape_parameter must be different than None if
            constant_shape_parameter is True
        constant_centers_parameter: bool, optional
            Sets the centers to a non-learnable constant.
            initial_centers_parameter must be different than None if
            constant_centers_parameter is True
        constant_weights_parameters: bool, optional
            Sets the weights to a non-learnable constant.
            initial_weights_parameters must be different than None if
            constant_weights_parameters is True
    """

    def __init__(self,
                 in_features_dim: int,
                 num_kernels: int,
                 out_features_dim: int,
                 radial_function: Callable[[torch.Tensor], torch.Tensor],
                 norm_function: Callable[[torch.Tensor], torch.Tensor],
                 normalization: bool = True,
                 initial_shape_parameter: torch.Tensor = None,
                 initial_centers_parameter: torch.Tensor = None,
                 initial_weights_parameters: torch.Tensor = None,
                 constant_shape_parameter: bool = False,
                 constant_centers_parameter: bool = False,
                 constant_weights_parameters: bool = False):
        super(RBFLayer, self).__init__()

        self.in_features_dim = in_features_dim
        self.num_kernels = num_kernels
        self.out_features_dim = out_features_dim
        self.radial_function = radial_function
        self.norm_function = norm_function
        self.normalization = normalization

        self.initial_shape_parameter = initial_shape_parameter
        self.constant_shape_parameter = constant_shape_parameter

        self.initial_centers_parameter = initial_centers_parameter
        self.constant_centers_parameter = constant_centers_parameter

        self.initial_weights_parameters = initial_weights_parameters
        self.constant_weights_parameters = constant_weights_parameters

        assert radial_function is not None  \
            and norm_function is not None
        assert normalization is False or normalization is True

        self._make_parameters()

    def _make_parameters(self) -> None:
        # Initialize linear combination weights
        if self.constant_weights_parameters:
            self.weights = nn.Parameter(
                self.initial_weights_parameters, requires_grad=False)
        else:
            self.weights = nn.Parameter(
                torch.zeros(
                    self.out_features_dim,
                    self.num_kernels,
                    dtype=torch.float32))

        # Initialize kernels' centers
        if self.constant_centers_parameter:
            self.kernels_centers = nn.Parameter(
                self.initial_centers_parameter, requires_grad=False)
        else:
            self.kernels_centers = nn.Parameter(
                torch.zeros(
                    self.num_kernels,
                    self.in_features_dim,
                    dtype=torch.float32))

        # Initialize shape parameter
        if self.constant_shape_parameter:
            self.log_shapes = nn.Parameter(
                self.initial_shape_parameter, requires_grad=False)
        else:
            self.log_shapes = nn.Parameter(
                torch.zeros(self.num_kernels, dtype=torch.float32))

        self.reset()

    def reset(self,
              upper_bound_kernels: float = 1.0,
              std_shapes: float = 0.1,
              gain_weights: float = 1.0) -> None:
        """
        Resets all the parameters.

        Parameters
        ----------
            upper_bound_kernels: float, optional
                Randomly samples the centers of the kernels from a uniform
                distribution U(-x, x) where x = upper_bound_kernels
            std_shapes: float, optional
                Randomly samples the log-shape parameters from a normal
                distribution with mean 0 and std std_shapes
            gain_weights: float, optional
                Randomly samples the weights used to linearly combine the
                output of the kernels from a xavier_uniform with gain
                equal to gain_weights
        """
        if self.initial_centers_parameter is None:
            nn.init.uniform_(
                self.kernels_centers,
                a=-upper_bound_kernels,
                b=upper_bound_kernels)

        if self.initial_shape_parameter is None:
            nn.init.normal_(self.log_shapes, mean=0.0, std=std_shapes)

        if self.initial_weights_parameters is None:
            nn.init.xavier_uniform_(self.weights, gain=gain_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Computes the ouput of the RBF layer given an input vector

        Parameters
        ----------
            input: torch.Tensor
                Input tensor of size B x Fin, where B is the batch size,
                and Fin is the feature space dimensionality of the input

        Returns
        ----------
            out: torch.Tensor
                Output tensor of size B x Fout, where B is the batch
                size of the input, and Fout is the output feature space
                dimensionality
        """

        # Input has size B x Fin
        batch_size = input.size(0)

        # Compute difference from centers
        # c has size B x num_kernels x Fin
        c = self.kernels_centers.expand(batch_size, self.num_kernels,
                                        self.in_features_dim)

        diff = input.view(batch_size, 1, self.in_features_dim) - c

        # Apply norm function; c has size B x num_kernels
        r = self.norm_function(diff)

        # Apply parameter, eps_r has size B x num_kernels
        eps_r = self.log_shapes.exp().expand(batch_size, self.num_kernels) * r

        # Apply radial basis function; rbf has size B x num_kernels
        rbfs = self.radial_function(eps_r)

        # Apply normalization
        # (check https://en.wikipedia.org/wiki/Radial_basis_function_network)
        if self.normalization:
            # 1e-9 prevents division by 0
            rbfs = rbfs / (1e-9 + rbfs.sum(dim=-1)).unsqueeze(-1)

        # Take linear combination
        out = self.weights.expand(batch_size, self.out_features_dim,
                                  self.num_kernels) * rbfs.view(
                                      batch_size, 1, self.num_kernels)

        return out.sum(dim=-1)

    @property
    def get_kernels_centers(self):
        """ Returns the centers of the kernels """
        return self.kernels_centers.detach()

    @property
    def get_weights(self):
        """ Returns the linear combination weights """
        return self.weights.detach()

    @property
    def get_shapes(self):
        """ Returns the shape parameters """
        return self.log_shapes.detach().exp()



class AnisotropicRBFLayer(nn.Module):
    """
    Anisotropic RBF Layer.

    Each RBF kernel is defined by:
      - a center c_i,
      - a scaling matrix M_i = L_i L_i^T (which is semipositive definite),
      - and a weight vector for combining the kernels.

    The layer computes for an input x:

      y(x) = sum_{i=1}^N a_i * φ( (x - c_i)^T M_i (x - c_i) )
           = sum_{i=1}^N a_i * φ( ||L_i (x-c_i)||^2 )

    Parameters:
      in_features_dim: int
          Dimensionality of the input.
      num_kernels: int
          Number of RBF kernels.
      out_features_dim: int
          Dimensionality of the output.
      radial_function: Callable[[torch.Tensor], torch.Tensor]
          The kernel function φ(·) that accepts a tensor (e.g. squared distance)
          and returns a tensor of the same shape.
      normalization: bool (default True)
          If True, normalizes the RBF outputs (e.g. so that they sum to one).
      initial_centers: Optional[torch.Tensor]
          (num_kernels, in_features_dim) tensor to initialize kernel centers.
      initial_weights: Optional[torch.Tensor]
          (out_features_dim, num_kernels) tensor to initialize linear combination weights.
      initial_L: Optional[torch.Tensor]
          (num_kernels, in_features_dim, in_features_dim) tensor to initialize the
          scaling matrices factors L.
      constant_centers, constant_weights, constant_L: bool (default False)
          If True, the corresponding parameters are not learnable.
    """

    def __init__(self,
                 in_features_dim: int,
                 num_kernels: int,
                 out_features_dim: int,
                 radial_function: Callable[[torch.Tensor], torch.Tensor],
                 normalization: bool = True,
                 initial_centers: torch.Tensor = None,
                 initial_weights: torch.Tensor = None,
                 initial_L: torch.Tensor = None,
                 constant_centers: bool = False,
                 constant_weights: bool = False,
                 constant_L: bool = False):
        super(AnisotropicRBFLayer, self).__init__()
        self.in_features_dim = in_features_dim
        self.num_kernels = num_kernels
        self.out_features_dim = out_features_dim
        self.radial_function = radial_function
        self.normalization = normalization

        # Initialize centers
        if constant_centers:
            assert initial_centers is not None, "initial_centers must be provided if constant_centers is True"
            self.centers = nn.Parameter(initial_centers, requires_grad=False)
        else:
            self.centers = nn.Parameter(torch.zeros(num_kernels, in_features_dim))

        # Initialize linear combination weights
        if constant_weights:
            assert initial_weights is not None, "initial_weights must be provided if constant_weights is True"
            self.weights = nn.Parameter(initial_weights, requires_grad=False)
        else:
            self.weights = nn.Parameter(torch.zeros(out_features_dim, num_kernels))

        # Initialize the scaling factors L for each kernel, such that M = L L^T is semipositive definite.
        if constant_L:
            assert initial_L is not None, "initial_L must be provided if constant_L is True"
            self.L = nn.Parameter(initial_L, requires_grad=False)
        else:
            # A common initialization is to start with the identity matrix for each kernel.
            self.L = nn.Parameter(torch.eye(in_features_dim).unsqueeze(0).repeat(num_kernels, 1, 1))

        self.reset_parameters()

    def reset_parameters(self, center_bound: float = 1.0, gain_weights: float = 1.0):
        # Initialize centers uniformly in [-center_bound, center_bound]
        if self.centers.requires_grad:
            nn.init.uniform_(self.centers, -center_bound, center_bound)
        # Initialize weights using Xavier initialization.
        if self.weights.requires_grad:
            nn.init.xavier_uniform_(self.weights, gain=gain_weights)
        # For L, we initialize to the identity (you might add noise here if desired).
        if self.L.requires_grad:
            self.L.data.copy_(torch.eye(self.in_features_dim).unsqueeze(0).repeat(self.num_kernels, 1, 1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters:
          input: Tensor of shape (B, in_features_dim)

        Returns:
          out: Tensor of shape (B, out_features_dim)
        """
        batch_size = input.size(0)
        # Compute differences: shape (B, num_kernels, in_features_dim)
        diff = input.unsqueeze(1) - self.centers.unsqueeze(0)

        # For each kernel i and each input, compute L_i (x - c_i).
        # Using einsum: L has shape (num_kernels, in_features_dim, in_features_dim)
        # and diff has shape (B, num_kernels, in_features_dim).
        # The result z has shape (B, num_kernels, in_features_dim).
        z = torch.einsum('n d e, b n e -> b n d', self.L, diff)

        # Compute the squared Mahalanobis distance for each kernel:
        # (x-c)^T M (x-c) = ||L (x-c)||^2.
        sq_dist = (z ** 2).sum(dim=-1)  # shape: (B, num_kernels)

        # Apply the radial basis function φ.
        rbfs = self.radial_function(sq_dist)  # shape: (B, num_kernels)

        # Optional normalization (e.g., to sum to 1 for each input sample).
        if self.normalization:
            rbfs = rbfs / (rbfs.sum(dim=-1, keepdim=True) + 1e-9)

        # Compute the final output as a weighted sum of the RBF responses.
        # weights has shape (out_features_dim, num_kernels) and rbfs has shape (B, num_kernels).
        out = torch.einsum('b n, o n -> b o', rbfs, self.weights)
        return out


if __name__ == '__main__':
    def l_norm(x, p=2):
        return torch.norm(x, p=p, dim=-1)


    # Gaussian RBF
    def rbf_gaussian(x):
        return (-x.pow(2)).exp()


    model = AnisotropicRBFLayer(
        in_features_dim=3,
        num_kernels=3,
        out_features_dim=1,
        radial_function=rbf_gaussian,
    )
    x = torch.randn(4, 3)
    y = model(x)
    print(y)
    
