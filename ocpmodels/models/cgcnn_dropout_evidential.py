"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
"""

Changes:
Dropout incorporated inbetween every fully collected layer in the CGCNN.
The dropout rate dropout_rate is applied to each fully connected layer in the CGCNN.
i.e. dropout_rate = 0.30 means any neuron in any fully connected layer has a 30% chance of being dropped.
This is equivalent to saying, "About 30% of neurons in any given fully connected layer will be dropped."

Additionally, deep evidential regression (DER) is impelmented for the CGCNN.
This involves three changes:
    (1) A custom 'DenseNormalGamma' output layer that computes the evidential hyperdistribution parameters for uncertainty.
    (2) Constrained softplus output, to ensure that all the hyperdistribution parameters are positive and well-defined.
    (3) A custom evidential loss function
"""


# One of the steps to implementing DER is creating a custom output layer for the model
# This custom output layer has the same weight/bias initialization of a standard dense, linear PyTorch layer
# Except, it calculates evidential distribution parameters




from ocpmodels.datasets.embeddings import KHOT_EMBEDDINGS, QMOF_KHOT_EMBEDDINGS
from ocpmodels.models.base import BaseModel
import math
import numpy as np
class DenseNormalGamma(torch.nn.Module):
    # Default output size is 4 because there are 4 parameters to the evidential distribution
    def __init__(self, inputSize, outputSize=4):
        super().__init__()
        self.inputSize, self.outputSize = inputSize, outputSize

        # Neural network weight tensor
        weights = torch.Tensor(inputSize, outputSize)

        # torch.nn.Parameter() converts a tensort to a special type of tensor that can be used as a module parameter
        self.weights = torch.nn.Parameter(weights)

        # Bias tensor
        bias = torch.Tensor(outputSize)

        # Turn the bias into a module parameter as well
        self.bias = torch.nn.Parameter(bias)

        # Initializing the weights and biases
        # Weight initialization (equivalent to standard dense, linear PyTorch layer)
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / np.sqrt(fan_in)

        # Bias initialization (equivalent to standard dense, linear PyTorch layer)
        torch.nn.init.uniform_(self.bias, -bound, bound)

    # Define a new function called evidence()
    # All this function does is perform SoftPlus on the evidential distribution parameters.
    # This has the effect of ensuring all parameters are positive at all times
    def evidence(self, x):
        return torch.nn.functional.softplus(x)

    # Now a new function needs to be defined to describe how data is forward-passed through the model
    def forward(self, x):
        # Multiply the incoming data by the neural network edge weights
        self.weightMultipliedData = torch.matmul(x, self.weights)
        # Add the neural network node biases
        self.biasesAdded = torch.add(self.weightMultipliedData, self.bias)

        # Split the output layer (which contains evidential hyperdist. parameters)
        # into individual evidential distribution parameters
        mu, logv, logalpha, logbeta = torch.split(self.biasesAdded, 1, dim=1)

        # Ensure positive values to the hyperdistribution parameters
        logv = self.evidence(logv)
        logalpha = self.evidence(logalpha) + 1
        logbeta = self.evidence(logbeta)

        # Return the overall output of the positive-ensured hyperdistr. parameters.
        # This is our forward pass result.
        return torch.cat([mu, logv, logalpha, logbeta], dim=1)


@registry.register_model('cgcnn_dropout_evidential_v2')
class CGCNN(BaseModel):
    r"""Implementation of the Crystal Graph CNN model from the
    `"Crystal Graph Convolutional Neural Networks for an Accurate
    and Interpretable Prediction of Material Properties"
    <https://arxiv.org/abs/1710.10324>`_ paper.

    Args:
        num_atoms (int): Number of atoms.
        bond_feat_dim (int): Dimension of bond features.
        num_targets (int): Number of targets to predict.
        use_pbc (bool, optional): If set to :obj:`True`, account for periodic boundary conditions.
            (default: :obj:`True`)
        regress_forces (bool, optional): If set to :obj:`True`, predict forces by differentiating
            energy with respect to positions.
            (default: :obj:`True`)
        atom_embedding_size (int, optional): Size of atom embeddings.
            (default: :obj:`64`)
        num_graph_conv_layers (int, optional): Number of graph convolutional layers.
            (default: :obj:`6`)
        fc_feat_size (int, optional): Size of fully connected layers.
            (default: :obj:`128`)
        num_fc_layers (int, optional): Number of fully connected layers.
            (default: :obj:`4`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        num_gaussians (int, optional): Number of Gaussians used for smearing.
            (default: :obj:`50.0`)
    """
    """
    Additional Args:
        use_dropout (bool): Utilize dropout or not.
            (default: False)
        dropout_rate (float16): Probably of dropping a neuron in any given fully connected layer.
            (default: 0.00)
        dropout_on_inference (bool): Whether or not to drop nodes post-training on predictions.
            (default: False)
        use_evidence (bool): Whether to utilize evidential regression or not.
            (default: False)
    """

    def __init__(
        self,
        num_atoms,
        bond_feat_dim,
        num_targets,
        use_pbc=True,
        regress_forces=True,
        atom_embedding_size=64,
        num_graph_conv_layers=6,
        fc_feat_size=128,
        num_fc_layers=4,
        otf_graph=False,
        cutoff=6.0,
        num_gaussians=50,
        embeddings='khot',
        # Use dropout at all in the model
        use_dropout=False,
        # The degree of dropout for each fully-connected layer.
        dropout_rate=0.00,
        # Use dropout during model prediction (post-training)
        dropout_on_inference=False,
        # Use evidential deep learning
        use_evidence=False,
        # Evidential regularization term hyperparameters
        lamb=0.0,
    ):
        super(CGCNN, self).__init__(num_atoms, bond_feat_dim, num_targets)
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph

        # Apply custom dropout arguments passed to __init__ onto custom CGCNN class attributes
        # Now, they are officially a part of the CGCNN class.
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.dropout_on_inference = False

        # Apply custom evidential learning arguments passed to __init__ onto custom CGCNN class attributes
        # Now, they are officially a part of the CGCNN class.
        self.use_evidence = use_evidence
        self.lamb = lamb

        # Get CGCNN atom embeddings
        if embeddings == 'khot':
            embeddings = KHOT_EMBEDDINGS
        elif embeddings == 'qmof':
            embeddings = QMOF_KHOT_EMBEDDINGS
        else:
            raise ValueError(
                'embedding mnust be either "khot" for original CGCNN K-hot elemental embeddings or "qmof" for QMOF K-hot elemental embeddings'
            )
        self.embedding = torch.zeros(100, len(embeddings[1]))
        for i in range(100):
            self.embedding[i] = torch.tensor(embeddings[i + 1])
        self.embedding_fc = nn.Linear(len(embeddings[1]), atom_embedding_size)

        self.convs = nn.ModuleList(
            [
                CGCNNConv(node_dim=atom_embedding_size,
                          edge_dim=bond_feat_dim, cutoff=cutoff,)
                for _ in range(num_graph_conv_layers)
            ]
        )

        self.conv_to_fc = nn.Sequential(
            nn.Linear(atom_embedding_size, fc_feat_size), nn.Softplus()
        )

        # If dropout is specified via use_dropout = True
        if self.use_dropout:
            if num_fc_layers > 1:
                layers = []
                for _ in range(num_fc_layers - 1):
                    layers.append(nn.Linear(fc_feat_size, fc_feat_size))
                    # Add dropout here with dropout rate stored in dropout_rate
                    layers.append(torch.nn.Dropout(
                        p=self.dropout_rate, inplace=False))
                    layers.append(nn.Softplus())
                self.fcs = nn.Sequential(*layers)

        # If dropout is specified via use_dropout = False
        else:
            if num_fc_layers > 1:
                layers = []
                for _ in range(num_fc_layers - 1):
                    layers.append(nn.Linear(fc_feat_size, fc_feat_size))
                    layers.append(nn.Softplus())
                self.fcs = nn.Sequential(*layers)

        # self.fc_out refers to the final fully-connected layer that connects to the target variables
        # If dropout is specified via use_dropout = True
        if self.use_dropout:
            fc_out_layers = []
            if self.use_evidence:
                fc_out_layers.append(DenseNormalGamma(inputSize=fc_feat_size))
            else:
                fc_out_layers.append(nn.Linear(fc_feat_size, self.num_targets))
            fc_out_layers.append(torch.nn.Dropout(
                p=dropout_rate, inplace=False))
            self.fc_out = nn.Sequential(*fc_out_layers)

        # If dropout is specified via use_dropout = False
        else:
            if self.use_evidence:
                self.fc_out = DenseNormalGamma(inputSize=fc_feat_size)
            else:
                self.fc_out = nn.Linear(fc_feat_size, self.num_targets)

        self.cutoff = cutoff
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        # Get node features
        if self.embedding.device != data.atomic_numbers.device:
            self.embedding = self.embedding.to(data.atomic_numbers.device)
        data.x = self.embedding[data.atomic_numbers.long() - 1]

        pos = data.pos

        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.cutoff, 50)
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        if self.use_pbc:
            out = get_pbc_distances(
                pos, data.edge_index, data.cell, data.cell_offsets, data.neighbors,
            )

            data.edge_index = out['edge_index']
            distances = out['distances']
        else:
            data.edge_index = radius_graph(
                data.pos, r=self.cutoff, batch=data.batch)
            row, col = data.edge_index
            distances = (pos[row] - pos[col]).norm(dim=-1)

        data.edge_attr = self.distance_expansion(distances)
        # Forward pass through the network
        mol_feats = self._convolve(data)
        mol_feats = self.conv_to_fc(mol_feats)
        if hasattr(self, 'fcs'):
            mol_feats = self.fcs(mol_feats)

        energy = self.fc_out(mol_feats)
        return energy

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy = self._forward(data)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy, data.pos, grad_outputs=torch.ones_like(energy), create_graph=True,
                )[0]
            )
            return energy, forces
        else:
            return energy

    def _convolve(self, data):
        """
        Returns the output of the convolution layers before they are passed
        into the dense layers.
        """
        node_feats = self.embedding_fc(data.x)
        for f in self.convs:
            node_feats = f(node_feats, data.edge_index, data.edge_attr)
        mol_feats = global_mean_pool(node_feats, data.batch)
        return mol_feats


class CGCNNConv(MessagePassing):
    """Implements the message passing layer from
    `"Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties"
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`.
    """

    def __init__(self, node_dim, edge_dim, cutoff=6.0, **kwargs):
        super(CGCNNConv, self).__init__(aggr='add')
        self.node_feat_size = node_dim
        self.edge_feat_size = edge_dim
        self.cutoff = cutoff

        self.lin1 = nn.Linear(
            2 * self.node_feat_size + self.edge_feat_size, 2 * self.node_feat_size,
        )
        self.bn1 = nn.BatchNorm1d(2 * self.node_feat_size)
        self.ln1 = nn.LayerNorm(self.node_feat_size)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)

        self.lin1.bias.data.fill_(0)

        self.bn1.reset_parameters()
        self.ln1.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """
        Arguments:
            x has shape [num_nodes, node_feat_size]
            edge_index has shape [2, num_edges]
            edge_attr is [num_edges, edge_feat_size]
        """
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0)))
        out = nn.Softplus()(self.ln1(out) + x)
        return out

    def message(self, x_i, x_j, edge_attr):
        """
        Arguments:
            x_i has shape [num_edges, node_feat_size]
            x_j has shape [num_edges, node_feat_size]
            edge_attr has shape [num_edges, edge_feat_size]

        Returns:
            tensor of shape [num_edges, node_feat_size]
        """
        z = self.lin1(torch.cat([x_i, x_j, edge_attr], dim=1))
        z = self.bn1(z)
        z1, z2 = z.chunk(2, dim=1)
        z1 = nn.Sigmoid()(z1)
        z2 = nn.Softplus()(z2)
        return z1 * z2
