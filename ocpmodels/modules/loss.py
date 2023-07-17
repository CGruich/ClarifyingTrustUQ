import torch
from torch import nn
import numpy as np

from ocpmodels.common import distutils

# For multiple datatype support
from typing import Union


class L2MAELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        assert reduction in ['mean', 'sum']

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        dists = torch.norm(input - target, p=2, dim=-1)
        if self.reduction == 'mean':
            return torch.mean(dists)
        elif self.reduction == 'sum':
            return torch.sum(dists)


class AtomwiseL2Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        assert reduction in ['mean', 'sum']

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor,
    ):
        assert natoms.shape[0] == input.shape[0] == target.shape[0]
        assert len(natoms.shape) == 1  # (nAtoms, )

        dists = torch.norm(input - target, p=2, dim=-1)
        loss = natoms * dists

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


class EvidentialLoss(nn.Module):
    r"""Creates a criterion that measures the evidential negative log likelihood (eNLL) between each element in
    the input :math:`x` and target :math:`y`.

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.


    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        lamb (float, optional): Specifies how much to weight the regularization term (nll_reg) in calculating EvidentialLoss. e.g. 0.20 is 20% weight on regularization, 1.0 is 100%, etc.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then
          :math:`(*)`, same shape as the input.
    """

    def __init__(self, reduction: str = 'sum', lamb: float = 0.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.lamb = lamb
        assert reduction in ['mean', 'sum']

    # Custom function for calculating the negative log likelihood of the evidential loss function
    def NIG_NLL(self, y, gamma, v, alpha, beta, reduction='sum'):

        twoBlambda = 2 * beta * (1 + v)

        nll = (
            0.5 * torch.log(np.pi / v)
            - alpha * torch.log(twoBlambda)
            + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        if self.reduction == 'mean':
            return torch.mean(nll)
        elif self.reduction == 'sum':
            return torch.sum(nll)

    # Custom function for calculating the regularizer term of the evidential loss function
    def NIG_Reg(self, y, gamma, v, alpha, beta, reduction='sum'):

        # Get the predictive error
        error = torch.abs(y - gamma)

        evidence = 2 * v + (alpha)
        reg = error * evidence

        if self.reduction == 'mean':
            return torch.mean(reg)
        elif self.reduction == 'sum':
            return torch.sum(reg)

    # One of the steps to implementing DER is to define a custom loss function for the neural network
    # This is done here.
    def EvidentialRegression(self, pred, target, lamb=0.0, reduction='sum'):

        # Get the evidential distribution parameters from the model output
        # Calculate the negative log likelihood term of the evidential loss function
        loss_nll = self.NIG_NLL(
            target, pred['energy'], pred['v'], pred['alpha'], pred['beta'], reduction
        )
        # Calculate the regularizer term (see original paper)
        loss_reg = self.NIG_Reg(
            target, pred['energy'], pred['v'], pred['alpha'], pred['beta'], reduction
        )

        return loss_nll + (lamb * loss_reg)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lamb: float = 0.0,
        reduction: str = 'sum',
    ) -> torch.Tensor:

        nll_error = self.EvidentialRegression(pred, target, self.lamb, self.reduction)
        return nll_error


class DDPLoss(nn.Module):
    def __init__(self, loss_fn, reduction='mean'):
        super().__init__()
        self.loss_fn = loss_fn
        self.loss_fn.reduction = 'sum'
        self.reduction = reduction
        assert reduction in ['mean', 'sum']

    def forward(
        self,
        # torch.Tensor for non-evidential regression predictions
        # Dictionary (i.e., dict) for evidential regression predictions
        pred: Union[torch.Tensor, dict],
        target: torch.Tensor = None,
        natoms: torch.Tensor = None,
        batch_size: int = None,
    ):
        if natoms is None:
            loss = self.loss_fn(pred, target)
        else:  # atom-wise loss
            loss = self.loss_fn(pred, target, natoms)
        if self.reduction == 'mean':
            # If we are performing evidential regression,
            if type(pred) == dict:
                firstPredKey = next(iter(pred))
                num_samples = (
                    batch_size if batch_size is not None else pred[firstPredKey].shape[0]
                )

            else:
                num_samples = batch_size if batch_size is not None else pred.shape[0]

            # If using deep evidential regression,

            if type(pred) == dict:
                num_samples = distutils.all_reduce(
                    num_samples, device=pred[firstPredKey].device
                )
            else:
                num_samples = distutils.all_reduce(num_samples, device=pred.device)
            # Multiply by world size since gradients are averaged
            # across DDP replicas
            return loss * distutils.get_world_size() / num_samples
        else:
            return loss
