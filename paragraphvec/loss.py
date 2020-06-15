import torch
import torch.nn as nn
import numpy as np


class NegativeSampling(nn.Module):
    """Negative sampling loss as proposed by T. Mikolov et al. in Distributed
    Representations of Words and Phrases and their Compositionality.
    """
    def __init__(self):
        super(NegativeSampling, self).__init__()
        self._log_sigmoid = nn.LogSigmoid()

    def forward(self, scores):
        """Computes the value of the loss function.

        Parameters
        ----------
        scores: autograd.Variable of size (batch_size, num_noise_words + 1)
            Sparse unnormalized log probabilities. The first element in each
            row is the ground truth score (i.e. the target), other elements
            are scores of samples from the noise distribution.
        """
        k = scores.size()[1] - 1
        return -torch.sum(
            self._log_sigmoid(scores[:, 0])
            + torch.sum(self._log_sigmoid(-scores[:, 1:]), dim=1) / k
        ) / scores.size()[0]


class NegativeSamplingWithSpline(nn.Module):
    """Negative sampling loss as proposed by T. Mikolov et al. in Distributed
    Representations of Words and Phrases and their Compositionality.

    Adapted to incorporate constraints on spline derivatives
    """
    def __init__(self):
        super(NegativeSamplingWithSpline, self).__init__()
        self._log_sigmoid = nn.LogSigmoid()

    def forward(self, scores, D, knot_locs):
        """Computes the value of the loss function.

        Parameters
        ----------
        scores: autograd.Variable of size (batch_size, num_noise_words + 1)
            Sparse unnormalized log probabilities. The first element in each
            row is the ground truth score (i.e. the target), other elements
            are scores of samples from the noise distribution.
        """
        k = scores.size()[1] - 1
        loss = -torch.sum(
            self._log_sigmoid(scores[:, 0])
            + torch.sum(self._log_sigmoid(-scores[:, 1:]), dim=1) / k
        )

        for k in knot_locs:

            # TODO: don't hard code in the min knot value as 0!!!  For now it's ok, but really should fix this ...
            if k != 0:
                loss = loss + torch.log(torch.norm(D[k, :, :] - D[k - 1, :, :]))

        return loss / scores.size()[0]
