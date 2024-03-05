"""Loss functions."""


import torch
import torch.nn.functional as F

def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor, torch.Tensor
      Target value.
    y_pred: np.array, tf.Tensor, torch.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor, torch.Tensor
      The huber loss.
    """
    return F.smooth_l1_loss(y_true, y_pred, reduction='none')


def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor, torch.Tensor
      Target value.
    y_pred: np.array, tf.Tensor, torch.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor, torch.Tensor
      The mean huber loss.
    """
    return torch.mean(huber_loss(y_true, y_pred))
