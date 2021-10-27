import numpy as np
from numpy import ndarray, zeros
from typing import Sequence, Optional

from datatypes.measurements import GnssMeasurement
from datatypes.eskf_states import NominalState, ErrorStateGauss
from datatypes.multivargaussian import MultiVarGaussStamped

import solution


def get_NIS(z_gnss: GnssMeasurement,
            z_gnss_pred_gauss: MultiVarGaussStamped,
            marginal_idxs: Optional[Sequence[int]] = None
            ) -> float:
    """Calculate NIS

    Args:
        z_gnss (GnssMeasurement): gnss measurement
        z_gnss_pred_gauss (MultiVarGaussStamped): predicted gnss measurement
        marginal_idxs (Optional[Sequence[int]]): Sequence of marginal indexes.
            For example used for calculating NIS in only xy direction.  

    Returns:
        NIS (float): NIS value
    """

    pos = z_gnss.pos

    if marginal_idxs == None:
        NIS = z_gnss_pred_gauss.mahalanobis_distance_sq(pos)

    else:

        z = z_gnss_pred_gauss.marginalize(marginal_idxs)
        pos = pos[marginal_idxs]

        NIS = z.mahalanobis_distance_sq(pos)

    return NIS


def get_error(x_true: NominalState,
              x_nom: NominalState,
              ) -> 'ndarray[15]':
    """Finds the error (difference) between True state and 
    nominal state. See (Table 10.1).


    Returns:
        error (ndarray[15]): difference between x_true and x_nom. 
    """

    error = np.zeros(15)

    error[0:3] = x_true.pos - x_nom.pos
    error[3:6] = x_true.vel - x_nom.vel
    error[6:9] = (x_nom.ori.conjugate()@x_true.ori).as_avec()
    error[9:12] = x_true.accm_bias - x_nom.accm_bias
    error[12:15] = x_true.gyro_bias - x_nom.gyro_bias

    return error


def get_NEES(error: 'ndarray[15]',
             x_err: ErrorStateGauss,
             marginal_idxs: Optional[Sequence[int]] = None
             ) -> float:
    """Calculate NEES

    Args:
        error (ndarray[15]): errors between x_true and x_nom (from get_error)
        x_err (ErrorStateGauss): estimated error
        marginal_idxs (Optional[Sequence[int]]): Sequence of marginal indexes.
            For example used for calculating NEES for only the position. 

    Returns:
        NEES (float): NEES value
    """

    error_marginalized = error[marginal_idxs]
    x_err_marginalized = x_err.marginalize(marginal_idxs)

    NEES = x_err_marginalized.mahalanobis_distance_sq(error_marginalized)

    return NEES


def get_time_pairs(unique_data, data):
    """match data from two different time series based on timestamps"""
    gt_dict = dict(([x.ts, x] for x in unique_data))
    pairs = [(gt_dict[x.ts], x) for x in data if x.ts in gt_dict]
    times = [pair[0].ts for pair in pairs]
    return times, pairs
