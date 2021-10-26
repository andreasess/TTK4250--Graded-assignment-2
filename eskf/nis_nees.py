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

    z_bar = z_gnss_pred_gauss.mean
    S = z_gnss_pred_gauss.cov
    v = z_gnss.pos - z_bar

    if marginal_idxs == None:
        NIS = v.T @ np.linalg.inv(S) @ v

    else:

        col = np.array(marginal_idxs)
        rows = np.ix_(np.array(marginal_idxs))

        v_indxed = v[col]
        S_indxed = S[rows, col]

        NIS = v_indxed.T @ np.linalg.inv(S_indxed) @ v_indxed

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

    P = x_err.cov

    if marginal_idxs == None:
        error_indxed = error
        P_indxed = P
        
    else:
        col = np.array(marginal_idxs)
        all_indx = np.ix_(np.array(marginal_idxs), col)

        error_indxed = error[col]
        P_indxed = P[all_indx]

    NEES = error_indxed.T @ np.linalg.inv(P_indxed) @ error_indxed

    return NEES


def get_time_pairs(unique_data, data):
    """match data from two different time series based on timestamps"""
    gt_dict = dict(([x.ts, x] for x in unique_data))
    pairs = [(gt_dict[x.ts], x) for x in data if x.ts in gt_dict]
    times = [pair[0].ts for pair in pairs]
    return times, pairs
