import numpy as np
from numpy import cos, ndarray, sin, zeros
import scipy
from dataclasses import dataclass, field
from typing import Tuple
from functools import cache

from datatypes.multivargaussian import MultiVarGaussStamped
from datatypes.measurements import (ImuMeasurement,
                                    CorrectedImuMeasurement,
                                    GnssMeasurement)
from datatypes.eskf_states import NominalState, ErrorStateGauss
from utils.indexing import block_3x3
from utils.indexing import block_15x15

from quaternion import RotationQuaterion
from cross_matrix import get_cross_matrix

import solution


@dataclass
class ESKF():

    accm_std: float
    accm_bias_std: float
    accm_bias_p: float

    gyro_std: float
    gyro_bias_std: float
    gyro_bias_p: float

    gnss_std_ne: float
    gnss_std_d: float

    accm_correction: 'ndarray[3,3]'
    gyro_correction: 'ndarray[3,3]'
    lever_arm: 'ndarray[3]'

    do_approximations: bool
    use_gnss_accuracy: bool = False

    Q_err: 'ndarray[12,12]' = field(init=False, repr=False)
    g: 'ndarray[3]' = np.array([0, 0, 9.82])

    def __post_init__(self):

        #self.accm_correction = np.around(self.accm_correction, 1)
        #self.gyro_correction = np.around(self.gyro_correction, 1)

        self.Q_err = scipy.linalg.block_diag(
            self.accm_std ** 2 * self.accm_correction @ self.accm_correction.T,
            self.gyro_std ** 2 * self.gyro_correction @ self.gyro_correction.T,
            self.accm_bias_std ** 2 * np.eye(3),
            self.gyro_bias_std ** 2 * np.eye(3),
        )
        self.gnss_cov = np.diag([self.gnss_std_ne]*2 + [self.gnss_std_d])**2

    def correct_z_imu(self,
                      x_nom_prev: NominalState,
                      z_imu: ImuMeasurement,
                      ) -> CorrectedImuMeasurement:
        """Correct IMU measurement so it gives a measurmenet of acceleration 
        and angular velocity in body.

        Hint: self.accm_correction and self.gyro_correction translates 
        measurements from IMU frame (probably not correct name) to body frame

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_imu (ImuMeasurement): raw IMU measurement

        Returns:
            CorrectedImuMeasurement: corrected IMU measurement
        """

        acc = self.accm_correction @ (z_imu.acc - x_nom_prev.accm_bias)
        avel = self.gyro_correction @ (z_imu.avel - x_nom_prev.gyro_bias)

        z_corr = CorrectedImuMeasurement(z_imu.ts, acc, avel)

        return z_corr

    def predict_nominal(self,
                        x_nom_prev: NominalState,
                        z_corr: CorrectedImuMeasurement,
                        ) -> NominalState:
        """Predict the nominal state, given a corrected IMU measurement

        Hint: Discrete time prediction of equation (10.58)
        See the assignment description for more hints 

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_corr (CorrectedImuMeasurement): corrected IMU measuremnt

        Returns:
            x_nom_pred (NominalState): predicted nominal state
        """

        Ts = z_corr.ts - x_nom_prev.ts

        rotmat = x_nom_prev.ori.R
        acc = rotmat @ (z_corr.acc) + self.g
        avel = z_corr.avel

        vel = x_nom_prev.vel + Ts*acc

        pos = x_nom_prev.pos + Ts*x_nom_prev.vel + ((Ts**2)/2)*acc

        kappa = Ts*avel
        kappa_norm = np.linalg.norm(kappa)
        real_part = cos(kappa_norm/2)
        
        if kappa_norm != 0:
            vec_part = sin(kappa_norm/2)*kappa.T/kappa_norm
            q = RotationQuaterion(real_part, vec_part)
            ori = x_nom_prev.ori @ q
        else:
            ori = x_nom_prev.ori

        #Bias needs discrete part    

        p_accm_d = np.exp(-self.accm_bias_p * Ts)
        acc_bias = p_accm_d*x_nom_prev.accm_bias

        p_gyro_d = np.exp(-self.gyro_bias_p * Ts)
        gyro_bias = p_gyro_d*x_nom_prev.gyro_bias

        x_nom_pred = NominalState(pos, vel, ori, acc_bias, gyro_bias, z_corr.ts)

        return x_nom_pred

    def get_error_A_continous(self,
                              x_nom_prev: NominalState,
                              z_corr: CorrectedImuMeasurement,
                              ) -> 'ndarray[15,15]':
        """Get the transition matrix, A, in (10.68)

        Hint: The S matrices can be created using get_cross_matrix. In the book
        a perfect IMU is expected (thus many I matrices). Here we have 
        to use the correction matrices, self.accm_correction and 
        self.gyro_correction, instead of som of the I matrices.  

        You can use block_3x3 to simplify indexing if you want to.
        The first I element in A can be set as A[block_3x3(0, 1)] = np.eye(3)

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_corr (CorrectedImuMeasurement): corrected IMU measurement
        Returns:
            A (ndarray[15,15]): A
        """

        #Ts = z_corr.ts - x_nom_prev.ts
        #p = 1/Ts

        A = np.zeros((15,15))

        A[block_3x3(0, 1)] = np.eye(3)
        A[block_3x3(1, 2)] = -x_nom_prev.ori.R @ get_cross_matrix(z_corr.acc)
        A[block_3x3(1, 3)] = -x_nom_prev.ori.R @ self.accm_correction
        A[block_3x3(2, 2)] = -get_cross_matrix(z_corr.avel)
        A[block_3x3(2, 4)] = -self.gyro_correction
        A[block_3x3(3, 3)] = -self.accm_bias_p*np.eye(3)
        A[block_3x3(4, 4)] = -self.gyro_bias_p*np.eye(3)

        return A

    def get_error_GQGT_continous(self,
                                 x_nom_prev: NominalState
                                 ) -> 'ndarray[15, 12]':
        """The noise covariance matrix, GQGT, in (10.68)

        From (Theorem 3.2.2) we can see that (10.68) can be written as 
        d/dt x_err = A@x_err + G@n == A@x_err + m
        where m is gaussian with mean 0 and covariance G @ Q @ G.T. Thats why
        we need GQGT.

        Hint: you can use block_3x3 to simplify indexing if you want to.
        The first I element in G can be set as G[block_3x3(2, 1)] = -np.eye(3)

        Args:
            x_nom_prev (NominalState): previous nominal state
        Returns:
            GQGT (ndarray[15, 15]): G @ Q @ G.T
        """

        G = np.zeros((15,12))
        G[block_3x3(1, 0)] = -x_nom_prev.ori.R
        G[block_3x3(2, 1)] = -np.eye(3)
        G[block_3x3(3, 2)] = np.eye(3)
        G[block_3x3(4, 3)] = np.eye(3)

        GQGT = G@self.Q_err@G.T

        return GQGT

    def get_van_loan_matrix(self, V: 'ndarray[30, 30]'):
        """Use this funciton in get_discrete_error_diff to get the van loan 
        matrix. See (4.63)

        All the tests are ran with do_approximations=False

        Args:
            V (ndarray[30, 30]): [description]

        Returns:
            VanLoanMatrix (ndarray[30, 30]): VanLoanMatrix
        """
        if self.do_approximations:
            # second order approcimation of matrix exponential which is faster
            VanLoanMatrix = np.eye(*V.shape) + V + (V@V) / 2
        else:
            VanLoanMatrix = scipy.linalg.expm(V)
        return VanLoanMatrix

    def get_discrete_error_diff(self,
                                x_nom_prev: NominalState,
                                z_corr: CorrectedImuMeasurement,
                                ) -> Tuple['ndarray[15, 15]',
                                           'ndarray[15, 15]']:
        """Get the discrete equivalents of A and GQGT in (4.63)

        Hint: you should use get_van_loan_matrix to get the van loan matrix

        See (4.5 Discretization) and (4.63) for more information. 
        Or see "Discretization of process noise" in 
        https://en.wikipedia.org/wiki/Discretization

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_corr (CorrectedImuMeasurement): corrected IMU measurement

        Returns:
            Ad (ndarray[15, 15]): discrede transition matrix
            GQGTd (ndarray[15, 15]): discrete noise covariance matrix
        """
        Ts = z_corr.ts - x_nom_prev.ts
        A = self.get_error_A_continous(x_nom_prev, z_corr)
        GQGT = self.get_error_GQGT_continous(x_nom_prev)

        
        V = zeros((30,30))

        V[block_15x15(0, 0)] = -A
        V[block_15x15(0, 1)] = GQGT
        V[block_15x15(1, 1)] = A.T

        van_loan = self.get_van_loan_matrix(V*Ts)
        
        V1 = van_loan[15:,15:]
        V2 = van_loan[:15, 15:]

        #V1 = van_loan[block_15x15(0,1)]
        #V2 = van_loan[block_15x15(1,1)]

        GQGTd = V1.T @ V2
        Ad = V1.T

        return Ad, GQGTd 

    def predict_x_err(self,
                      x_nom_prev: NominalState,
                      x_err_prev_gauss: ErrorStateGauss,
                      z_corr: CorrectedImuMeasurement,
                      ) -> ErrorStateGauss:
        """Predict the error state

        Hint: This is doing a discrete step of (10.68) where x_err 
        is a multivariate gaussian.

        Args:
            x_nom_prev (NominalState): previous nominal state
            x_err_prev_gauss (ErrorStateGauss): previous error state gaussian
            z_corr (CorrectedImuMeasurement): corrected IMU measuremnt

        Returns:
            x_err_pred (ErrorStateGauss): predicted error state
        """
        Ad, GQTd = self.get_discrete_error_diff(x_nom_prev, z_corr)

        x_err_pred = ErrorStateGauss(Ad @ x_err_prev_gauss.mean, Ad @ x_err_prev_gauss.cov @ Ad.T + GQTd, z_corr.ts)

        return x_err_pred

    def predict_from_imu(self,
                         x_nom_prev: NominalState,
                         x_err_gauss: ErrorStateGauss,
                         z_imu: ImuMeasurement,
                         ) -> Tuple[NominalState, ErrorStateGauss]:
        """Method called every time an IMU measurement is received

        Args:
            x_nom_prev (NominalState): previous nominal state
            x_err_gauss (ErrorStateGauss): previous error state gaussian
            z_imu (ImuMeasurement): raw IMU measurement

        Returns:
            x_nom_pred (NominalState): predicted nominal state
            x_err_pred (ErrorStateGauss): predicted error state
        """

        z_corr = self.correct_z_imu(x_nom_prev, z_imu)

        x_nom_pred = self.predict_nominal(x_nom_prev, z_corr)
        x_err_pred = self.predict_x_err(x_nom_prev, x_err_gauss, z_corr)

        return x_nom_pred, x_err_pred

    def get_gnss_measurment_jac(self, x_nom: NominalState) -> 'ndarray[3,15]':
        """Get the measurement jacobian, H.

        Hint: the gnss antenna has a relative position to the center given by
        self.lever_arm. How will the gnss measurement change if the drone is 
        rotated differently? Use get_cross_matrix and some other stuff :) 

        Returns:
            H (ndarray[3, 15]): [description]
        """
        R = x_nom.ori.R
        S = get_cross_matrix(self.lever_arm)
        H = np.block([np.eye(3), np.zeros((3, 3)), -R @ S, np.zeros((3, 6))])
        
        return H

    def get_gnss_cov(self, z_gnss: GnssMeasurement) -> 'ndarray[3,3]':
        """Use this function in predict_gnss_measurement to get R. 
        Get gnss covariance estimate based on gnss estimated accuracy. 

        All the test data has self.use_gnss_accuracy=False, so this does not 
        affect the tests.

        There is no given solution to this function, feel free to play around!

        Returns:
            gnss_cov (ndarray[3,3]): the estimated gnss covariance
        """
        if self.use_gnss_accuracy and z_gnss.accuracy is not None:
            # play around with this part, the suggested way is not optimal
            gnss_cov = (z_gnss.accuracy/3)**2 * self.gnss_cov

        else:
            # dont change this part
            gnss_cov = self.gnss_cov
        return gnss_cov

    def predict_gnss_measurement(self,
                                 x_nom: NominalState,
                                 x_err: ErrorStateGauss,
                                 z_gnss: GnssMeasurement,
                                 ) -> MultiVarGaussStamped:
        """Predict the gnss measurement

        Hint: z_gnss is only used in get_gnss_cov and to get timestamp for 
        the predicted measurement

        Args:
            x_nom (NominalState): previous nominal state
            x_err (ErrorStateGauss): previous error state gaussian
            z_gnss (GnssMeasurement): gnss measurement

        Returns:
            z_gnss_pred_gauss (MultiVarGaussStamped): gnss prediction gaussian
        """
        H = self.get_gnss_measurment_jac(x_nom)
        R = self.get_gnss_cov(z_gnss)
        P = x_err.cov
        
        S = H @ P @ H.T + R

        mean = x_nom.pos + x_nom.ori.R @ self.lever_arm

        z_gnss_pred_gauss = MultiVarGaussStamped(mean, S, z_gnss.ts)
        
        return z_gnss_pred_gauss

    def get_x_err_upd(self,
                      x_nom: NominalState,
                      x_err: ErrorStateGauss,
                      z_gnss_pred_gauss: MultiVarGaussStamped,
                      z_gnss: GnssMeasurement
                      ) -> ErrorStateGauss:
        """Update the error state from a gnss measurement

        Hint: see (10.75)
        Due to numerical error its recomended use the robust calculation of 
        posterior covariance.

        I_WH = np.eye(*P.shape) - W @ H
        P_upd = (I_WH @ P @ I_WH.T + W @ R @ W.T)

        Args:
            x_nom (NominalState): previous nominal state
            x_err (ErrorStateGauss): previous error state gaussian
            z_gnss_pred_gauss (MultiVarGaussStamped): gnss prediction gaussian
            z_gnss (GnssMeasurement): gnss measurement

        Returns:
            x_err_upd_gauss (ErrorStateGauss): updated error state gaussian
        """
        P = x_err.cov
        H = self.get_gnss_measurment_jac(x_nom)

        var_ne = self.gnss_std_ne ** 2
        var_d = self.gnss_std_d ** 2
        R = np.diag([var_ne, var_ne, var_d])
        W = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)

        I_WH = np.eye(*P.shape) - W @ H
        P_upd = (I_WH @ P @ I_WH.T + W @ R @ W.T)
        
        mean = W @ (z_gnss.pos - z_gnss_pred_gauss.mean)

        x_err_upd_gauss = ErrorStateGauss(mean, P_upd, z_gnss.ts)

        return x_err_upd_gauss

    def inject(self,
               x_nom_prev: NominalState,
               x_err_upd: ErrorStateGauss
               ) -> Tuple[NominalState, ErrorStateGauss]:
        """Perform the injection step

        Hint: see (10.85) and (10.72) on how to inject into nominal state.
        See (10.86) on how to find error state after injection

        Args:
            x_nom_prev (NominalState): previous nominal state
            x_err_upd (ErrorStateGauss): updated error state gaussian

        Returns:
            x_nom_inj (NominalState): nominal state after injection
            x_err_inj (ErrorStateGauss): error state gaussian after injection
        """

        q2 = RotationQuaterion(1, 0.5*x_err_upd.avec)
        quatprod = x_nom_prev.ori @ q2

        x_nom_inj = NominalState(x_nom_prev.pos + x_err_upd.pos, 
                                 x_nom_prev.vel + x_err_upd.vel,
                                 quatprod,
                                 x_nom_prev.accm_bias + x_err_upd.accm_bias,
                                 x_nom_prev.gyro_bias + x_err_upd.gyro_bias, x_err_upd.ts)
        
        mean = np.zeros(15)

        G = np.zeros((15, 15))
        G[:6, :6] = np.eye(6)
        G[6:9, 6:9] = np.eye(3) - get_cross_matrix(0.5*x_err_upd.avec)
        G[9:15, 9:15] = np.eye(6)
        P = x_err_upd.cov
        cov = G @ P @ G.T

        x_err_inj = ErrorStateGauss(mean, cov, x_err_upd.ts)

        return x_nom_inj, x_err_inj

    def update_from_gnss(self,
                         x_nom_prev: NominalState,
                         x_err_prev: NominalState,
                         z_gnss: GnssMeasurement,
                         ) -> Tuple[NominalState,
                                    ErrorStateGauss,
                                    MultiVarGaussStamped]:
        """Method called every time an gnss measurement is received.


        Args:
            x_nom_prev (NominalState): [description]
            x_nom_prev (NominalState): [description]
            z_gnss (GnssMeasurement): gnss measurement

        Returns:
            x_nom_inj (NominalState): previous nominal state 
            x_err_inj (ErrorStateGauss): previous error state
            z_gnss_pred_gauss (MultiVarGaussStamped): predicted gnss 
                measurement, used for NIS calculations.
        """
        z_gnss_pred_gauss = self.predict_gnss_measurement(x_nom_prev, x_err_prev, z_gnss)
        x_err_upd = self.get_x_err_upd(x_nom_prev, x_err_prev, z_gnss_pred_gauss, z_gnss)
        x_nom_inj, x_err_inj = self.inject(x_nom_prev, x_err_upd)
        
        return x_nom_inj, x_err_inj, z_gnss_pred_gauss
