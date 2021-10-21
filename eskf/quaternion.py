import numpy as np
from numpy import arcsin, arctan2, ndarray
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from config import DEBUG

import solution
from cross_matrix import get_cross_matrix


@dataclass
class RotationQuaterion:
    """Class representing a rotation quaternion (norm = 1). Has some useful
    methods for converting between rotation representations.

    Hint: You can implement all methods yourself, or use scipys Rotation class.
    scipys Rotation uses the xyzw notation for quats while the book uses wxyz
    (this i really annoying, I know).

    Args:
        real_part (float): eta (n) in the book, w in scipy notation
        vec_part (ndarray[3]): epsilon in the book, (x,y,z) in scipy notation
    """
    real_part: float
    vec_part: 'ndarray[3]'

    def __post_init__(self):
        if DEBUG:
            assert len(self.vec_part) == 3

        norm = np.sqrt(self.real_part**2 + sum(self.vec_part**2))
        if not np.allclose(norm, 1):
            self.real_part /= norm
            self.vec_part /= norm

        if self.real_part < 0:
            self.real_part *= -1
            self.vec_part *= -1

    def multiply(self, other: 'RotationQuaterion') -> 'RotationQuaterion':
        """Multiply two rotation quaternions
        Hint: see (10.33)

        As __matmul__ is implemented for this class, you can use:
        q1@q2 which is equivalent to q1.multiply(q2)

        Args:
            other (RotationQuaternion): the other quaternion    
        Returns:
            quaternion_product (RotationQuaternion): the product
        """

        # TODO replace this with your own code
        #quaternion_product = solution.quaternion.RotationQuaterion.multiply(
        #    self, other)

        eta_a = self.real_part
        epsilon_a = self.vec_part

        eta_b = other.real_part
        epsilon_b = other.vec_part
        epsilon_cross_prod = get_cross_matrix(epsilon_a) @ epsilon_b

        quatprod_real_part = eta_a*eta_b - epsilon_a@epsilon_b
        quatprod_vec_part = eta_b * epsilon_a + eta_a * epsilon_b + epsilon_cross_prod

        quaternion_product = RotationQuaterion(quatprod_real_part, quatprod_vec_part)

        return quaternion_product

    def conjugate(self) -> 'RotationQuaterion':
        """Get the conjugate of the RotationQuaternion"""

        # TODO replace this with your own code
        #conj = solution.quaternion.RotationQuaterion.conjugate(self)
        conj = RotationQuaterion(self.real_part, -self.vec_part)

        return conj

    def as_rotmat(self) -> 'ndarray[3,3]':
        """Get the rotation matrix representation of self

        Returns:
            R (ndarray[3,3]): rotation matrix
        """

        # TODO replace this with your own code
        #R = solution.quaternion.RotationQuaterion.as_rotmat(self)
        I = np.eye(3)
        eta = self.real_part
        epsilon = self.vec_part
        skewsym_epsilon = get_cross_matrix(epsilon)

        R = I + 2*eta*skewsym_epsilon + 2*skewsym_epsilon@skewsym_epsilon

        return R

    @property
    def R(self) -> 'ndarray[3,3]':
        return self.as_rotmat()

    def as_euler(self) -> 'ndarray[3]':
        """Get the euler angle representation of self

        Returns:
            euler (ndarray[3]): extrinsic xyz euler angles (roll, pitch, yaw)
        """

        # TODO replace this with your own code
        #euler = solution.quaternion.RotationQuaterion.as_euler(self)
        eta = self.real_part
        epsilon = self.vec_part

        phi = arctan2(2*(epsilon[2]*epsilon[1] + eta*epsilon[0]),
                      eta**2 - epsilon[0]**2 - epsilon[1]**2 + epsilon[2]**2)

        theta = arcsin(2*(eta*epsilon[1] - epsilon[0]*epsilon[2]))

        psi = arctan2(2*(epsilon[0]*epsilon[1] + eta*epsilon[2]),
                      eta**2 + epsilon[0]**2 - epsilon[1]**2 - epsilon[2]**2)
        
        euler = np.array([phi, theta, psi])

        return euler

    def as_avec(self) -> 'ndarray[3]':
        """Get the angles vector representation of self

        Returns:
            euler (ndarray[3]): extrinsic xyz euler angles (roll, pitch, yaw)
        """

        # TODO replace this with your own code
        #avec = solution.quaternion.RotationQuaterion.as_avec(self)
        r = Rotation.from_matrix(self.as_rotmat())
        avec = r.as_rotvec()
        
        return avec

    @staticmethod
    def from_euler(euler: 'ndarray[3]') -> 'RotationQuaterion':
        """Get a rotation quaternion from euler angles
        usage: rquat = RotationQuaterion.from_euler(euler)

        Args:
            euler (ndarray[3]): extrinsic xyz euler angles (roll, pitch, yaw)

        Returns:
            rquat (RotationQuaternion): the rotation quaternion
        """
        scipy_quat = Rotation.from_euler('xyz', euler).as_quat()
        rquat = RotationQuaterion(scipy_quat[3], scipy_quat[:3])
        return rquat

    def _as_scipy_quat(self):
        """If you're using scipys Rotation class, this can be handy"""
        return np.append(self.vec_part, self.real_part)

    def __iter__(self):
        return iter([self.real_part, self.vec_part])

    def __matmul__(self, other) -> 'RotationQuaterion':
        """Lets u use the @ operator, q1@q2 == q1.multiply(q2)"""
        return self.multiply(other)
