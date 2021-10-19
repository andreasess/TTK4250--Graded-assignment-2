import numpy as np
from numpy import ndarray
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from config import DEBUG

import solution


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

        norm = self.real_part**2 + sum(self.vec_part**2)
        if not np.allclose(norm, 1):
            self.real_part /= norm
            self.vec_part /= norm

        if self.real_part < 0:
            self.real_part *= -1
            self.vec_part *= -1

    def multiply(self, other: 'RotationQuaterion') -> 'RotationQuaterion':
        solution.used["quaternion.RotationQuaterion.multiply"] = True
        """Multiply two rotation quaternions
        Hint: see (10.33)

        As __matmul__ is implemented for this class, you can use:
        q1@q2 which is equivalent to q1.multiply(q2)

        Args:
            other (RotationQuaternion): the other quaternion    
        Returns:
            quaternion_product (RotationQuaternion): the product
        """
        n_a, e_a = self
        n_b, e_b = other
        real_part = n_a*n_b - np.dot(e_a, e_b)
        vec_part = n_b*e_a + n_a*e_b + np.cross(e_a, e_b)

        quaternion_product = RotationQuaterion(real_part, vec_part)
        return quaternion_product

    def conjugate(self) -> 'RotationQuaterion':
        solution.used["quaternion.RotationQuaterion.conjugate"] = True
        """Get the conjugate of the RotationQuaternion"""
        conj = RotationQuaterion(self.real_part, -self.vec_part)
        return conj

    def as_rotmat(self) -> 'ndarray[3,3]':
        solution.used["quaternion.RotationQuaterion.as_rotmat"] = True
        """Get the rotation matrix representation of self

        Returns:
            R (ndarray[3,3]): rotation matrix
        """
        R = Rotation.from_quat(self._as_scipy_quat()).as_matrix()
        return R

    @property
    def R(self) -> 'ndarray[3,3]':
        return self.as_rotmat()

    def as_euler(self) -> 'ndarray[3]':
        solution.used["quaternion.RotationQuaterion.as_euler"] = True
        """Get the euler angle representation of self

        Returns:
            euler (ndarray[3]): extrinsic xyz euler angles (roll, pitch, yaw)
        """
        euler = Rotation.from_quat(self._as_scipy_quat()).as_euler('xyz')
        return euler

    def as_avec(self) -> 'ndarray[3]':
        solution.used["quaternion.RotationQuaterion.as_avec"] = True
        """Get the angles vector representation of self

        Returns:
            euler (ndarray[3]): extrinsic xyz euler angles (roll, pitch, yaw)
        """
        avec = Rotation.from_quat(self._as_scipy_quat()).as_rotvec()
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
