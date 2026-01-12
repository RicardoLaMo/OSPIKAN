"""
Extended unit tests for TensorGA with edge cases and comprehensive validation.

Tests rotor estimation for:
- General cases (various angles)
- Anti-parallel vectors (180 degrees)
- Orthogonal vectors (90 degrees)
- Near-parallel vectors
- Rotor invariants (R * R̃ = 1)
- Wedge product properties
- Bivector extraction
"""

import torch
import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.geometry.tensor_ga import TensorGA


class TestRotorEstimation:
    """Test corrected rotor estimation with edge cases."""

    def test_general_case_45deg(self):
        """Test 45-degree rotation in e12 plane."""
        ga = TensorGA(device='cpu')

        # u = e1, v = (e1 + e2)/√2
        u = torch.zeros(1, 16)
        u[0, 1] = 1.0

        v = torch.zeros(1, 16)
        v[0, 1] = 1.0 / torch.sqrt(torch.tensor(2.0))
        v[0, 2] = 1.0 / torch.sqrt(torch.tensor(2.0))

        R = ga.estimate_rotor(u, v)

        # Apply rotor
        v_rotated = ga.apply_rotor(R, u)

        # Check result matches target (extract vector parts only)
        v_rotated_vec = ga.extract_grade(v_rotated, grade=1)
        v_vec = ga.extract_grade(v, grade=1)

        assert torch.allclose(v_rotated_vec, v_vec, atol=1e-5), \
            f"45° rotation failed: {v_rotated_vec} != {v_vec}"

    def test_general_case_30deg(self):
        """Test 30-degree rotation."""
        ga = TensorGA(device='cpu')

        # u = e1, v at 30 degrees from e1 in e12 plane
        u = torch.zeros(1, 16)
        u[0, 1] = 1.0

        v = torch.zeros(1, 16)
        angle = torch.tensor(30.0 * torch.pi / 180.0)  # 30 degrees
        v[0, 1] = torch.cos(angle)
        v[0, 2] = torch.sin(angle)

        R = ga.estimate_rotor(u, v)
        v_rotated = ga.apply_rotor(R, u)

        v_rotated_vec = ga.extract_grade(v_rotated, grade=1)
        v_vec = ga.extract_grade(v, grade=1)

        assert torch.allclose(v_rotated_vec, v_vec, atol=1e-4), \
            "30° rotation failed"

    def test_anti_parallel_case(self):
        """Test rotor for anti-parallel vectors (180 degrees)."""
        ga = TensorGA(device='cpu')

        # u = e1, v = -e1
        u = torch.zeros(1, 16)
        u[0, 1] = 1.0

        v = torch.zeros(1, 16)
        v[0, 1] = -1.0

        R = ga.estimate_rotor(u, v)

        # Apply and verify
        v_rotated = ga.apply_rotor(R, u)
        v_rotated_vec = ga.extract_grade(v_rotated, grade=1)
        v_vec = ga.extract_grade(v, grade=1)

        assert torch.allclose(v_rotated_vec, v_vec, atol=1e-4), \
            f"Anti-parallel rotation failed: {v_rotated_vec} != {v_vec}"

    def test_anti_parallel_case_e2(self):
        """Test anti-parallel case with different basis vector."""
        ga = TensorGA(device='cpu')

        # u = e2, v = -e2
        u = torch.zeros(1, 16)
        u[0, 2] = 1.0

        v = torch.zeros(1, 16)
        v[0, 2] = -1.0

        R = ga.estimate_rotor(u, v)
        v_rotated = ga.apply_rotor(R, u)
        v_rotated_vec = ga.extract_grade(v_rotated, grade=1)
        v_vec = ga.extract_grade(v, grade=1)

        assert torch.allclose(v_rotated_vec, v_vec, atol=1e-4), \
            "Anti-parallel rotation for e2 failed"

    def test_orthogonal_case(self):
        """Test rotor for orthogonal vectors."""
        ga = TensorGA(device='cpu')

        # u = e1, v = e2 (90 degrees)
        u = torch.zeros(1, 16)
        u[0, 1] = 1.0

        v = torch.zeros(1, 16)
        v[0, 2] = 1.0

        R = ga.estimate_rotor(u, v)

        v_rotated = ga.apply_rotor(R, u)
        v_rotated_vec = ga.extract_grade(v_rotated, grade=1)
        v_vec = ga.extract_grade(v, grade=1)

        assert torch.allclose(v_rotated_vec, v_vec, atol=1e-4), \
            f"Orthogonal rotation failed: {v_rotated_vec} != {v_vec}"

    def test_orthogonal_3d(self):
        """Test orthogonal rotation in 3D space."""
        ga = TensorGA(device='cpu')

        # u = e1, v = e3
        u = torch.zeros(1, 16)
        u[0, 1] = 1.0

        v = torch.zeros(1, 16)
        v[0, 4] = 1.0  # e3 is at index 4

        R = ga.estimate_rotor(u, v)
        v_rotated = ga.apply_rotor(R, u)
        v_rotated_vec = ga.extract_grade(v_rotated, grade=1)
        v_vec = ga.extract_grade(v, grade=1)

        assert torch.allclose(v_rotated_vec, v_vec, atol=1e-4), \
            "3D orthogonal rotation failed"

    def test_near_parallel_case(self):
        """Test rotor for nearly parallel vectors (small angle)."""
        ga = TensorGA(device='cpu')

        # u = e1, v ≈ e1 (1 degree rotation)
        u = torch.zeros(1, 16)
        u[0, 1] = 1.0

        v = torch.zeros(1, 16)
        angle = torch.tensor(1.0 * torch.pi / 180.0)  # 1 degree
        v[0, 1] = torch.cos(angle)
        v[0, 2] = torch.sin(angle)

        R = ga.estimate_rotor(u, v)
        v_rotated = ga.apply_rotor(R, u)
        v_rotated_vec = ga.extract_grade(v_rotated, grade=1)
        v_vec = ga.extract_grade(v, grade=1)

        assert torch.allclose(v_rotated_vec, v_vec, atol=1e-4), \
            "Near-parallel rotation failed"

    def test_rotor_unitarity_invariant(self):
        """Test that rotor satisfies R*R̃ = 1."""
        ga = TensorGA(device='cpu')

        u = torch.randn(5, 16)
        v = torch.randn(5, 16)

        # Make them vectors only
        vec_mask = torch.zeros(16)
        vec_mask[[1, 2, 4, 8]] = 1.0
        u = u * vec_mask
        v = v * vec_mask

        R = ga.estimate_rotor(u, v)
        R_rev = ga.reverse(R)

        # R * R̃ should be scalar 1
        product = ga.geometric_product(R, R_rev)

        # Check scalar part is 1
        assert torch.allclose(product[:, 0], torch.ones(5), atol=1e-4), \
            f"Rotor unitarity failed: scalar part = {product[:, 0]}"

        # Check all other components are ~0
        assert torch.allclose(product[:, 1:], torch.zeros(5, 15), atol=1e-4), \
            "Rotor unitarity failed: non-zero non-scalar components"

    def test_rotor_magnitude_preservation(self):
        """Test that rotor preserves vector magnitude."""
        ga = TensorGA(device='cpu')

        u = torch.zeros(3, 16)
        u[:, 1] = torch.tensor([1.0, 0.5, 2.0])

        v = torch.zeros(3, 16)
        v[:, 2] = torch.tensor([1.0, 0.5, 2.0])

        R = ga.estimate_rotor(u, v)

        # Test vectors with various magnitudes
        test_vecs = torch.zeros(3, 16)
        test_vecs[:, 1] = torch.tensor([1.0, 2.0, 0.5])
        test_vecs[:, 2] = torch.tensor([0.5, 1.0, 1.5])

        original_mag_sq = ga.magnitude_sq(test_vecs)
        rotated_vecs = ga.apply_rotor(R, test_vecs)
        rotated_mag_sq = ga.magnitude_sq(rotated_vecs)

        assert torch.allclose(original_mag_sq, rotated_mag_sq, atol=1e-4), \
            f"Magnitude not preserved: {original_mag_sq} != {rotated_mag_sq}"

    def test_batch_rotor_estimation(self):
        """Test rotor estimation with batched inputs."""
        ga = TensorGA(device='cpu')

        batch_size = 10
        u = torch.randn(batch_size, 16)
        v = torch.randn(batch_size, 16)

        vec_mask = torch.zeros(16)
        vec_mask[[1, 2, 4, 8]] = 1.0
        u = u * vec_mask
        v = v * vec_mask

        R = ga.estimate_rotor(u, v)

        # All rotors should be unitary
        R_rev = ga.reverse(R)
        product = ga.geometric_product(R, R_rev)

        assert torch.allclose(product[:, 0], torch.ones(batch_size), atol=1e-4), \
            "Batch rotor unitarity failed"


class TestWedgeProduct:
    """Test wedge product implementation."""

    def test_basis_wedge_e12(self):
        """Test e1 ∧ e2 = e12."""
        ga = TensorGA(device='cpu')

        e1 = torch.zeros(1, 16)
        e1[0, 1] = 1.0

        e2 = torch.zeros(1, 16)
        e2[0, 2] = 1.0

        e12 = ga.wedge_product(e1, e2)

        # Should only have e12 component (index 3)
        assert torch.abs(e12[0, 3] - 1.0) < 1e-6, \
            f"e1∧e2 should equal e12: got {e12[0, 3]}"

        # Other components should be zero
        non_e12_indices = [i for i in range(16) if i != 3]
        assert torch.allclose(e12[0, non_e12_indices],
                             torch.zeros(15), atol=1e-6), \
            "e1∧e2 should only have e12 component"

    def test_basis_wedge_e13(self):
        """Test e1 ∧ e3 = e13."""
        ga = TensorGA(device='cpu')

        e1 = torch.zeros(1, 16)
        e1[0, 1] = 1.0

        e3 = torch.zeros(1, 16)
        e3[0, 4] = 1.0  # e3 is at index 4

        e13 = ga.wedge_product(e1, e3)

        # Should have e13 component (index 5 = 1+4)
        assert torch.abs(e13[0, 5] - 1.0) < 1e-6, \
            f"e1∧e3 should equal e13: got {e13[0, 5]}"

    def test_wedge_antisymmetry(self):
        """Test a ∧ b = -(b ∧ a)."""
        ga = TensorGA(device='cpu')

        a = torch.randn(3, 16)
        b = torch.randn(3, 16)

        ab = ga.wedge_product(a, b)
        ba = ga.wedge_product(b, a)

        assert torch.allclose(ab, -ba, atol=1e-5), \
            "Wedge product antisymmetry failed"

    def test_wedge_self_zero(self):
        """Test a ∧ a = 0."""
        ga = TensorGA(device='cpu')

        a = torch.randn(3, 16)
        aa = ga.wedge_product(a, a)

        assert torch.allclose(aa, torch.zeros_like(aa), atol=1e-5), \
            "Self-wedge product should be zero"

    def test_wedge_product_distributivity(self):
        """Test a ∧ (b + c) = a ∧ b + a ∧ c."""
        ga = TensorGA(device='cpu')

        a = torch.randn(2, 16)
        b = torch.randn(2, 16)
        c = torch.randn(2, 16)

        left = ga.wedge_product(a, b + c)
        right = ga.wedge_product(a, b) + ga.wedge_product(a, c)

        assert torch.allclose(left, right, atol=1e-4), \
            "Wedge product distributivity failed"


class TestGradeExtraction:
    """Test grade extraction functionality."""

    def test_extract_scalar(self):
        """Test extracting scalar (grade 0) component."""
        ga = TensorGA(device='cpu')

        mv = torch.randn(3, 16)
        scalar = ga.extract_grade(mv, grade=0)

        # Only index 0 should be non-zero
        assert torch.allclose(scalar[:, 1:], torch.zeros(3, 15), atol=1e-10), \
            "Grade-0 extraction should only keep index 0"

        # Should preserve scalar value
        assert torch.allclose(scalar[:, 0], mv[:, 0], atol=1e-10), \
            "Scalar value not preserved"

    def test_extract_vector(self):
        """Test extracting vector (grade 1) components."""
        ga = TensorGA(device='cpu')

        mv = torch.randn(3, 16)
        vector = ga.extract_grade(mv, grade=1)

        # Only indices 1, 2, 4, 8 should be non-zero
        vector_indices = [1, 2, 4, 8]
        other_indices = [i for i in range(16) if i not in vector_indices]

        assert torch.allclose(vector[:, other_indices],
                             torch.zeros(3, len(other_indices)), atol=1e-10), \
            "Grade-1 extraction should only keep vector indices"

    def test_extract_bivector(self):
        """Test extracting bivector (grade 2) components."""
        ga = TensorGA(device='cpu')

        mv = torch.randn(3, 16)
        bivector = ga.extract_grade(mv, grade=2)

        # Bivector indices: 3, 5, 6, 9, 10, 12
        bivector_indices = [3, 5, 6, 9, 10, 12]
        other_indices = [i for i in range(16) if i not in bivector_indices]

        assert torch.allclose(bivector[:, other_indices],
                             torch.zeros(3, len(other_indices)), atol=1e-10), \
            "Grade-2 extraction should only keep bivector indices"


class TestBivectorExtraction:
    """Test bivector extraction and interpretation."""

    def test_extract_bivector_from_rotor_90deg(self):
        """Test extracting bivector from known 90-degree rotor."""
        ga = TensorGA(device='cpu')

        # Create rotor for 90-degree rotation in e12 plane
        # R = cos(45°) + sin(45°)*e12
        R = torch.zeros(1, 16)
        R[0, 0] = torch.cos(torch.tensor(torch.pi / 4))  # cos(45°)
        R[0, 3] = torch.sin(torch.tensor(torch.pi / 4))  # sin(45°)*e12

        B, angle, axis_plane = ga.extract_bivector(R)

        # Check angle is 90 degrees (π/2)
        expected_angle = torch.tensor([[torch.pi / 2]])
        assert torch.allclose(angle, expected_angle, atol=1e-4), \
            f"Angle extraction failed: {angle} != {expected_angle}"

        # Check bivector is in e12 plane
        assert torch.abs(axis_plane[0, 0] - 1.0) < 1e-4, \
            f"e12 component should be 1: got {axis_plane[0, 0]}"
        assert torch.allclose(axis_plane[0, 1:], torch.zeros(5), atol=1e-4), \
            "Other bivector components should be zero"

    def test_extract_bivector_from_rotor_45deg(self):
        """Test extracting bivector from 45-degree rotor."""
        ga = TensorGA(device='cpu')

        # Create 45-degree rotor
        R = torch.zeros(1, 16)
        R[0, 0] = torch.cos(torch.tensor(torch.pi / 8))  # cos(22.5°)
        R[0, 3] = torch.sin(torch.tensor(torch.pi / 8))  # sin(22.5°)*e12

        B, angle, axis_plane = ga.extract_bivector(R)

        # Check angle is 45 degrees (π/4)
        expected_angle = torch.tensor([[torch.pi / 4]])
        assert torch.allclose(angle, expected_angle, atol=1e-4), \
            f"45° angle extraction failed: {angle} != {expected_angle}"

    def test_bivector_plane_interpretation(self):
        """Test financial interpretation of bivector planes."""
        ga = TensorGA(device='cpu')

        # Create bivector with known components
        B = torch.zeros(1, 16)
        B[0, 3] = 0.7  # return-volatility (e12)
        B[0, 5] = 0.3  # return-momentum (e13)

        interpretation = ga.bivector_to_rotation_plane(B)

        assert len(interpretation) == 1, "Should have one interpretation per batch"

        result = interpretation[0]
        assert result['dominant_plane'] == 'return-volatility', \
            f"Dominant plane should be return-volatility: got {result['dominant_plane']}"

        assert abs(result['components']['return-volatility'] - 0.7) < 1e-6, \
            "return-volatility component incorrect"
        assert abs(result['components']['return-momentum'] - 0.3) < 1e-6, \
            "return-momentum component incorrect"


class TestRotorComposition:
    """Test rotor composition."""

    def test_rotor_composition_sequential(self):
        """Test composing two rotors gives correct combined rotation."""
        ga = TensorGA(device='cpu')

        # Create two 45-degree rotations
        u = torch.zeros(1, 16)
        u[0, 1] = 1.0  # e1

        v1 = torch.zeros(1, 16)
        angle1 = torch.tensor(45.0 * torch.pi / 180.0)
        v1[0, 1] = torch.cos(angle1)
        v1[0, 2] = torch.sin(angle1)

        v2 = torch.zeros(1, 16)
        angle2 = torch.tensor(90.0 * torch.pi / 180.0)
        v2[0, 1] = torch.cos(angle2)
        v2[0, 2] = torch.sin(angle2)

        R1 = ga.estimate_rotor(u, v1)
        R2 = ga.estimate_rotor(v1, v2)

        # Compose rotors
        R_total = ga.rotor_composition(R1, R2)

        # Apply composed rotor
        result = ga.apply_rotor(R_total, u)
        result_vec = ga.extract_grade(result, grade=1)

        # Should equal v2
        v2_vec = ga.extract_grade(v2, grade=1)

        assert torch.allclose(result_vec, v2_vec, atol=1e-4), \
            f"Rotor composition failed: {result_vec} != {v2_vec}"


def test_full_ga_pipeline():
    """Integration test for full GA pipeline."""
    ga = TensorGA(device='cpu')

    # Create test data
    u = torch.zeros(1, 16)
    u[0, 1] = 1.0

    v = torch.zeros(1, 16)
    v[0, 2] = 1.0

    # Estimate rotor
    R = ga.estimate_rotor(u, v)

    # Validate rotor
    R_rev = ga.reverse(R)
    product = ga.geometric_product(R, R_rev)
    assert torch.allclose(product[:, 0], torch.ones(1), atol=1e-4), \
        "Rotor not unitary"

    # Extract bivector
    B, angle, axis_plane = ga.extract_bivector(R)

    # Interpret
    interpretation = ga.bivector_to_rotation_plane(B)
    assert interpretation[0]['dominant_plane'] == 'return-volatility', \
        "Plane interpretation incorrect"

    # Apply rotor
    v_rotated = ga.apply_rotor(R, u)
    v_rotated_vec = ga.extract_grade(v_rotated, grade=1)
    v_vec = ga.extract_grade(v, grade=1)

    assert torch.allclose(v_rotated_vec, v_vec, atol=1e-4), \
        "Full pipeline failed"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
