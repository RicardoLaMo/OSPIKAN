"""
Mathematical Invariant Validator for Geometric Algebra operations.

Validates that GA operations satisfy fundamental mathematical properties:
- Rotor unitarity: R * R̃ = 1
- Grade purity: rotors are even-grade only
- Magnitude preservation: |R*v*R̃| = |v|
- Eigenvalue properties: real, positive, orthogonal eigenvectors
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class GAInvariantValidator:
    """
    Validates geometric algebra mathematical invariants.

    Ensures all GA operations satisfy their defining mathematical properties,
    providing confidence in numerical correctness and detecting edge cases.
    """

    def __init__(self, tolerance: float = 1e-5):
        """
        Initialize validator with numerical tolerance.

        Args:
            tolerance: Maximum acceptable numerical error for invariant checks
        """
        self.tolerance = tolerance
        self.validation_log: List[Dict] = []

    def validate_rotor(self, ga_engine, R: torch.Tensor) -> Dict:
        """
        Validates rotor mathematical properties.

        Checks:
        1. Unitarity: R * R̃ = 1
        2. Even grade only: R ∈ {scalar + bivector}
        3. Magnitude preservation: |R*v*R̃| = |v|

        Args:
            ga_engine: TensorGA instance for GA operations
            R: [batch, 16] rotor tensor

        Returns:
            dict with validation results:
                - is_valid: bool, overall validation status
                - checks: dict of individual check results
                - warnings: list of warning messages
        """
        results = {
            'is_valid': True,
            'checks': {},
            'warnings': []
        }

        batch_size = R.shape[0]

        # Check 1: Unitarity R * R̃ = 1
        R_rev = ga_engine.reverse(R)
        RR_rev = ga_engine.geometric_product(R, R_rev)

        # Scalar part should be 1
        scalar_error = torch.abs(RR_rev[:, 0] - 1.0)
        unitarity_satisfied = torch.all(scalar_error < self.tolerance)

        # Other components should be ~0
        other_components_error = torch.norm(RR_rev[:, 1:], dim=1)
        purity_satisfied = torch.all(other_components_error < self.tolerance)

        results['checks']['unitarity'] = {
            'passed': bool(unitarity_satisfied and purity_satisfied),
            'max_scalar_error': float(torch.max(scalar_error)),
            'max_other_error': float(torch.max(other_components_error))
        }

        if not unitarity_satisfied or not purity_satisfied:
            results['is_valid'] = False
            results['warnings'].append(
                f"Rotor unitarity failed: scalar_error={torch.max(scalar_error):.2e}, "
                f"other_error={torch.max(other_components_error):.2e}"
            )

        # Check 2: Even grade only (scalar + bivector for rotors)
        # Indices: 0 (scalar), 3,5,6,9,10,12 (bivectors)
        even_grade_mask = torch.zeros(16, device=R.device)
        even_grade_mask[[0, 3, 5, 6, 9, 10, 12]] = 1.0
        odd_components = R * (1 - even_grade_mask)
        odd_grade_error = torch.norm(odd_components, dim=1)
        grade_satisfied = torch.all(odd_grade_error < self.tolerance)

        results['checks']['even_grade'] = {
            'passed': bool(grade_satisfied),
            'max_odd_grade_error': float(torch.max(odd_grade_error))
        }

        if not grade_satisfied:
            results['warnings'].append(
                f"Rotor contains odd grades: error={torch.max(odd_grade_error):.2e}"
            )

        # Check 3: Magnitude preservation for random test vectors
        test_vectors = torch.randn(batch_size, 16, device=R.device)
        vec_mask = torch.zeros(16, device=R.device)
        vec_mask[[1, 2, 4, 8]] = 1.0
        test_vectors = test_vectors * vec_mask

        original_mag_sq = ga_engine.magnitude_sq(test_vectors)
        rotated_vectors = ga_engine.apply_rotor(R, test_vectors)
        rotated_mag_sq = ga_engine.magnitude_sq(rotated_vectors)

        mag_error = torch.abs(original_mag_sq - rotated_mag_sq)
        magnitude_preserved = torch.all(mag_error < self.tolerance)

        results['checks']['magnitude_preservation'] = {
            'passed': bool(magnitude_preserved),
            'max_error': float(torch.max(mag_error))
        }

        if not magnitude_preserved:
            results['warnings'].append(
                f"Rotor doesn't preserve magnitude: error={torch.max(mag_error):.2e}"
            )
            results['is_valid'] = False

        return results

    def validate_eigendecomposition(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        original_matrix: np.ndarray
    ) -> Dict:
        """
        Validates eigenvalue decomposition properties.

        Checks:
        1. All eigenvalues real and non-negative
        2. Eigenvectors orthonormal
        3. Reconstruction: C = U * Λ * U^T
        4. Condition number reasonable

        Args:
            eigenvalues: [k] eigenvalues
            eigenvectors: [n, k] eigenvector matrix
            original_matrix: [n, n] original correlation/covariance matrix

        Returns:
            dict with validation results
        """
        results = {
            'is_valid': True,
            'checks': {},
            'warnings': []
        }

        # Check 1: Real and non-negative eigenvalues
        if not np.all(np.isreal(eigenvalues)):
            results['is_valid'] = False
            results['warnings'].append("Eigenvalues contain imaginary components")

        if not np.all(eigenvalues >= -self.tolerance):
            results['is_valid'] = False
            neg_eigs = eigenvalues[eigenvalues < -self.tolerance]
            results['warnings'].append(
                f"Negative eigenvalues detected: min={np.min(eigenvalues):.2e}"
            )

        results['checks']['eigenvalue_validity'] = {
            'all_real': bool(np.all(np.isreal(eigenvalues))),
            'all_non_negative': bool(np.all(eigenvalues >= -self.tolerance)),
            'min_eigenvalue': float(np.min(eigenvalues)),
            'max_eigenvalue': float(np.max(eigenvalues))
        }

        # Check 2: Orthonormality of eigenvectors
        gram_matrix = eigenvectors.T @ eigenvectors
        identity = np.eye(eigenvectors.shape[1])
        orthogonality_error = np.linalg.norm(gram_matrix - identity)

        orthogonal = orthogonality_error < self.tolerance * 10  # Relaxed tolerance

        results['checks']['orthogonality'] = {
            'passed': bool(orthogonal),
            'error': float(orthogonality_error)
        }

        if not orthogonal:
            results['warnings'].append(
                f"Eigenvectors not orthonormal: error={orthogonality_error:.2e}"
            )

        # Check 3: Reconstruction accuracy
        Lambda = np.diag(eigenvalues)
        reconstructed = eigenvectors @ Lambda @ eigenvectors.T

        # Only compare relevant sub-block
        k = len(eigenvalues)
        n = min(original_matrix.shape[0], eigenvectors.shape[0])
        reconstruction_error = np.linalg.norm(
            reconstructed[:n, :n] - original_matrix[:n, :n]
        )

        reconstruction_ok = reconstruction_error < self.tolerance * 100  # Relaxed

        results['checks']['reconstruction'] = {
            'passed': bool(reconstruction_ok),
            'error': float(reconstruction_error)
        }

        # Check 4: Condition number
        if len(eigenvalues) > 1 and eigenvalues[-1] > 1e-10:
            condition_number = eigenvalues[0] / eigenvalues[-1]
        else:
            condition_number = np.inf

        results['checks']['condition_number'] = {
            'value': float(condition_number),
            'well_conditioned': bool(condition_number < 1e6)
        }

        if condition_number > 1e6:
            results['warnings'].append(
                f"Poor condition number: {condition_number:.2e}"
            )

        return results

    def log_validation(self, validation_name: str, results: Dict):
        """
        Logs validation results for audit trail.

        Args:
            validation_name: Name of validation (e.g., "rotor_validation")
            results: Results dict from validation
        """
        import datetime
        self.validation_log.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'validation': validation_name,
            'results': results
        })

        if not results['is_valid']:
            logger.warning(f"{validation_name} FAILED: {results['warnings']}")
        else:
            logger.info(f"{validation_name} PASSED")

    def get_validation_summary(self) -> Dict:
        """
        Returns summary of all validations performed.

        Returns:
            dict with summary statistics
        """
        if not self.validation_log:
            return {
                'total_validations': 0,
                'passed': 0,
                'failed': 0,
                'failure_rate': 0.0
            }

        total = len(self.validation_log)
        failed = sum(1 for v in self.validation_log if not v['results']['is_valid'])
        passed = total - failed

        return {
            'total_validations': total,
            'passed': passed,
            'failed': failed,
            'failure_rate': failed / total if total > 0 else 0.0,
            'recent_failures': [
                {
                    'validation': v['validation'],
                    'timestamp': v['timestamp'],
                    'warnings': v['results']['warnings']
                }
                for v in self.validation_log[-10:]
                if not v['results']['is_valid']
            ]
        }
