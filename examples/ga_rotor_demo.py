"""
Demonstration of TRUE Geometric Algebra Rotor Analysis

This script demonstrates the corrected geometric algebra implementation
with proper rotors, bivectors, and sandwich products replacing the
previous matrix-based approach.

Run this to see how TRUE GA rotors work for regime analysis.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.geometry.tensor_ga import TensorGA
from src.validation.ga_invariants import GAInvariantValidator


def demo_basic_rotor_operations():
    """Demonstrates basic GA rotor operations."""
    print("=" * 70)
    print("DEMO 1: Basic Geometric Algebra Rotor Operations")
    print("=" * 70)

    ga = TensorGA(device='cpu')
    validator = GAInvariantValidator(tolerance=1e-5)

    # Create two vectors
    print("\n1. Creating vectors u and v:")
    u = torch.zeros(1, 16)
    u[0, 1] = 1.0  # e1 direction (return)
    print(f"   u = e1 (return direction)")

    v = torch.zeros(1, 16)
    angle_deg = 45.0
    angle_rad = angle_deg * np.pi / 180.0
    v[0, 1] = np.cos(angle_rad)  # e1 component
    v[0, 2] = np.sin(angle_rad)  # e2 component (volatility)
    print(f"   v = {v[0,1]:.3f}*e1 + {v[0,2]:.3f}*e2 (45° rotation)")

    # Estimate rotor
    print("\n2. Estimating rotor R that rotates u to v:")
    R = ga.estimate_rotor(u, v)
    print(f"   Rotor R (scalar + bivector components):")
    print(f"   Scalar:  {R[0,0]:.4f}")
    print(f"   e12:     {R[0,3]:.4f} (return-volatility plane)")

    # Validate rotor
    print("\n3. Validating rotor invariants:")
    validation = validator.validate_rotor(ga, R)
    if validation['is_valid']:
        print("   ✓ Rotor is mathematically valid!")
        print(f"   ✓ Unitarity error: {validation['checks']['unitarity']['max_scalar_error']:.2e}")
        print(f"   ✓ Magnitude preservation error: {validation['checks']['magnitude_preservation']['max_error']:.2e}")
    else:
        print("   ✗ Rotor validation failed:")
        for warning in validation['warnings']:
            print(f"     - {warning}")

    # Extract bivector and angle
    print("\n4. Extracting bivector and rotation angle:")
    B, angle, axis_plane = ga.extract_bivector(R)
    print(f"   Rotation angle: {float(angle[0]) * 180 / np.pi:.2f}°")

    # Interpret rotation plane
    plane_interp = ga.bivector_to_rotation_plane(B)
    print(f"   Dominant rotation plane: {plane_interp[0]['dominant_plane']}")
    print(f"   Plane magnitude: {plane_interp[0]['dominant_magnitude']:.4f}")

    # Apply rotor using sandwich product
    print("\n5. Applying rotor to vector u using sandwich product v' = R*u*R̃:")
    v_rotated = ga.apply_rotor(R, u)
    v_rotated_vec = ga.extract_grade(v_rotated, grade=1)
    print(f"   Result: {v_rotated_vec[0,1]:.4f}*e1 + {v_rotated_vec[0,2]:.4f}*e2")
    print(f"   Target: {v[0,1]:.4f}*e1 + {v[0,2]:.4f}*e2")

    error = torch.norm(v_rotated_vec - ga.extract_grade(v, grade=1))
    print(f"   Error: {float(error):.2e}")

    if error < 1e-4:
        print("   ✓ Rotation successful!")


def demo_edge_cases():
    """Demonstrates edge case handling."""
    print("\n" + "=" * 70)
    print("DEMO 2: Edge Case Handling (Anti-parallel and Orthogonal)")
    print("=" * 70)

    ga = TensorGA(device='cpu')

    # Anti-parallel case
    print("\n1. Anti-parallel vectors (180° rotation):")
    u = torch.zeros(1, 16)
    u[0, 1] = 1.0  # e1

    v = torch.zeros(1, 16)
    v[0, 1] = -1.0  # -e1

    R = ga.estimate_rotor(u, v)
    B, angle, axis_plane = ga.extract_bivector(R)

    print(f"   u = e1")
    print(f"   v = -e1")
    print(f"   Rotation angle: {float(angle[0]) * 180 / np.pi:.2f}°")

    v_rotated = ga.apply_rotor(R, u)
    v_rotated_vec = ga.extract_grade(v_rotated, grade=1)
    error = torch.norm(v_rotated_vec - ga.extract_grade(v, grade=1))

    if error < 1e-4:
        print("   ✓ Anti-parallel case handled correctly!")
    else:
        print(f"   ✗ Error: {float(error):.2e}")

    # Orthogonal case
    print("\n2. Orthogonal vectors (90° rotation):")
    u = torch.zeros(1, 16)
    u[0, 1] = 1.0  # e1 (return)

    v = torch.zeros(1, 16)
    v[0, 2] = 1.0  # e2 (volatility)

    R = ga.estimate_rotor(u, v)
    B, angle, axis_plane = ga.extract_bivector(R)

    print(f"   u = e1 (return)")
    print(f"   v = e2 (volatility)")
    print(f"   Rotation angle: {float(angle[0]) * 180 / np.pi:.2f}°")

    plane_interp = ga.bivector_to_rotation_plane(B)
    print(f"   Rotation plane: {plane_interp[0]['dominant_plane']}")

    v_rotated = ga.apply_rotor(R, u)
    v_rotated_vec = ga.extract_grade(v_rotated, grade=1)
    error = torch.norm(v_rotated_vec - ga.extract_grade(v, grade=1))

    if error < 1e-4:
        print("   ✓ Orthogonal case handled correctly!")
    else:
        print(f"   ✗ Error: {float(error):.2e}")


def demo_regime_rotation():
    """Demonstrates regime analysis with synthetic data."""
    print("\n" + "=" * 70)
    print("DEMO 3: Regime Rotation Analysis with Synthetic Data")
    print("=" * 70)

    ga = TensorGA(device='cpu')

    # Generate synthetic regime transition
    print("\n1. Generating synthetic data: Calm → Stressed regime transition")

    n_points = 100
    dates = pd.date_range('2024-01-01', periods=n_points, freq='D')

    # Create smooth transition from calm to stressed
    # Calm: low volatility, steady returns
    # Stressed: high volatility, correlated moves

    returns = np.zeros(n_points)
    volatility = np.zeros(n_points)
    momentum = np.zeros(n_points)
    volume_ratio = np.zeros(n_points)

    for i in range(n_points):
        # Smooth transition using sigmoid
        stress_level = 1 / (1 + np.exp(-(i - 50) / 5))

        # Returns become more negative under stress
        returns[i] = -0.001 * stress_level + np.random.normal(0, 0.01 * (1 + stress_level))

        # Volatility increases
        volatility[i] = 0.1 + 0.2 * stress_level + np.random.normal(0, 0.01)

        # Momentum becomes negative
        momentum[i] = 0.005 * (1 - stress_level) + np.random.normal(0, 0.01)

        # Volume increases
        volume_ratio[i] = 1.0 + 0.5 * stress_level + np.random.normal(0, 0.1)

    # Create tensor data
    data_tensor = torch.tensor(
        np.column_stack([returns, volatility, momentum, volume_ratio]),
        dtype=torch.float32
    )

    print(f"   Generated {n_points} data points")
    print(f"   Features: log_return, volatility, momentum, volume_ratio")

    # Embed data
    print("\n2. Embedding data into geometric algebra multivectors:")
    embeddings = ga.embed_data(data_tensor)
    print(f"   Embeddings shape: {embeddings.shape}")

    # Compute rotors between consecutive time steps
    print("\n3. Computing rotors between consecutive states:")
    rotors = []
    angles = []
    dominant_planes = []

    for i in range(len(embeddings) - 1):
        R = ga.estimate_rotor(embeddings[i:i+1], embeddings[i+1:i+2])
        B, angle, axis_plane = ga.extract_bivector(R)
        plane_interp = ga.bivector_to_rotation_plane(B)

        rotors.append(R)
        angles.append(float(angle[0]) * 180 / np.pi)  # Convert to degrees
        dominant_planes.append(plane_interp[0]['dominant_plane'])

    print(f"   Computed {len(rotors)} rotors")

    # Analyze rotation patterns
    print("\n4. Analyzing rotation patterns:")

    # Calm period (first 40 points)
    calm_angles = angles[:40]
    print(f"   Calm period (days 0-40):")
    print(f"     Mean rotation: {np.mean(calm_angles):.2f}°")
    print(f"     Std rotation:  {np.std(calm_angles):.2f}°")

    # Transition period (points 40-60)
    transition_angles = angles[40:60]
    print(f"   Transition period (days 40-60):")
    print(f"     Mean rotation: {np.mean(transition_angles):.2f}°")
    print(f"     Std rotation:  {np.std(transition_angles):.2f}°")

    # Stressed period (points 60+)
    stressed_angles = angles[60:]
    print(f"   Stressed period (days 60+):")
    print(f"     Mean rotation: {np.mean(stressed_angles):.2f}°")
    print(f"     Std rotation:  {np.std(stressed_angles):.2f}°")

    # Most common rotation planes
    from collections import Counter
    plane_counts = Counter(dominant_planes)
    print(f"\n   Most common rotation planes:")
    for plane, count in plane_counts.most_common(3):
        print(f"     {plane}: {count} occurrences ({count/len(dominant_planes)*100:.1f}%)")

    # Visualization
    print("\n5. Creating visualization...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Rotation angles over time
    axes[0].plot(dates[1:], angles, linewidth=1, color='purple', alpha=0.7)
    axes[0].axvline(dates[50], color='red', linestyle='--', alpha=0.5, label='Regime transition')
    axes[0].set_title('GA Rotor Rotation Angles Over Time (True Clifford Algebra)', fontweight='bold')
    axes[0].set_ylabel('Rotation Angle (degrees)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Rolling mean of rotation angles
    window = 10
    rolling_mean = pd.Series(angles).rolling(window).mean()
    axes[1].plot(dates[1:], rolling_mean, linewidth=2, color='darkblue')
    axes[1].axvline(dates[50], color='red', linestyle='--', alpha=0.5)
    axes[1].set_title(f'Rolling {window}-Day Mean Rotation', fontweight='bold')
    axes[1].set_ylabel('Mean Rotation (degrees)')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Regime indicator (based on rotation threshold)
    regime = np.array(['Calm' if a < 30 else 'Transition' if a < 60 else 'Stressed'
                      for a in angles])
    regime_numeric = np.array([1 if r == 'Calm' else 2 if r == 'Transition' else 3
                               for r in regime])

    colors = {'Calm': 'green', 'Transition': 'orange', 'Stressed': 'red'}
    for regime_type in ['Calm', 'Transition', 'Stressed']:
        mask = regime == regime_type
        if mask.any():
            axes[2].scatter(dates[1:][mask], regime_numeric[mask],
                          c=colors[regime_type], label=regime_type, s=20, alpha=0.6)

    axes[2].axvline(dates[50], color='black', linestyle='--', alpha=0.5, label='True transition')
    axes[2].set_title('Regime Detection (GA-based)', fontweight='bold')
    axes[2].set_ylabel('Regime')
    axes[2].set_yticks([1, 2, 3])
    axes[2].set_yticklabels(['Calm', 'Transition', 'Stressed'])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('ga_rotor_demo_output.png', dpi=150, bbox_inches='tight')
    print("   ✓ Visualization saved to: ga_rotor_demo_output.png")

    plt.show()

    print("\n" + "=" * 70)
    print("✓ Demo complete! The visualization shows how TRUE GA rotors detect regime transitions.")
    print("  Notice how rotation angles spike during the regime transition around day 50.")
    print("=" * 70)


def main():
    """Run all demonstrations."""
    print("\n")
    print("=" * 70)
    print("  TRUE GEOMETRIC ALGEBRA (Clifford Algebra) DEMONSTRATION")
    print("  Replacing Matrix-Based 'Rotors' with Real GA Rotors")
    print("=" * 70)
    print()
    print("This demonstration shows the CORRECTED implementation:")
    print("  - Proper rotor estimation (handles anti-parallel & orthogonal cases)")
    print("  - Bivector extraction and rotation plane analysis")
    print("  - Sandwich product application: v' = R*v*R̃")
    print("  - Mathematical invariant validation")
    print()

    try:
        demo_basic_rotor_operations()
        demo_edge_cases()
        demo_regime_rotation()

        print("\n" + "=" * 70)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Review ga_rotor_demo_output.png to see regime detection in action")
        print("2. Integrate this GA rotor logic into the notebook Cell 8d")
        print("3. Run tests: pytest tests/test_tensor_ga_extended.py -v")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
