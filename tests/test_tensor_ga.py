import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.geometry.tensor_ga import TensorGA

def test_basis_product():
    print("Testing Basis Product...")
    ga = TensorGA(device='cpu')
    
    # Create batch of 1
    # e1 (index 1) * e2 (index 2) should be e12 (index 3)
    e1 = torch.zeros(1, 16)
    e1[0, 1] = 1.0
    
    e2 = torch.zeros(1, 16)
    e2[0, 2] = 1.0
    
    e12 = ga.geometric_product(e1, e2)
    
    print(f"e1: {e1[0].nonzero().tolist()}")
    print(f"e2: {e2[0].nonzero().tolist()}")
    print(f"e1 * e2 result indices: {e12[0].nonzero().tolist()}")
    print(f"e1 * e2 result value at index 3: {e12[0, 3]}")
    
    if e12[0, 3] == 1.0 and e12[0].sum() == 1.0:
        print("PASS: e1 * e2 = e12")
    else:
        print("FAIL: Basis product incorrect")

    # Test anti-commutativity: e2 * e1 = -e12
    e21 = ga.geometric_product(e2, e1)
    print(f"e2 * e1 result value at index 3: {e21[0, 3]}")
    if e21[0, 3] == -1.0:
        print("PASS: e2 * e1 = -e12")
    else:
        print("FAIL: Anti-commutativity incorrect")

def test_rotor():
    print("\nTesting Rotor Estimation...")
    ga = TensorGA(device='cpu')
    
    # Rotate e1 to e2 (90 degrees in e12 plane)
    u = torch.zeros(1, 16)
    u[0, 1] = 1.0 # e1
    
    v = torch.zeros(1, 16)
    v[0, 2] = 1.0 # e2
    
    # R = (1 + vu) / |1+vu| = (1 + e2e1)/sqrt(2) = (1 - e12)/sqrt(2)
    # Cos(45) - Sin(45)e12
    R = ga.estimate_rotor(u, v)
    
    print(f"Rotor scalar: {R[0, 0]:.4f}")
    print(f"Rotor bivector (idx 3): {R[0, 3]:.4f}")
    
    expected_val = 1.0 / (2**0.5)
    if torch.isclose(R[0, 0], torch.tensor(expected_val)) and torch.isclose(R[0, 3], torch.tensor(-expected_val)):
        print("PASS: Rotor correct for 90 degree rotation")
    else:
        print("FAIL: Rotor values incorrect")

if __name__ == "__main__":
    test_basis_product()
    test_rotor()
