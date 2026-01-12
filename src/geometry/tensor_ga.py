import torch
import itertools

class TensorGA:
    """
    GPU-accelerated Geometric Algebra wrapper using PyTorch for Cl(4,0).
    Models financial regimes as rotations in a 4D Euclidean geometric algebra space.
    """
    def __init__(self, n_features=4, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.n_features = n_features
        self.dim = 2 ** n_features  # 16 for 4D
        
        # Basis vector definition (bits represent present basis vectors)
        # 0: scalar, 1: e1, 2: e2, 4: e3, 8: e4, ...
        self.basis = list(range(self.dim))
        
        # Precompute multiplication tables
        self._construct_cayley_table()
        
    def _construct_cayley_table(self):
        """
        Constructs the Cayley table for Geometric Product in Cl(n,0).
        Stores indices and signs for A * B.
        """
        # Table of result basis indices: [16, 16]
        self.gp_indices = torch.zeros((self.dim, self.dim), dtype=torch.long, device=self.device)
        # Table of result signs: [16, 16]
        self.gp_signs = torch.zeros((self.dim, self.dim), dtype=torch.float32, device=self.device)
        
        for i in range(self.dim):
            for j in range(self.dim):
                idx, sign = self._multiply_basis(i, j)
                self.gp_indices[i, j] = idx
                self.gp_signs[i, j] = sign

    def _multiply_basis(self, a_bits, b_bits):
        """
        Multiplies two basis blades represented by bitmaps.
        Returns (result_bitmap, sign).
        Metric is Euclidean (+,+,+,+).
        """
        # XOR gives the resulting basis blade (symmetric difference of sets)
        res_bits = a_bits ^ b_bits
        
        # Calculate sign based on swaps required to reorder
        # Standard algorithm: count swaps to bring matching vectors together
        sign = 1
        
        # Convert bitmaps to lists of active basis vectors, e.g., 3 (011) -> [0, 1]
        a_vecs = [k for k in range(self.n_features) if (a_bits >> k) & 1]
        b_vecs = [k for k in range(self.n_features) if (b_bits >> k) & 1]
        
        # Concatenate: a followed by b
        sequence = a_vecs + b_vecs
        
        swaps = 0
        # Bubble sort to canonical order
        for k in range(len(sequence)):
            for l in range(len(sequence) - 1 - k):
                if sequence[l] > sequence[l+1]:
                    sequence[l], sequence[l+1] = sequence[l+1], sequence[l]
                    swaps += 1
                elif sequence[l] == sequence[l+1]:
                    # In Euclidean metric, e_i * e_i = +1. 
                    # They annihilate from the blade representation, sign doesn't flip due to metric.
                    # But we need to account for them being adjacent.
                    pass
                    
        if swaps % 2 == 1:
            sign = -1
            
        return res_bits, sign

    def embed_data(self, data_tensor):
        """
        Embeds feature tensor [batch, 4] into multivectors.
        Mapping:
        Index 0: Scalar (1.0)
        Index 1 (1): e1
        Index 2 (2): e2
        Index 4 (4): e3
        Index 8 (8): e4
        """
        batch_size = data_tensor.shape[0]
        mv = torch.zeros(batch_size, self.dim, device=self.device)
        
        # Scalar part = 1 (Homogeneous coordinate for projective geometry)
        mv[:, 0] = 1.0 
        
        # Vector parts (indices 1, 2, 4, 8)
        # data_tensor cols: log_return, vol, mom, vol_ratio
        mv[:, 1] = data_tensor[:, 0]
        mv[:, 2] = data_tensor[:, 1]
        mv[:, 4] = data_tensor[:, 2]
        mv[:, 8] = data_tensor[:, 3]
        
        return mv

    def geometric_product(self, A, B):
        """
        Computes Geometric Product C = A * B.
        A, B: [batch, 16]
        """
        # This implementation uses memory-intensive indexing for simplicity and batch support.
        # For very large batches/dims, a custom kernel is better.
        
        batch_size = A.shape[0]
        
        # Flatten A and B for broadcasting: [batch, 16, 1] * [batch, 1, 16] -> [batch, 16, 16]
        # This gives the product of every component pair
        outer = A.unsqueeze(2) * B.unsqueeze(1) # [batch, 16, 16]
        
        # Apply signs
        signed_products = outer * self.gp_signs.unsqueeze(0) # [batch, 16, 16]
        
        # Accumulate into result based on result indices
        # We use scatter_add
        C = torch.zeros_like(A)
        
        # Flatten the last two dims to iterate or scatter
        flat_products = signed_products.view(batch_size, -1) # [batch, 256]
        flat_indices = self.gp_indices.view(-1) # [256]
        
        # Expand indices for batch
        batch_indices = flat_indices.unsqueeze(0).expand(batch_size, -1) # [batch, 256]
        
        # Scatter add
        C.scatter_add_(1, batch_indices, flat_products)
        
        return C

    def reverse(self, A):
        """
        Computes the Reverse of multivector A.
        Reverses the order of basis vectors in blades.
        Sign change: (-1)^(k(k-1)/2) for grade k.
        """
        # Precompute sign mask if efficient, or just apply per grade
        # Grade 0, 1: +, Grade 2: -, Grade 3: -, Grade 4: +
        # Indices:
        # Gr 0: 0 (+)
        # Gr 1: 1, 2, 4, 8 (+)
        # Gr 2: 3, 5, 6, 9, 10, 12 (-) (e.g. e1e2 -> e2e1 = -e1e2)
        # Gr 3: 7, 11, 13, 14 (-)
        # Gr 4: 15 (+)
        
        # Construct mask once
        if not hasattr(self, 'reverse_mask'):
            self.reverse_mask = torch.ones(self.dim, device=self.device)
            # Grade 2 (2 bits set)
            gr2 = [3, 5, 6, 9, 10, 12]
            self.reverse_mask[gr2] = -1.0
            # Grade 3 (3 bits set)
            gr3 = [7, 11, 13, 14]
            self.reverse_mask[gr3] = -1.0
            
        return A * self.reverse_mask

    def magnitude_sq(self, A):
        """
        Returns |A|^2 = Scalar part of (A * ~A)
        """
        rev_A = self.reverse(A)
        prod = self.geometric_product(A, rev_A)
        return prod[:, 0].unsqueeze(1) # Scalar part

    def normalize(self, A):
        mag_sq = self.magnitude_sq(A)
        # Avoid div by zero
        mag = torch.sqrt(torch.clamp(mag_sq, min=1e-8))
        return A / mag

    def wedge_product(self, A, B):
        """
        Computes the wedge (outer) product A ∧ B.

        Formula: A ∧ B = (A*B - B*A) / 2
        This gives the antisymmetric part of the geometric product.

        Args:
            A, B: [batch, 16] multivectors

        Returns:
            [batch, 16] multivector containing only antisymmetric part (bivector for vectors)
        """
        AB = self.geometric_product(A, B)
        BA = self.geometric_product(B, A)
        return (AB - BA) / 2.0

    def extract_grade(self, mv, grade):
        """
        Extracts components of specific grade from multivector.

        Grade structure in Cl(4,0):
        - Grade 0: scalar (index 0)
        - Grade 1: vectors (indices 1, 2, 4, 8)
        - Grade 2: bivectors (indices 3, 5, 6, 9, 10, 12)
        - Grade 3: trivectors (indices 7, 11, 13, 14)
        - Grade 4: pseudoscalar (index 15)

        Args:
            mv: [batch, 16] multivector
            grade: int in [0, 1, 2, 3, 4]

        Returns:
            [batch, 16] multivector with only specified grade components
        """
        if not hasattr(self, 'grade_masks'):
            self._construct_grade_masks()

        if grade not in self.grade_masks:
            return torch.zeros_like(mv)

        return mv * self.grade_masks[grade]

    def _construct_grade_masks(self):
        """Precomputes masks for each grade."""
        self.grade_masks = {}

        for idx in range(self.dim):
            # Count number of bits set = grade
            grade = bin(idx).count('1')

            if grade not in self.grade_masks:
                self.grade_masks[grade] = torch.zeros(self.dim, device=self.device)

            self.grade_masks[grade][idx] = 1.0

    def _find_perpendicular_bivector(self, vec):
        """
        Finds a unit bivector perpendicular to given vector.
        Used for anti-parallel rotor case (180-degree rotation).

        Args:
            vec: [batch, 16] vector

        Returns:
            [batch, 16] unit bivector
        """
        batch_size = vec.shape[0]
        result = torch.zeros_like(vec)

        # Try e1 ∧ vec for each batch element
        e1 = torch.zeros_like(vec)
        e1[:, 1] = 1.0

        wedge = self.wedge_product(e1, vec)
        mag_sq = self.magnitude_sq(wedge)

        # For elements where e1 works (not parallel)
        good_mask = (mag_sq.squeeze(1) > 1e-6)

        if good_mask.any():
            result[good_mask] = self.normalize(wedge[good_mask])

        # For elements parallel to e1, try e2 ∧ vec
        bad_mask = ~good_mask
        if bad_mask.any():
            e2 = torch.zeros_like(vec[bad_mask])
            e2[:, 2] = 1.0
            wedge2 = self.wedge_product(e2, vec[bad_mask])
            result[bad_mask] = self.normalize(wedge2)

        return result

    def estimate_rotor(self, mv_from, mv_to):
        """
        Estimates the rotor R that aligns vector u to vector v.
        Handles anti-parallel, orthogonal, and general cases correctly.

        Mathematical formulas:
        - General: R = (1 + v*u) / |1 + v*u|
        - Anti-parallel (u = -v): R = exp(π/2 * B_perp) = B_perp (bivector for 180° rotation)
        - Orthogonal: R = cos(θ/2) + sin(θ/2) * B where θ = acos(u·v), B = normalized(v ∧ u)

        Args:
            mv_from: [batch, 16] source multivector (primarily vector)
            mv_to: [batch, 16] target multivector (primarily vector)

        Returns:
            R: Rotor [batch, 16] (even-grade multivector: scalar + bivector components)
        """
        # Extract only vector parts to ensure clean rotation
        vec_mask = torch.zeros(self.dim, device=self.device)
        vec_mask[[1, 2, 4, 8]] = 1.0

        u = mv_from * vec_mask
        v = mv_to * vec_mask

        u = self.normalize(u)
        v = self.normalize(v)

        # Calculate geometric product vu = v * u
        # vu = v · u + v ∧ u (Scalar + Bivector)
        vu = self.geometric_product(v, u)

        # Extract dot product (scalar part)
        dot_uv = vu[:, 0]  # [batch]

        # Extract wedge product (bivector part)
        wedge_uv = self.extract_grade(vu, grade=2)

        batch_size = u.shape[0]
        R = torch.zeros_like(u)

        # Process each batch element with appropriate formula
        for i in range(batch_size):
            if dot_uv[i] < -0.9999:
                # Anti-parallel case: u = -v, need 180-degree rotation
                # Use perpendicular bivector
                B_perp = self._find_perpendicular_bivector(u[i:i+1])
                # For 180° rotation: R = exp(π/2 * B) = cos(π/2) + sin(π/2)*B = B
                R[i] = B_perp[0]

            elif torch.abs(dot_uv[i]) < 0.0001:
                # Orthogonal case: use half-angle formula for numerical stability
                # θ = acos(u·v) ≈ π/2
                theta = torch.acos(torch.clamp(dot_uv[i], -1.0, 1.0))
                half_theta = theta / 2.0

                # Normalize wedge product to get unit bivector
                wedge_mag_sq = self.magnitude_sq(wedge_uv[i:i+1])
                B = wedge_uv[i:i+1] / torch.sqrt(torch.clamp(wedge_mag_sq, min=1e-8))

                # R = cos(θ/2) + sin(θ/2) * B
                R[i, 0] = torch.cos(half_theta)
                R[i] += torch.sin(half_theta) * B[0]

            else:
                # General case: standard formula R = (1 + v*u) / |1 + v*u|
                rotor_unscaled = vu[i:i+1].clone()
                rotor_unscaled[:, 0] += 1.0  # Add 1 to scalar part

                # Normalize rotor
                R[i] = self.normalize(rotor_unscaled)[0]

        return R

    def apply_rotor(self, R, v):
        """
        Applies rotor R to vector v using sandwich product.

        Formula: v' = R * v * R̃
        where R̃ is the reverse of R

        This preserves the grade and magnitude of v while rotating it.

        Args:
            R: [batch, 16] rotor (even-grade multivector)
            v: [batch, 16] vector to rotate

        Returns:
            [batch, 16] rotated vector
        """
        R_rev = self.reverse(R)

        # First product: R * v
        Rv = self.geometric_product(R, v)

        # Second product: (R * v) * R̃
        v_prime = self.geometric_product(Rv, R_rev)

        return v_prime

    def rotor_composition(self, R1, R2):
        """
        Composes two rotors: R_total = R2 * R1

        Applying R_total to a vector v gives the same result as
        applying R1 first, then R2:
        v' = R_total * v * R̃_total = R2 * (R1 * v * R̃1) * R̃2

        Args:
            R1: [batch, 16] first rotor
            R2: [batch, 16] second rotor

        Returns:
            [batch, 16] composed rotor
        """
        R_composed = self.geometric_product(R2, R1)
        return self.normalize(R_composed)

    def extract_bivector(self, R):
        """
        Extracts bivector (grade-2) part from rotor and computes rotation angle.

        A rotor has the form: R = cos(θ/2) + sin(θ/2) * B
        where B is a unit bivector representing the rotation plane.

        Args:
            R: [batch, 16] rotor tensor

        Returns:
            B: [batch, 16] bivector components only
            angle: [batch, 1] rotation angle θ (in radians)
            axis_plane: [batch, 6] coefficients of bivector basis elements
                        [e12, e13, e14, e23, e24, e34]
        """
        # Extract grade-2 components (bivector)
        B = self.extract_grade(R, grade=2)

        # Scalar part gives cos(θ/2)
        scalar = R[:, 0:1]

        # Compute angle: θ = 2 * acos(scalar)
        # Clamp to avoid numerical issues
        angle = 2.0 * torch.acos(torch.clamp(scalar, -1.0, 1.0))

        # Bivector magnitude should be sin(θ/2)
        B_mag_sq = self.magnitude_sq(B)
        B_mag = torch.sqrt(torch.clamp(B_mag_sq, min=1e-8))

        # Normalize bivector to get rotation plane
        B_normalized = B / torch.clamp(B_mag, min=1e-8)

        # Extract 6 bivector basis coefficients
        # Indices in Cl(4,0): e12=3, e13=5, e14=9, e23=6, e24=10, e34=12
        axis_plane = torch.stack([
            B_normalized[:, 3],   # e1∧e2
            B_normalized[:, 5],   # e1∧e3
            B_normalized[:, 9],   # e1∧e4
            B_normalized[:, 6],   # e2∧e3
            B_normalized[:, 10],  # e2∧e4
            B_normalized[:, 12],  # e3∧e4
        ], dim=1)

        return B, angle, axis_plane

    def bivector_to_rotation_plane(self, B):
        """
        Interprets bivector as rotation plane in financial feature space.

        In Cl(4,0) with basis {e1, e2, e3, e4} representing:
        e1: log_return
        e2: volatility
        e3: momentum
        e4: volume_ratio

        A bivector like 0.3*e1∧e2 + 0.1*e1∧e3 represents:
        - 0.3 units of rotation in the return-volatility plane
        - 0.1 units of rotation in the return-momentum plane

        Args:
            B: [batch, 16] bivector tensor

        Returns:
            List of dicts with plane interpretation and magnitudes for each batch element
        """
        plane_names = [
            ('return', 'volatility'),      # e1∧e2 (idx 3)
            ('return', 'momentum'),        # e1∧e3 (idx 5)
            ('return', 'volume'),          # e1∧e4 (idx 9)
            ('volatility', 'momentum'),    # e2∧e3 (idx 6)
            ('volatility', 'volume'),      # e2∧e4 (idx 10)
            ('momentum', 'volume'),        # e3∧e4 (idx 12)
        ]

        bivector_indices = [3, 5, 9, 6, 10, 12]

        results = []
        for i in range(B.shape[0]):
            components = {}
            for (name1, name2), idx in zip(plane_names, bivector_indices):
                components[f'{name1}-{name2}'] = float(B[i, idx])

            # Find dominant plane (largest absolute magnitude)
            abs_components = {k: abs(v) for k, v in components.items()}
            dominant_plane = max(abs_components, key=abs_components.get)

            results.append({
                'components': components,
                'dominant_plane': dominant_plane,
                'dominant_magnitude': abs_components[dominant_plane]
            })

        return results

    def analyze_regimes(self, tensor_data):
        """
        Embeds data and calculates rotors between consecutive time steps.
        
        Returns:
            rotors: Rotors representing change from t to t+1
            embeddings: The embedded multivectors
        """
        embeddings = self.embed_data(tensor_data)
        
        # Calculate rotors between t and t+1
        mv_t = embeddings[:-1]
        mv_t1 = embeddings[1:]
        
        rotors = self.estimate_rotor(mv_t, mv_t1)
        
        return rotors, embeddings

