"""
Unit tests for KAN (Kolmogorov-Arnold Network) layers.

Tests cover:
1. B-spline basis function computation
2. KAN layer forward pass
3. Gradient computation for physics-informed learning
4. SPIKAN architecture
5. Market PDE residuals
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

# Skip all tests if torch is not available
pytest.importorskip("torch")


class TestBSplineBasis:
    """Tests for B-spline basis function computation."""

    def test_bspline_basis_shape(self):
        """Test that B-spline basis has correct output shape."""
        from src.physics.kan_layers import bspline_basis

        batch_size = 16
        in_features = 4
        grid_size = 5
        spline_order = 3

        # Create input and grid
        x = torch.randn(batch_size, in_features)
        n_knots = grid_size + 2 * spline_order + 1
        grid = torch.linspace(-1.5, 1.5, n_knots)

        # Compute basis
        basis = bspline_basis(x, grid, spline_order)

        # Expected shape: (batch, features, n_basis) where n_basis = grid_size + spline_order
        n_basis = grid_size + spline_order
        assert basis.shape == (batch_size, in_features, n_basis)

    def test_bspline_basis_partition_of_unity(self):
        """Test that B-spline basis sums to approximately 1 (partition of unity)."""
        from src.physics.kan_layers import bspline_basis

        x = torch.linspace(-0.9, 0.9, 50).unsqueeze(-1)  # (50, 1)
        grid = torch.linspace(-1.5, 1.5, 12)  # grid_size=5, order=3

        basis = bspline_basis(x, grid, order=3)

        # Sum should be close to 1 for interior points
        basis_sum = basis.sum(dim=-1)
        interior = (x.squeeze() > -0.5) & (x.squeeze() < 0.5)
        assert torch.allclose(basis_sum[interior], torch.ones_like(basis_sum[interior]), atol=0.1)

    def test_bspline_basis_non_negative(self):
        """Test that B-spline basis functions are non-negative."""
        from src.physics.kan_layers import bspline_basis

        x = torch.randn(32, 4)
        grid = torch.linspace(-2, 2, 12)

        basis = bspline_basis(x, grid, order=3)

        assert (basis >= -1e-6).all(), "B-spline basis should be non-negative"


class TestKANLinear:
    """Tests for KANLinear layer."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        from src.physics.kan_layers import KANLinear

        batch_size = 16
        in_features = 8
        out_features = 4

        layer = KANLinear(in_features, out_features, grid_size=5, spline_order=3)
        x = torch.randn(batch_size, in_features)

        y = layer(x)

        assert y.shape == (batch_size, out_features)

    def test_gradient_flow(self):
        """Test that gradients flow through KAN layer."""
        from src.physics.kan_layers import KANLinear

        layer = KANLinear(4, 2, grid_size=5, spline_order=3)
        x = torch.randn(8, 4, requires_grad=True)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert layer.spline_weight.grad is not None
        assert layer.base_weight.grad is not None

    def test_regularization_loss(self):
        """Test that regularization loss is computed correctly."""
        from src.physics.kan_layers import KANLinear

        layer = KANLinear(4, 2)

        reg_loss = layer.regularization_loss(lambda_l1=0.01)

        assert reg_loss >= 0
        assert reg_loss.requires_grad


class TestKANNetwork:
    """Tests for multi-layer KAN network."""

    def test_network_forward(self):
        """Test multi-layer KAN network forward pass."""
        from src.physics.kan_layers import KANNetwork, KANLayerConfig

        config = KANLayerConfig(grid_size=5, spline_order=3)
        network = KANNetwork([8, 16, 8, 4], config=config, dropout=0.1)

        x = torch.randn(16, 8)
        y = network(x)

        assert y.shape == (16, 4)

    def test_network_residual_connections(self):
        """Test that residual connections work correctly."""
        from src.physics.kan_layers import KANNetwork

        # Same input/output dimension should use identity residual
        network = KANNetwork([8, 8, 8], residual=True)
        x = torch.randn(16, 8)
        y = network(x)

        assert y.shape == (16, 8)


class TestSPIKAN:
    """Tests for Separable PIKAN architecture."""

    def test_spikan_forward(self):
        """Test SPIKAN forward pass with all input modalities."""
        from src.physics.spikan import MarketSPIKAN, SPIKANConfig

        config = SPIKANConfig(
            n_assets=6,
            n_macro=4,
            n_geometric=4,
            hidden_dim=16,
            n_terms=4,
            output_dim=1,
        )
        model = MarketSPIKAN(config)

        batch_size = 8
        assets = torch.randn(batch_size, 6)
        macro = torch.randn(batch_size, 4)
        geometry = torch.randn(batch_size, 4)
        t = torch.randn(batch_size, 1)

        # Without time
        y1 = model(assets, macro, geometry)
        assert y1.shape == (batch_size, 1)

        # With time
        y2 = model(assets, macro, geometry, t)
        assert y2.shape == (batch_size, 1)

    def test_spikan_derivatives(self):
        """Test SPIKAN forward_with_derivatives for physics-informed loss."""
        from src.physics.spikan import MarketSPIKAN, SPIKANConfig

        # Test scalar output case
        config = SPIKANConfig(
            n_assets=4,
            n_macro=2,
            n_geometric=2,
            hidden_dim=8,
            n_terms=2,
            output_dim=1,
        )
        model = MarketSPIKAN(config)

        assets = torch.randn(4, 4)
        macro = torch.randn(4, 2)
        geometry = torch.randn(4, 2)
        t = torch.randn(4, 1)

        derivs = model.forward_with_derivatives(assets, macro, geometry, t)

        assert "u" in derivs
        assert "u_t" in derivs
        assert "u_x" in derivs
        assert "u_xx" in derivs
        assert derivs["u"].shape == (4, 1)
        assert derivs["u_t"].shape == (4, 1)
        # Scalar output: u_x shape is (batch, 1, n_assets)
        assert derivs["u_x"].shape == (4, 1, 4)
        assert derivs["u_xx"].shape == (4, 1, 4)

    def test_spikan_derivatives_multioutput(self):
        """Test SPIKAN forward_with_derivatives for multi-asset output."""
        from src.physics.spikan import MarketSPIKAN, SPIKANConfig

        # Test multi-output case
        config = SPIKANConfig(
            n_assets=4,
            n_macro=2,
            n_geometric=2,
            hidden_dim=8,
            n_terms=2,
            output_dim=4,  # Multi-asset output
        )
        model = MarketSPIKAN(config)

        assets = torch.randn(4, 4)
        macro = torch.randn(4, 2)
        geometry = torch.randn(4, 2)
        t = torch.randn(4, 1)

        derivs = model.forward_with_derivatives(assets, macro, geometry, t)

        assert derivs["u"].shape == (4, 4)
        assert derivs["u_t"].shape == (4, 4)
        # Multi-output: u_x shape is (batch, output_dim, n_assets)
        assert derivs["u_x"].shape == (4, 4, 4)
        assert derivs["u_xx"].shape == (4, 4, 4)

    def test_spikan_separable_structure(self):
        """Test that SPIKAN maintains separable structure."""
        from src.physics.spikan import MarketSPIKAN, SPIKANConfig

        config = SPIKANConfig(n_assets=4, n_macro=2, n_geometric=2)
        model = MarketSPIKAN(config)

        # Check that branches exist
        assert hasattr(model, "asset_branch")
        assert hasattr(model, "macro_branch")
        assert hasattr(model, "geometry_branch")

        # Check n_terms
        assert model.asset_branch.n_terms == config.n_terms


class TestMarketPDEs:
    """Tests for market-specific PDEs."""

    def test_burgers_residual(self):
        """Test Burgers equation residual computation."""
        from src.physics.market_pdes import BurgersEquation, PDEConfig

        config = PDEConfig(viscosity=0.01)
        pde = BurgersEquation(config)

        # Create test tensors
        u = torch.randn(8, 1)
        u_t = torch.randn(8, 1)
        u_x = torch.randn(8, 4)
        u_xx = torch.randn(8, 4)

        residual = pde.residual(u, u_t, u_x, u_xx)

        assert residual.shape[0] == 8
        assert residual.requires_grad or not u.requires_grad

    def test_burgers_shock_indicator(self):
        """Test Burgers shock formation indicator."""
        from src.physics.market_pdes import BurgersEquation

        pde = BurgersEquation()

        u = torch.tensor([[1.0], [2.0], [0.5]])
        u_x = torch.tensor([[0.1], [0.5], [0.01]])

        indicator = pde.shock_formation_indicator(u, u_x, viscosity=0.01)

        # Higher u * u_x should give higher indicator
        assert indicator[1] > indicator[2]

    def test_advection_diffusion_residual(self):
        """Test advection-diffusion residual."""
        from src.physics.market_pdes import AdvectionDiffusion

        pde = AdvectionDiffusion()

        u = torch.randn(8, 1)
        u_t = torch.randn(8, 1)
        u_x = torch.randn(8, 4)
        u_xx = torch.randn(8, 4)
        velocity = torch.randn(8, 4)

        residual = pde.residual(u, u_t, u_x, u_xx, velocity=velocity)

        assert residual.shape[0] == 8

    def test_physics_informed_loss(self):
        """Test combined physics-informed loss function."""
        from src.physics.market_pdes import physics_informed_loss, BurgersEquation, PDEConfig
        from src.physics.spikan import MarketSPIKAN, SPIKANConfig

        # Create small model
        model_config = SPIKANConfig(
            n_assets=4,
            n_macro=2,
            n_geometric=2,
            hidden_dim=8,
            n_terms=2,
        )
        model = MarketSPIKAN(model_config)

        pde_config = PDEConfig(lambda_pde=1.0, lambda_data=1.0)
        pde = BurgersEquation(pde_config)

        # Test data
        assets = torch.randn(4, 4)
        macro = torch.randn(4, 2)
        geometry = torch.randn(4, 2)
        t = torch.randn(4, 1)
        target = torch.randn(4, 1)

        losses = physics_informed_loss(
            model, assets, macro, geometry, t, target, pde, pde_config
        )

        assert "total" in losses
        assert "data" in losses
        assert "pde" in losses
        assert losses["total"] >= 0


class TestShockDetector:
    """Tests for shock detection module."""

    def test_momentum_gradient(self):
        """Test momentum gradient computation."""
        from src.physics.shock_detector import ShockDetector
        import pandas as pd

        detector = ShockDetector()
        momentum = pd.Series(np.cumsum(np.random.randn(100)))

        gradient = detector.compute_momentum_gradient(momentum, window=5)

        assert len(gradient) == len(momentum)
        assert gradient.isna().sum() < len(gradient)  # Some valid values

    def test_shock_formation_index(self):
        """Test shock formation index computation."""
        from src.physics.shock_detector import ShockDetector
        import pandas as pd

        detector = ShockDetector()
        momentum = pd.Series(np.cumsum(np.random.randn(200)))
        volatility = pd.Series(np.abs(np.random.randn(200)) * 0.2 + 0.1)

        sfi = detector.shock_formation_index(momentum, volatility, window=20)

        assert len(sfi) == len(momentum)

    def test_detect_shocks(self):
        """Test full shock detection pipeline."""
        from src.physics.shock_detector import ShockDetector
        import pandas as pd

        detector = ShockDetector()

        # Create test data with an obvious shock
        np.random.seed(42)
        returns = pd.Series(np.random.randn(300) * 0.02)
        returns.iloc[150] = -0.15  # Insert shock
        returns.iloc[151] = -0.10

        momentum = returns.rolling(10).sum()
        volatility = returns.rolling(20).std() * np.sqrt(252)

        results = detector.detect_shocks(momentum, volatility, returns, threshold=2.0)

        assert "shock_indicator" in results.columns
        assert "shock_intensity" in results.columns
        assert "shock_type" in results.columns


class TestFluidDynamics:
    """Tests for fluid dynamics feature integration."""

    def test_momentum_field(self):
        """Test momentum field computation at multiple scales."""
        from src.analysis.fluid_dynamics import compute_momentum_field
        import pandas as pd

        returns = pd.DataFrame({
            "A": np.random.randn(100) * 0.02,
            "B": np.random.randn(100) * 0.02,
        })

        momentum = compute_momentum_field(returns, windows=[5, 10, 20])

        assert "momentum_5d" in momentum.columns
        assert "momentum_10d" in momentum.columns
        assert "momentum_20d" in momentum.columns

    def test_effective_viscosity(self):
        """Test effective viscosity computation."""
        from src.analysis.fluid_dynamics import compute_effective_viscosity
        import pandas as pd

        volatility = pd.Series(np.abs(np.random.randn(300)) * 0.2 + 0.1)
        mst_stress = pd.Series(np.abs(np.random.randn(300)) + 1)

        viscosity = compute_effective_viscosity(
            volatility, mst_stress, base_viscosity=0.01
        )

        assert len(viscosity) == len(volatility)
        # Skip NaN values from rolling mean warmup period
        valid_viscosity = viscosity.dropna()
        assert (valid_viscosity > 0).all()
        assert (valid_viscosity <= 1).all()

    def test_fluid_regime_signals(self):
        """Test fluid dynamics regime signal generation."""
        from src.analysis.fluid_dynamics import fluid_regime_signals
        import pandas as pd

        features = pd.DataFrame({
            "shock_formation_index_z": np.random.randn(100),
            "momentum_decay_rate": np.abs(np.random.randn(100)) * 0.1,
        }, index=pd.date_range("2020-01-01", periods=100))

        signals = fluid_regime_signals(features, shock_threshold=2.0)

        assert len(signals) == 100
        assert set(signals.unique()) <= {"NORMAL", "SHOCK_IMMINENT", "MOMENTUM_PERSISTING", "MOMENTUM_DECAYING"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
