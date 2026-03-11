"""
Tests for KAN Knowledge Store.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from src.options.kan_store.config import KANStoreConfig
from src.options.kan_store.feature_bridge import (
    REGIME_FEATURE_COLS,
    features_to_tensor,
    normalize_features,
)
from src.options.kan_store.vol_surface_kan import VolSurfaceKAN
from src.options.kan_store.covariance_kan import CovarianceKAN
from src.options.kan_store.transition_kan import TransitionKAN
from src.options.kan_store.store import KANKnowledgeStore


class TestKANStoreConfig:
    """Test KANStoreConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = KANStoreConfig()

        assert config.vol_surface_layers == [10, 32, 32, 16, 1]
        assert config.covariance_layers == [8, 32, 32, 21]
        assert config.transition_layers == [9, 32, 32, 4]
        assert config.grid_size == 8
        assert config.spline_order == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = KANStoreConfig(
            vol_surface_layers=[10, 16, 8, 1],
            grid_size=5,
            spline_order=2,
        )

        assert config.vol_surface_layers == [10, 16, 8, 1]
        assert config.grid_size == 5
        assert config.spline_order == 2


class TestFeatureBridge:
    """Test feature bridge and conversions."""

    def test_features_to_tensor(self):
        """Test converting features dict to tensor."""
        features = {col: float(i) for i, col in enumerate(REGIME_FEATURE_COLS)}
        tensor = features_to_tensor(features)

        assert tensor.shape == (1, 8)
        assert tensor.dtype == torch.float32

    def test_features_to_tensor_missing_key(self):
        """Test error on missing feature."""
        features = {"ricci_mean_core_60d": 0.5}  # Missing other features

        with pytest.raises(KeyError):
            features_to_tensor(features)

    def test_normalize_features(self):
        """Test feature normalization."""
        # Create batch of features
        features = torch.randn(10, 8)

        normalized, mean, std = normalize_features(features)

        # Check shapes
        assert normalized.shape == features.shape
        assert mean.shape == (1, 8)
        assert std.shape == (1, 8)

        # Check mean and std are approximately zero and one
        assert torch.allclose(normalized.mean(dim=0), torch.zeros(8), atol=1e-6)
        assert torch.allclose(normalized.std(dim=0), torch.ones(8), atol=1e-6)


class TestVolSurfaceKAN:
    """Test VolSurfaceKAN network."""

    def test_forward_shapes(self):
        """Test forward pass output shapes."""
        model = VolSurfaceKAN()

        batch_size = 4
        regime = torch.randn(batch_size, 8)
        moneyness = torch.randn(batch_size, 1)
        time = torch.randn(batch_size, 1)

        output = model(regime, moneyness, time)

        assert output.shape == (batch_size, 1)
        assert output.dtype == torch.float32

    def test_vol_surface_output_range(self):
        """Test that vol surface output is in [0, 2]."""
        model = VolSurfaceKAN()
        model.eval()

        regime = torch.zeros(1, 8)
        moneyness = torch.zeros(1, 1)
        time = torch.ones(1, 1) * 0.25

        with torch.no_grad():
            vol = model(regime, moneyness, time)

        assert 0 <= vol.item() <= 2.0

    def test_vol_surface_custom_layers(self):
        """Test VolSurfaceKAN with custom layer sizes."""
        layers = [10, 16, 8, 1]
        model = VolSurfaceKAN(layer_sizes=layers)

        regime = torch.randn(2, 8)
        moneyness = torch.randn(2, 1)
        time = torch.randn(2, 1)

        output = model(regime, moneyness, time)
        assert output.shape == (2, 1)

    def test_vol_surface_regularization_loss(self):
        """Test regularization loss computation."""
        model = VolSurfaceKAN()
        model.train()

        regime = torch.randn(2, 8, requires_grad=True)
        moneyness = torch.randn(2, 1, requires_grad=True)
        time = torch.randn(2, 1, requires_grad=True)

        output = model(regime, moneyness, time)
        reg_loss = model.regularization_loss(lambda_l1=1e-4)

        assert reg_loss.item() >= 0
        assert not torch.isnan(reg_loss)


class TestCovarianceKAN:
    """Test CovarianceKAN network."""

    def test_forward_shapes(self):
        """Test forward pass output shapes."""
        model = CovarianceKAN(n_assets=6)

        batch_size = 4
        regime = torch.randn(batch_size, 8)

        output = model(regime)

        assert output.shape == (batch_size, 6, 6)
        assert output.dtype == torch.float32

    def test_covariance_psd(self):
        """Test that output is positive semidefinite."""
        model = CovarianceKAN(n_assets=3)
        model.eval()

        regime = torch.zeros(1, 8)

        with torch.no_grad():
            cov = model(regime)

        # Check symmetry
        assert torch.allclose(cov, cov.transpose(1, 2), atol=1e-5)

        # Check PSD by eigenvalues
        eigenvalues = torch.linalg.eigvalsh(cov)
        assert (eigenvalues >= -1e-5).all()  # Allow small numerical errors

    def test_covariance_diagonal_positive(self):
        """Test that diagonal elements are positive."""
        model = CovarianceKAN(n_assets=4)
        model.eval()

        regime = torch.randn(5, 8)

        with torch.no_grad():
            cov = model(regime)

        diagonals = torch.diagonal(cov, dim1=1, dim2=2)
        assert (diagonals > 0).all()

    def test_covariance_custom_assets(self):
        """Test CovarianceKAN with custom number of assets."""
        n_assets = 5
        model = CovarianceKAN(n_assets=n_assets)

        regime = torch.randn(2, 8)
        output = model(regime)

        assert output.shape == (2, n_assets, n_assets)

    def test_get_cholesky(self):
        """Test getting Cholesky factor."""
        model = CovarianceKAN(n_assets=3)
        model.eval()

        regime = torch.randn(2, 8)

        with torch.no_grad():
            L = model.get_cholesky(regime)
            cov = model(regime)

        # Verify L @ L.T == cov
        reconstructed_cov = torch.bmm(L, L.transpose(1, 2))
        assert torch.allclose(cov, reconstructed_cov, atol=1e-5)


class TestTransitionKAN:
    """Test TransitionKAN network."""

    def test_forward_shapes(self):
        """Test forward pass output shapes."""
        model = TransitionKAN(n_regimes=4)

        batch_size = 4
        regime = torch.randn(batch_size, 8)
        horizon = torch.randn(batch_size, 1)

        output = model(regime, horizon)

        assert output.shape == (batch_size, 4)
        assert output.dtype == torch.float32

    def test_transition_softmax_properties(self):
        """Test that output is valid probability distribution."""
        model = TransitionKAN(n_regimes=4)
        model.eval()

        regime = torch.zeros(5, 8)
        horizon = torch.ones(5, 1) * 0.1

        with torch.no_grad():
            probs = model(regime, horizon)

        # Check sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(5), atol=1e-6)

        # Check in [0, 1]
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_transition_custom_regimes(self):
        """Test TransitionKAN with custom number of regimes."""
        n_regimes = 6
        model = TransitionKAN(n_regimes=n_regimes)

        regime = torch.randn(2, 8)
        horizon = torch.randn(2, 1)

        output = model(regime, horizon)
        assert output.shape == (2, n_regimes)

    def test_get_logits(self):
        """Test getting raw logits."""
        model = TransitionKAN(n_regimes=4)
        model.eval()

        regime = torch.randn(2, 8)
        horizon = torch.randn(2, 1)

        with torch.no_grad():
            logits = model.get_logits(regime, horizon)
            probs = model(regime, horizon)

        # Verify softmax relationship
        expected_probs = torch.softmax(logits, dim=1)
        assert torch.allclose(probs, expected_probs, atol=1e-5)


class TestKANKnowledgeStore:
    """Test unified KANKnowledgeStore."""

    def test_initialization(self):
        """Test store initialization."""
        store = KANKnowledgeStore()

        assert hasattr(store, "vol_surface")
        assert hasattr(store, "covariance")
        assert hasattr(store, "transition")
        assert hasattr(store, "feature_bridge")

    def test_query_vol_surface(self):
        """Test volume surface query."""
        store = KANKnowledgeStore()
        store.eval()

        regime_features = {col: 0.0 for col in REGIME_FEATURE_COLS}
        vol = store.query_vol_surface(regime_features, log_moneyness=0.0, time_to_expiry=0.25)

        assert isinstance(vol, float)
        assert 0 <= vol <= 2.0

    def test_query_covariance(self):
        """Test covariance query."""
        store = KANKnowledgeStore()
        store.eval()

        regime_features = {col: 0.0 for col in REGIME_FEATURE_COLS}
        cov = store.query_covariance(regime_features)

        assert cov.shape == (6, 6)
        # Check symmetry
        assert torch.allclose(cov, cov.T, atol=1e-5)

    def test_query_transition(self):
        """Test transition query."""
        store = KANKnowledgeStore()
        store.eval()

        regime_features = {col: 0.0 for col in REGIME_FEATURE_COLS}
        probs = store.query_transition(regime_features, horizon=0.1)

        assert probs.shape == (4,)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_save_load(self):
        """Test saving and loading store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            store1 = KANKnowledgeStore()
            store1.eval()

            regime_features = {col: 0.5 for col in REGIME_FEATURE_COLS}

            with torch.no_grad():
                vol1 = store1.query_vol_surface(regime_features, 0.0, 0.25)
                cov1 = store1.query_covariance(regime_features)
                probs1 = store1.query_transition(regime_features, 0.1)

            store1.save(tmpdir)

            # Load and compare
            store2 = KANKnowledgeStore()
            store2.eval()
            store2.load(tmpdir)

            with torch.no_grad():
                vol2 = store2.query_vol_surface(regime_features, 0.0, 0.25)
                cov2 = store2.query_covariance(regime_features)
                probs2 = store2.query_transition(regime_features, 0.1)

            assert abs(vol1 - vol2) < 1e-5
            assert torch.allclose(cov1, cov2, atol=1e-5)
            assert torch.allclose(probs1, probs2, atol=1e-5)

    def test_device_movement(self):
        """Test moving store to different devices."""
        store = KANKnowledgeStore(device="cpu")

        # Move to cpu (should work)
        store.to("cpu")
        assert store.device == "cpu"

        # Check models are on cpu
        for param in store.vol_surface.parameters():
            assert param.device.type == "cpu"
            break

    def test_eval_train_modes(self):
        """Test switching between eval and train modes."""
        store = KANKnowledgeStore()

        store.eval()
        for module in [store.vol_surface, store.covariance, store.transition]:
            assert not module.training

        store.train()
        for module in [store.vol_surface, store.covariance, store.transition]:
            assert module.training

    def test_feature_normalization_fit(self):
        """Test fitting feature normalization."""
        store = KANKnowledgeStore()

        features = torch.randn(20, 8)
        store.fit_normalization(features)

        assert store.feature_bridge.mean is not None
        assert store.feature_bridge.std is not None
        assert store.feature_bridge.mean.shape == (1, 8)

    def test_regularization_loss(self):
        """Test regularization loss computation."""
        store = KANKnowledgeStore()
        store.train()

        reg_loss = store.get_total_regularization_loss(lambda_l1=1e-4)

        assert reg_loss.item() >= 0
        assert not torch.isnan(reg_loss)

    def test_forward_raises_error(self):
        """Test that forward() raises error."""
        store = KANKnowledgeStore()

        with pytest.raises(NotImplementedError):
            store()
