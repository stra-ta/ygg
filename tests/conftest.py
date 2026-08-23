"""
Shared pytest fixtures for the objectives test suite.

Provides a default :class:`model.config.ModelConfig` and a JAX PRNG key that is
re-derived per test so failures are reproducible but independent.
"""

import jax
import pytest

from model.config import ModelConfig


@pytest.fixture
def model_config() -> ModelConfig:
    """A default model configuration used across objective tests."""
    return ModelConfig()


@pytest.fixture
def key() -> jax.Array:
    """A fresh PRNG key for each test."""
    return jax.random.PRNGKey(0)
