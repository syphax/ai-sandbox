"""
Configuration loader for YAML files.
"""

import yaml
from pathlib import Path
from typing import Union, Dict, Any
from ..models import (
    GenerationConfig, ProductConfig, CorrelationConfig,
    SeasonalityConfig, TrendConfig, NoiseConfig, AnomalyConfig,
    SeasonalityType, TrendType, NoiseType, AnomalyType
)


class ConfigLoader:
    """Loads and parses YAML configuration files."""

    @staticmethod
    def load(path: Union[str, Path]) -> GenerationConfig:
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return ConfigLoader.from_dict(data)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> GenerationConfig:
        """Parse configuration from dictionary."""
        # Parse products
        products = [
            ConfigLoader._parse_product(p) for p in data.get('products', [])
        ]

        # Parse correlations
        correlations = [
            ConfigLoader._parse_correlation(c) for c in data.get('correlations', [])
        ]

        return GenerationConfig(
            start_date=data['start_date'],
            end_date=data['end_date'],
            frequency=data.get('frequency', 'H'),
            products=products,
            correlations=correlations,
            seed=data.get('seed'),
            output_format=data.get('output_format', 'csv'),
            metadata=data.get('metadata', {})
        )

    @staticmethod
    def _parse_product(data: Dict[str, Any]) -> ProductConfig:
        """Parse product configuration."""
        seasonality = [
            ConfigLoader._parse_seasonality(s) for s in data.get('seasonality', [])
        ]

        trend = None
        if 'trend' in data:
            trend = ConfigLoader._parse_trend(data['trend'])

        noise = None
        if 'noise' in data:
            noise = ConfigLoader._parse_noise(data['noise'])

        anomalies = [
            ConfigLoader._parse_anomaly(a) for a in data.get('anomalies', [])
        ]

        return ProductConfig(
            product_id=data['product_id'],
            baseline_demand=data['baseline_demand'],
            seasonality=seasonality,
            trend=trend,
            noise=noise,
            anomalies=anomalies,
            min_demand=data.get('min_demand', 0.0),
            max_demand=data.get('max_demand'),
            metadata=data.get('metadata', {})
        )

    @staticmethod
    def _parse_seasonality(data: Dict[str, Any]) -> SeasonalityConfig:
        """Parse seasonality configuration."""
        return SeasonalityConfig(
            type=SeasonalityType(data['type']),
            amplitude=data['amplitude'],
            period=data.get('period'),
            phase_shift=data.get('phase_shift', 0.0),
            harmonics=data.get('harmonics', 1),
            enabled=data.get('enabled', True)
        )

    @staticmethod
    def _parse_trend(data: Dict[str, Any]) -> TrendConfig:
        """Parse trend configuration."""
        return TrendConfig(
            type=TrendType(data['type']),
            coefficient=data.get('coefficient', 0.0),
            exponent=data.get('exponent', 1.0),
            change_points=data.get('change_points', []),
            enabled=data.get('enabled', True)
        )

    @staticmethod
    def _parse_noise(data: Dict[str, Any]) -> NoiseConfig:
        """Parse noise configuration."""
        return NoiseConfig(
            type=NoiseType(data['type']),
            std_dev=data.get('std_dev'),
            mean=data.get('mean', 0.0),
            min_value=data.get('min_value'),
            max_value=data.get('max_value'),
            lambda_param=data.get('lambda_param'),
            sigma=data.get('sigma'),
            enabled=data.get('enabled', True)
        )

    @staticmethod
    def _parse_anomaly(data: Dict[str, Any]) -> AnomalyConfig:
        """Parse anomaly configuration."""
        return AnomalyConfig(
            type=AnomalyType(data['type']),
            probability=data['probability'],
            magnitude=data['magnitude'],
            duration=data.get('duration', 1),
            locations=data.get('locations'),
            enabled=data.get('enabled', True)
        )

    @staticmethod
    def _parse_correlation(data: Dict[str, Any]) -> CorrelationConfig:
        """Parse correlation configuration."""
        return CorrelationConfig(
            source_product=data['source_product'],
            target_product=data['target_product'],
            coefficient=data['coefficient'],
            lag=data.get('lag', 0),
            type=data.get('type', 'linear'),
            enabled=data.get('enabled', True)
        )
