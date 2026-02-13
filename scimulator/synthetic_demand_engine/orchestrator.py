"""
Main orchestrator for demand generation pipeline.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime

from .models import (
    GenerationConfig, ProductConfig, DemandPattern,
    SeasonalityConfig, TrendConfig, NoiseConfig
)
from .generators.patterns import SeasonalityGenerator, TrendGenerator, BaselineGenerator
from .generators.noise import NoiseGenerator, AnomalyGenerator
from .generators.correlations import CorrelationEngine


class DemandOrchestrator:
    """Orchestrates the entire demand generation pipeline."""

    def __init__(self, config: GenerationConfig):
        """Initialize orchestrator with configuration."""
        self.config = config
        self.seed = config.seed

        # Initialize generators
        self.noise_gen = NoiseGenerator(seed=self.seed)
        self.anomaly_gen = AnomalyGenerator(seed=self.seed + 1 if self.seed else None)
        self.correlation_engine = CorrelationEngine(seed=self.seed + 2 if self.seed else None)

        # Generate time index
        self.timestamps = pd.date_range(
            start=config.start_date,
            end=config.end_date,
            freq=config.frequency
        )
        self.n_timesteps = len(self.timestamps)

    def generate(self) -> Dict[str, DemandPattern]:
        """
        Generate demand patterns for all products.

        Returns:
            Dictionary mapping product_id to DemandPattern
        """
        # Validate configuration
        self._validate_config()

        # Generate individual products
        demands = {}
        for product_config in self.config.products:
            pattern = self._generate_product(product_config)
            demands[product_config.product_id] = pattern

        # Apply cross-product correlations
        if self.config.correlations:
            demands = self._apply_correlations(demands)

        return demands

    def _generate_product(self, config: ProductConfig) -> DemandPattern:
        """Generate demand pattern for a single product."""
        n = self.n_timesteps

        # Initialize components dictionary
        components = {}

        # 1. Baseline
        baseline = BaselineGenerator.generate(n, config.baseline_demand)
        components['baseline'] = baseline.copy()
        demand = baseline.copy()

        # 2. Seasonality
        if config.seasonality:
            seasonality = np.zeros(n)
            for season_config in config.seasonality:
                season_component = SeasonalityGenerator.generate(
                    np.arange(n), season_config
                )
                seasonality += season_component
            components['seasonality'] = seasonality
            demand += seasonality

        # 3. Trend
        if config.trend:
            trend = TrendGenerator.generate(np.arange(n), config.trend)
            components['trend'] = trend
            demand += trend

        # 4. Noise
        if config.noise:
            noise = self.noise_gen.generate(n, config.noise)
            components['noise'] = noise
            demand += noise

        # 5. Anomalies
        anomaly_mask = None
        if config.anomalies:
            demand, anomaly_mask = self.anomaly_gen.generate_multiple(
                demand, config.anomalies
            )

        # 6. Apply constraints
        demand = np.clip(demand, config.min_demand, config.max_demand)

        # Create pattern object
        pattern = DemandPattern(
            product_id=config.product_id,
            timestamps=self.timestamps.values,
            values=demand,
            components=components,
            anomaly_mask=anomaly_mask,
            metadata=config.metadata.copy()
        )

        return pattern

    def _apply_correlations(self, demands: Dict[str, DemandPattern]) -> Dict[str, DemandPattern]:
        """Apply cross-product correlations."""
        # Extract demand arrays
        demand_arrays = {pid: pattern.values for pid, pattern in demands.items()}

        # Apply correlations
        correlated = self.correlation_engine.apply_correlations(
            demand_arrays, self.config.correlations
        )

        # Update patterns with correlated values
        result = {}
        for pid, pattern in demands.items():
            # Create new pattern with correlated values
            new_pattern = DemandPattern(
                product_id=pattern.product_id,
                timestamps=pattern.timestamps,
                values=correlated[pid],
                components=pattern.components,
                anomaly_mask=pattern.anomaly_mask,
                metadata=pattern.metadata
            )
            result[pid] = new_pattern

        return result

    def _validate_config(self):
        """Validate configuration before generation."""
        if self.n_timesteps == 0:
            raise ValueError("Invalid date range: no timestamps generated")

        # Validate correlations
        if self.config.correlations:
            product_ids = [p.product_id for p in self.config.products]
            errors = self.correlation_engine.validate_correlations(
                self.config.correlations, product_ids
            )
            if errors:
                raise ValueError(f"Correlation validation failed: {errors}")

    def export(self, demands: Dict[str, DemandPattern], output_path: str):
        """Export generated demands to file."""
        if self.config.output_format == "csv":
            self._export_csv(demands, output_path)
        elif self.config.output_format == "parquet":
            self._export_parquet(demands, output_path)
        elif self.config.output_format == "json":
            self._export_json(demands, output_path)
        else:
            raise ValueError(f"Unsupported output format: {self.config.output_format}")

    def _export_csv(self, demands: Dict[str, DemandPattern], output_path: str):
        """Export to CSV format."""
        # Create DataFrame
        data = {'timestamp': self.timestamps}
        for pid, pattern in demands.items():
            data[f'{pid}_demand'] = pattern.values
            if pattern.anomaly_mask is not None:
                data[f'{pid}_anomaly'] = pattern.anomaly_mask.astype(int)

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

    def _export_parquet(self, demands: Dict[str, DemandPattern], output_path: str):
        """Export to Parquet format."""
        data = {'timestamp': self.timestamps}
        for pid, pattern in demands.items():
            data[f'{pid}_demand'] = pattern.values
            if pattern.anomaly_mask is not None:
                data[f'{pid}_anomaly'] = pattern.anomaly_mask.astype(int)

        df = pd.DataFrame(data)
        df.to_parquet(output_path, index=False)

    def _export_json(self, demands: Dict[str, DemandPattern], output_path: str):
        """Export to JSON format."""
        import json

        output = {
            'metadata': self.config.metadata,
            'products': {pid: pattern.to_dict() for pid, pattern in demands.items()}
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
