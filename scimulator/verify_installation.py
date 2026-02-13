#!/usr/bin/env python3
"""
Verification script to check if the Synthetic Demand Engine is properly installed.
"""

import sys
from pathlib import Path

def verify_modules():
    """Verify all modules can be imported."""
    print("Verifying Synthetic Demand Generation Engine Installation...")
    print("="*70)

    errors = []

    # Test imports
    modules_to_test = [
        ("models", "synthetic_demand_engine.models"),
        ("orchestrator", "synthetic_demand_engine.orchestrator"),
        ("config.loader", "synthetic_demand_engine.config.loader"),
        ("generators.patterns", "synthetic_demand_engine.generators.patterns"),
        ("generators.noise", "synthetic_demand_engine.generators.noise"),
        ("generators.correlations", "synthetic_demand_engine.generators.correlations"),
        ("utils.validation", "synthetic_demand_engine.utils.validation"),
        ("utils.visualization", "synthetic_demand_engine.utils.visualization"),
        ("cli", "synthetic_demand_engine.cli"),
    ]

    for name, module_path in modules_to_test:
        try:
            __import__(module_path)
            print(f"✓ {name:30s} OK")
        except Exception as e:
            print(f"✗ {name:30s} FAILED: {e}")
            errors.append((name, str(e)))

    print("="*70)

    # Check files exist
    print("\nVerifying file structure...")
    print("="*70)

    required_files = [
        "synthetic_demand_engine/__init__.py",
        "synthetic_demand_engine/models.py",
        "synthetic_demand_engine/orchestrator.py",
        "synthetic_demand_engine/cli.py",
        "synthetic_demand_engine/config/__init__.py",
        "synthetic_demand_engine/config/loader.py",
        "synthetic_demand_engine/generators/__init__.py",
        "synthetic_demand_engine/generators/patterns.py",
        "synthetic_demand_engine/generators/noise.py",
        "synthetic_demand_engine/generators/correlations.py",
        "synthetic_demand_engine/utils/__init__.py",
        "synthetic_demand_engine/utils/validation.py",
        "synthetic_demand_engine/utils/visualization.py",
        "config/example_basic.yaml",
        "config/example_multi.yaml",
        "config/example_complex.yaml",
        "tests/test_engine.py",
        "example.py",
        "README.md",
        "requirements.txt",
        "setup.py",
    ]

    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            errors.append((file_path, "File not found"))

    print("="*70)

    # Summary
    if not errors:
        print("\n✓ ALL CHECKS PASSED!")
        print("\nThe Synthetic Demand Generation Engine is ready to use.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run example: python example.py")
        print("  3. Run tests: pytest tests/test_engine.py -v")
        print("  4. Try the CLI: python -m synthetic_demand_engine.cli config/example_basic.yaml -o output.csv")
        return 0
    else:
        print(f"\n✗ {len(errors)} ERRORS FOUND:")
        for name, error in errors:
            print(f"  - {name}: {error}")
        return 1

if __name__ == "__main__":
    sys.exit(verify_modules())
