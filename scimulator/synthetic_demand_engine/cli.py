"""
Command-line interface for synthetic demand generation.
"""

import argparse
import sys
from pathlib import Path

from .config.loader import ConfigLoader
from .orchestrator import DemandOrchestrator
from .utils.validation import DemandValidator
from .utils.visualization import DemandVisualizer


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Synthetic Demand Generation Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output file path'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate generated patterns'
    )

    parser.add_argument(
        '--plot',
        type=str,
        help='Save plots to this path'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    try:
        # Load configuration
        if args.verbose:
            print(f"Loading configuration from {args.config}...")

        config = ConfigLoader.load(args.config)

        # Generate demands
        if args.verbose:
            print(f"Generating demand patterns for {len(config.products)} products...")

        orchestrator = DemandOrchestrator(config)
        demands = orchestrator.generate()

        if args.verbose:
            print(f"Generated {len(demands)} demand patterns")

        # Validate
        if args.validate:
            if args.verbose:
                print("Validating patterns...")

            validation_results = DemandValidator.validate_all(demands)

            all_valid = True
            for pid, (is_valid, errors) in validation_results.items():
                if not is_valid:
                    all_valid = False
                    print(f"Validation failed for {pid}:")
                    for error in errors:
                        print(f"  - {error}")

            if all_valid and args.verbose:
                print("All patterns validated successfully")

        # Export
        if args.verbose:
            print(f"Exporting to {args.output}...")

        orchestrator.export(demands, args.output)

        if args.verbose:
            print("Export complete")

        # Plot
        if args.plot:
            if args.verbose:
                print(f"Generating plots to {args.plot}...")

            import matplotlib
            matplotlib.use('Agg')

            fig = DemandVisualizer.plot_multiple(demands)
            fig.savefig(args.plot, dpi=150, bbox_inches='tight')

            if args.verbose:
                print("Plots saved")

        print("Success!")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
