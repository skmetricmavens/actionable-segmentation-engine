#!/usr/bin/env python3
"""
Script: train_clv_model.py

Purpose: Train a CLV prediction model from customer profile data.

Usage:
    python scripts/train_clv_model.py --input profiles.json --output models/clv_model.joblib
    python scripts/train_clv_model.py --input profiles.json --model-type random_forest --cv-splits 5

This script loads customer profiles, trains a CLV prediction model, and saves
the trained model for later use.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.schemas import CustomerProfile
from src.features.clv.training import CLVTrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_profiles_from_json(path: Path) -> list[CustomerProfile]:
    """Load customer profiles from JSON file.

    Args:
        path: Path to JSON file containing profile data

    Returns:
        List of CustomerProfile objects
    """
    logger.info(f"Loading profiles from {path}")

    with open(path) as f:
        data = json.load(f)

    # Handle both list of profiles and dict with "profiles" key
    if isinstance(data, dict) and "profiles" in data:
        profiles_data = data["profiles"]
    elif isinstance(data, list):
        profiles_data = data
    else:
        raise ValueError("Expected JSON array or object with 'profiles' key")

    profiles = [CustomerProfile(**p) for p in profiles_data]
    logger.info(f"Loaded {len(profiles)} profiles")

    return profiles


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train a CLV prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default settings
    python scripts/train_clv_model.py --input profiles.json --output model.joblib

    # Train with random forest and 5-fold CV
    python scripts/train_clv_model.py --input profiles.json --output model.joblib \\
        --model-type random_forest --cv-splits 5

    # Train with holdout evaluation
    python scripts/train_clv_model.py --input profiles.json --output model.joblib \\
        --holdout 0.2
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to input JSON file with customer profiles",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path to save trained model (.joblib)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["gradient_boosting", "random_forest", "ridge"],
        default="gradient_boosting",
        help="Model type to train (default: gradient_boosting)",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation splits (default: 5)",
    )
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.0,
        help="Fraction of data to hold out for evaluation (default: 0, no holdout)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Model version string (default: auto-generated)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate inputs
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Load profiles
    try:
        profiles = load_profiles_from_json(args.input)
    except Exception as e:
        logger.error(f"Failed to load profiles: {e}")
        return 1

    if len(profiles) < 10:
        logger.error(f"Need at least 10 profiles for training, got {len(profiles)}")
        return 1

    # Configure training
    config = TrainingConfig(
        model_type=args.model_type,  # type: ignore
        n_splits=args.cv_splits,
    )

    trainer = CLVTrainer(config)

    # Train model
    logger.info(f"Training {args.model_type} model with {args.cv_splits}-fold CV")

    if args.holdout > 0:
        logger.info(f"Using {args.holdout:.0%} holdout for evaluation")
        result, holdout_metrics = trainer.train_with_holdout(
            profiles, holdout_fraction=args.holdout
        )
        logger.info(f"Holdout Metrics:")
        logger.info(f"  MAE:  {holdout_metrics['mae']:.2f}")
        logger.info(f"  RMSE: {holdout_metrics['rmse']:.2f}")
        logger.info(f"  R2:   {holdout_metrics['r2']:.4f}")
        logger.info(f"  MAPE: {holdout_metrics['mape']:.2f}%")
    else:
        result = trainer.train(profiles)

    # Log results
    logger.info(f"Training completed in {result.training_time:.2f}s")
    logger.info(f"CV Scores (MAE): {', '.join(f'{s:.2f}' for s in result.cv_scores)}")
    logger.info(f"Mean CV MAE: {sum(result.cv_scores) / len(result.cv_scores):.2f}")
    logger.info(f"Final Metrics:")
    logger.info(f"  MAE:  {result.final_metrics['mae']:.2f}")
    logger.info(f"  RMSE: {result.final_metrics['rmse']:.2f}")
    logger.info(f"  R2:   {result.final_metrics['r2']:.4f}")

    # Log top features
    sorted_features = sorted(
        result.feature_importances.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    logger.info("Top 10 Features:")
    for name, importance in sorted_features[:10]:
        logger.info(f"  {name}: {importance:.4f}")

    # Generate version if not provided
    version = args.version or datetime.now().strftime("v%Y%m%d_%H%M%S")

    # Save model
    try:
        output_path = trainer.save_model(result, args.output, version=version)
        logger.info(f"Model saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
