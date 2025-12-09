"""
Module: synthetic_generator

Purpose: Generate Bloomreach EBQ-compatible synthetic data for testing.

Key Functions:
- SyntheticDataGenerator: Main generator class
- generate_dataset: Generate complete synthetic dataset
- generate_events: Generate realistic event sequences
- generate_customer_properties: Generate customer attributes
- generate_merge_history: Generate customer ID merge history

Architecture Notes:
- Deterministic with seed for reproducibility
- Generates realistic event sequences (view -> cart -> purchase)
- Includes edge cases (single-event customers, multiple merges)
- Supports multiple dataset sizes (1k, 10k, 100k)
"""

# TODO: Implement synthetic data generator
