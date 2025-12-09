"""
Module: joiner

Purpose: Customer ID unification with merge chain resolution.

Key Functions:
- resolve_customer_merges: Resolve all merge chains to canonical IDs
- apply_merges_to_events: Apply merge mapping to event records
- MergeMap: Type alias for merge mapping dictionary

Architecture Notes:
- Handles recursive merge chains (A -> B -> C)
- Detects and raises on circular merges
- Critical for preventing double-counting
"""

# TODO: Implement customer ID merge resolution
