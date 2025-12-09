"""
Module: bq_client

Purpose: BigQuery data extraction client for Bloomreach EBQ data.

Key Functions:
- BQClient: Client for loading events and customer data
- load_events: Load event data from BigQuery or synthetic source
- load_customer_properties: Load customer properties
- load_id_history: Load customer ID merge history

Architecture Notes:
- Supports both real BigQuery and synthetic data sources
- Uses mock data source for MVP
- I/O-bound operations use asyncio
"""

# TODO: Implement BigQuery client
