"""Polymarket L1 wallet authentication.

Wraps py-clob-client initialization with API key derivation.
"""

import structlog
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds

from config import settings

logger = structlog.get_logger(__name__)

# Polymarket CLOB API host
POLYMARKET_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet


def create_clob_client() -> ClobClient:
    """Create and return an authenticated ClobClient instance.

    Returns:
        Authenticated ClobClient ready for API calls.

    Raises:
        ValueError: If required credentials are missing.
    """
    if not settings.polymarket_private_key:
        logger.warning("no_private_key", msg="Running without auth — read-only mode")
        return ClobClient(POLYMARKET_HOST, chain_id=CHAIN_ID)

    client = ClobClient(
        POLYMARKET_HOST,
        key=settings.polymarket_private_key,
        chain_id=CHAIN_ID,
    )

    if settings.polymarket_api_key and settings.polymarket_secret:
        client.set_api_creds(
            ApiCreds(
                api_key=settings.polymarket_api_key,
                api_secret=settings.polymarket_secret,
                api_passphrase=settings.wallet_address,
            )
        )
        logger.info("auth_initialized", mode="api_creds_provided")
    else:
        try:
            api_creds = client.derive_api_key()
            client.set_api_creds(api_creds)
            logger.info("auth_initialized", mode="derived_api_key")
        except Exception as exc:
            logger.error("auth_derivation_failed", error=str(exc))
            raise

    return client
