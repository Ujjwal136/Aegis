## Aegis WeilChain Audit

This project now uses the official Weilliptic Python SDK directly from
the backend. The applet interface file is kept in this folder for
reference, while commits are sent via:

from weil_wallet import PrivateKey, Wallet, WeilClient

## Runtime setup

1. Install Python dependency: weil-wallet
2. Place private_key.wc in project root
3. Optionally set WEIL_KEY_PATH in .env

## Audit message format

AEGIS|{event_type}|{threat_type}|{trace_id}|{session_id}|{layer_used}|{confidence}|{timestamp_utc}|{weilchain_hash}

No raw PII is sent on-chain.
