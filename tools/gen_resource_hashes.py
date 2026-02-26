#!/usr/bin/env python3
"""Thin entrypoint for resource hash generation and verification."""

from __future__ import annotations

from spectral_tools.resource_hashes import main


if __name__ == "__main__":
    raise SystemExit(main())
