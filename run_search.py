#!/usr/bin/env python3
"""Convenience wrapper so the repo works without installing.

Installed users get the `evonas-search` command instead.
"""
from evonas.cli_search import main

if __name__ == "__main__":
    main()
