#!/usr/bin/env python3
"""Convenience wrapper so the repo works without installing.

Installed users get the `evonas-design` command instead.
"""
from evonas.cli_design import main

if __name__ == "__main__":
    main()
