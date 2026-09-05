#!/usr/bin/env python3
"""Persistent, resumable three-prime endpoint-minor campaign."""
from run_residual_hafnian_gpu import Runner
import reduce_six_by_thirty_optimized as campaign

if __name__ == "__main__":
    raise SystemExit(Runner(campaign, 30, "six_by_thirty_hafnian_gpu").main())
