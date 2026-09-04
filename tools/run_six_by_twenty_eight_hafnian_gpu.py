#!/usr/bin/env python3
"""Persistent, resumable multi-GPU 6x28 campaign."""
from run_residual_hafnian_gpu import Runner
import reduce_six_by_twenty_eight_hafnian as campaign

if __name__ == "__main__":
    raise SystemExit(Runner(campaign, 28, "six_by_twenty_eight_hafnian_gpu").main())
