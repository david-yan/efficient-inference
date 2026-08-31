#!/usr/bin/env python3
"""
Unit tests for tau-bench capacity benchmark harness
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

TAU2_SRC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "tau2-bench", "src")
)
if TAU2_SRC_DIR not in sys.path:
    sys.path.insert(0, TAU2_SRC_DIR)

from benchmark_tau_capacity import run_tau_concurrency_tier
from run_tau_comparison import SCENARIO_CONFIGS, format_tau_comparison_table


class TestTauCapacityHarness(unittest.TestCase):

    def test_scenario_configs_defined(self):
        self.assertIn("scenario_1_same_local", SCENARIO_CONFIGS)
        self.assertIn("scenario_2_local_agent_vertex_user", SCENARIO_CONFIGS)
        self.assertIn("scenario_3_different_local", SCENARIO_CONFIGS)

    def test_format_tau_comparison_table(self):
        mock_data = {
            "scenario_1_same_local": {
                "num_tasks": 5,
                "results": [
                    {
                        "concurrency": 1,
                        "total_throughput_tok_s": 250.0,
                        "agent_turn_latency_p50_ms": 120.0,
                        "success_rate_perc": 80.0,
                    }
                ]
            }
        }
        table_output = format_tau_comparison_table(mock_data, "Test Report")
        self.assertIn("MULTI-TURN AGENT CAPACITY COMPARATIVE REPORT", table_output)
        self.assertIn("C=1", table_output)
        self.assertIn("250.0", table_output)


if __name__ == "__main__":
    unittest.main()
