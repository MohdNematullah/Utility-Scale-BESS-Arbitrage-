"""
backtesting/check_cli.py
========================

Verification script for Unified CLI Engine (Part 8.5.5).
"""

import subprocess
import sys

LINE = "=" * 75

print(LINE)
print(" COMMAND-LINE INTERFACE (CLI) VERIFICATION")
print(LINE)

commands_to_test = [
    [sys.executable, "-m", "backtesting.cli", "--help"],
    [sys.executable, "-m", "backtesting.cli", "list"],
    [sys.executable, "-m", "backtesting.cli", "list", "--category", "Battery_Chemistry"],
    [sys.executable, "-m", "backtesting.cli", "compare", "--help"],
    [sys.executable, "-m", "backtesting.cli", "export-dashboard", "--help"],
]

for cmd in commands_to_test:
    cmd_str = " ".join(cmd[1:])
    print(f"Testing Command: {cmd_str}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error executing command: {res.stderr}")
        sys.exit(1)
    print("  • Command executed successfully ✓")

print(LINE)
print("CLI parser, subcommands, and dispatching verified successfully ✓")
print(LINE)