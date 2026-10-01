#!/bin/bash
# Double-click to start the Padel Analyzer (macOS). First run sets everything up.
cd "$(dirname "$0")" || exit 1
exec bash ./start-linux.sh
