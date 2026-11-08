#!/bin/bash
# Double-click to check that AIClear works on this Mac. Writes selftest_report.txt.
cd "$(dirname "$0")" || exit 1
xattr -dr com.apple.quarantine . 2>/dev/null
./runtime/bin/python3.12 -X utf8 selftest.py
echo
read -n 1 -s -r -p "按任意键关闭..."
echo
