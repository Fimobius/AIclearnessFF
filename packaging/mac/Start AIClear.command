#!/bin/bash
# Double-click to start AIClear (macOS). Closing this Terminal window stops it.
cd "$(dirname "$0")" || exit 1

# macOS quarantines downloaded files and would block the bundled Python and every library
# one by one. The user has just approved opening this script, so clear the flag on the rest
# of the app folder (only this folder, nothing else on the system).
xattr -dr com.apple.quarantine . 2>/dev/null

./runtime/bin/python3.12 -X utf8 launcher.py
status=$?
if [ $status -ne 0 ]; then
  echo
  read -n 1 -s -r -p "程序异常退出，请截图上面的信息。按任意键关闭..."
  echo
fi
exit $status
