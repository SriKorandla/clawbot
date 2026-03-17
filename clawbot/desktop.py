from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DesktopCapabilities:
    """
    macOS desktop integration helpers.

    These are intentionally minimal and safe:
    - notify: user-visible notification
    - clipboard_set: copy text to clipboard
    - open_target: open a URL or file path using the OS "open" command
    """

    enabled: bool = True

    def _ensure_macos(self) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("Desktop integration is currently supported only on macOS (darwin).")

    def notify(self, title: str, message: str, *, sound: str | None = None) -> str:
        self._ensure_macos()
        if not self.enabled:
            return "Desktop integration disabled."

        # Use terminal-notifier as primary method (more reliable than osascript).
        tn = self._which("terminal-notifier")
        if tn:
            args = [tn, "-title", title, "-message", message]
            if sound:
                args += ["-sound", sound]
            r = subprocess.run(args, capture_output=True, text=True, check=False)
            if r.returncode == 0:
                return "Notification sent (via terminal-notifier)."
            # If terminal-notifier fails, log the error and fall back to osascript
            error_msg = r.stderr.strip() or r.stdout.strip() or f"Exit code {r.returncode}"
            # Fall back to osascript below, but we'll know terminal-notifier failed

        # Fallback to osascript if terminal-notifier is not installed or failed
        if not tn:
            # Provide helpful error if terminal-notifier is missing
            return (
                "Notification failed: terminal-notifier not found. "
                "Install with: brew install terminal-notifier\n"
                "Or check System Settings → Notifications → Terminal/iTerm for osascript notifications."
            )

        # Escape for AppleScript safely by passing as a single -e script with quoted strings.
        # We still defensively replace quotes to avoid breaking the script.
        t = title.replace('"', '\\"')
        m = message.replace('"', '\\"')
        if sound:
            s = sound.replace('"', '\\"')
            script = f'display notification "{m}" with title "{t}" sound name "{s}"'
        else:
            script = f'display notification "{m}" with title "{t}"'
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            error_msg = result.stderr.strip() or "Unknown error"
            return f"Failed to send notification: {error_msg}. Note: macOS may require notification permissions for Terminal/iTerm."
        return "Notification sent (check your notification center if you don't see it)."

    def clipboard_set(self, text: str) -> str:
        self._ensure_macos()
        if not self.enabled:
            return "Desktop integration disabled."

        p = subprocess.Popen(
            ["pbcopy"],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _, stderr = p.communicate(input=text.encode("utf-8"))
        if p.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="replace").strip() or "Unknown error"
            return f"Failed to copy to clipboard: {error_msg}"
        return "Copied to clipboard."

    def open_target(self, target: str) -> str:
        self._ensure_macos()
        if not self.enabled:
            return "Desktop integration disabled."

        # Allow opening either a URL or a local path.
        t = target.strip()
        if not t:
            raise ValueError("target cannot be empty")

        # If it's a path, expand it.
        if "://" not in t and (t.startswith("~") or t.startswith("/") or Path(t).exists()):
            t = str(Path(t).expanduser().absolute())

        subprocess.run(["open", t], check=False)
        return f"Opened: {t}"

    def say(self, text: str, *, voice: str | None = None) -> str:
        """Audible confirmation that desktop integration is working."""
        self._ensure_macos()
        if not self.enabled:
            return "Desktop integration disabled."
        args = ["say"]
        if voice:
            args += ["-v", voice]
        args.append(text)
        r = subprocess.run(args, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            return f"Failed to speak: {(r.stderr or '').strip() or 'Unknown error'}"
        return "Spoken."

    @staticmethod
    def _which(cmd: str) -> str | None:
        from shutil import which

        return which(cmd)
