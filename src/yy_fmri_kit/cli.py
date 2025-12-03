"""Console script for yy_fmri_kit."""

import subprocess
import os
import sys
from argparse import ArgumentParser


def cmd_update(args: list[str] | None = None) -> None:
    """
    Safely update the local git repository by pulling from origin/main.
    Usage: yy-fmri-kit update
    """
    # Check if this is a git repo
    if not os.path.isdir(".git"):
        print("❌ This directory is not a git repository.")
        print("   Run this command inside your cloned yy_fmri_kit folder.")
        sys.exit(1)

    # Show current branch and remotes (best-effort)
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        branch = "main"

    try:
        remotes = subprocess.check_output(["git", "remote", "-v"], text=True)
        print("🔗 Git remotes:\n", remotes)
    except subprocess.CalledProcessError:
        print("⚠️  Could not read git remotes (continuing).")

    # Ask confirmation
    answer = input(f"Pull latest changes from origin/{branch}? (y/n) ").strip().lower()
    if answer != "y":
        print("❌ Update canceled.")
        sys.exit(0)

    # Run git pull
    try:
        subprocess.run(["git", "pull", "origin", branch], check=True)
        print("✅ Repository updated successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to pull updates.")
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    """
    Entry point for the `yy-fmri-kit` CLI.
    """
    parser = ArgumentParser(prog="yy-fmri-kit", description="yy_fmri_kit command-line tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- `update` subcommand ---
    update_parser = subparsers.add_parser(
        "update",
        help="Pull latest changes from the git remote (origin/<current-branch>).",
    )
    update_parser.set_defaults(func=cmd_update)

    args = parser.parse_args(argv)

    # Dispatch to the chosen subcommand
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        sys.exit(1)

    # Strip argparse namespace when passing to handler (we don't need sub-args now)
    func(None)


if __name__ == "__main__":
    main()
