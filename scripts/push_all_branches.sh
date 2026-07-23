#!/usr/bin/env bash
# Push EVERY branch from a bundle-clone to a GitHub repo.
#
# Why this exists: `git clone insightdocs.bundle` only creates ONE local branch
# (the checked-out one); the rest arrive as remote-tracking refs. So a plain
# `git push --all` pushes just that one branch — which is why only `release`
# showed up on GitHub. This script pushes them all.
#
# Usage (run inside the folder you cloned from the bundle):
#   bash scripts/push_all_branches.sh https://github.com/<you>/<repo>.git
set -e
REPO_URL="${1:?Usage: push_all_branches.sh <github-repo-url>}"

git remote remove gh 2>/dev/null || true
git remote add gh "$REPO_URL"
# drop origin/HEAD so it is not mistaken for a branch named HEAD
git remote set-head origin --delete 2>/dev/null || true

# push each remote-tracking branch from the bundle to a real branch on GitHub
git push gh "refs/remotes/origin/*:refs/heads/*"

echo
echo "Done. All branches pushed to $REPO_URL"
git ls-remote --heads gh | awk '{print "  "$2}'
