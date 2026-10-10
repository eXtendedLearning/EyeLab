# Git hooks

Tracked here so every clone and every agent gets the same rules.

Enable once **per clone** (desktop, Surface, any agent sandbox):

```
git config core.hooksPath tools/githooks
```

| hook | what it does |
|---|---|
| `commit-msg` | requires `0.2.014 message`, where the number equals the bumped version in `python/eyelab_version.py` and has not been used before. See `docs/VERSIONING.md`. |

Needs `python` or `python3` on PATH; if neither exists the hook passes silently.
On Linux/macOS clones the file must be executable (`chmod +x tools/githooks/commit-msg`).
