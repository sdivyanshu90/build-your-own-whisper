# Contributing

Thanks for considering a contribution!

## Workflow

1. Fork and branch from `main` (`feat/...`, `fix/...`, `docs/...`).
2. `make install-dev` — sets up the venv and pre-commit hooks.
3. Make your change **with tests**. Bug fixes need a regression test that fails before
   the fix; features need happy-path *and* failure-mode tests.
4. Before pushing: `make format lint typecheck test`.
5. If you touched the API surface, regenerate the spec: `make openapi` (CI enforces it).
6. Open a PR describing *what* and *why*; link any related issue.

## Ground rules

* Keep the dependency footprint minimal — new runtime dependencies need a strong
  justification in the PR description.
* Public behavior changes require documentation updates in `docs/` and a
  `CHANGELOG.md` entry under *Unreleased*.
* Follow the error-type conventions in [docs/development.md](docs/development.md).
* Model/format changes (checkpoint payload, tokenizer JSON) must bump the respective
  `format_version` and keep a clear failure message for old artifacts.

## Commit style

Conventional-ish: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `ci:`.
Squash-merge is fine; the PR title becomes the changelog line.

## Code of conduct

Be kind, be constructive, assume good faith.
