# OpenSpec changes — local validation & CI

This file explains how to author and validate OpenSpec changes locally and suggests a CI workflow for automated validation.

## Local validation (Windows)

- Preferred (cmd.exe):

```powershell
cmd /c "openspec validate <change-id> --strict"
```

- If you are in PowerShell and you see a PSSecurityException referencing `openspec.ps1`, either call via `cmd /c` as above or change the execution policy for the current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
```

## Authoring checklist

- Pick a verb-led change id: `add-foo`, `update-bar`
- Create `openspec/changes/<change-id>/proposal.md` with Why, What Changes, Impact, Owners, Acceptance Criteria
- Create `openspec/changes/<change-id>/tasks.md` with stepwise implementation and tests
- Add spec deltas under `openspec/changes/<change-id>/specs/<capability>/spec.md` using `## ADDED|MODIFIED|REMOVED` headers and at least one `#### Scenario:` per requirement
- Run local validation and fix any formatting or normative-language issues

## Example GitHub Actions workflow snippet

Place a workflow in `.github/workflows/openspec-validate.yml` to validate changes on PRs.

```yaml
name: OpenSpec validate
on:
  pull_request:
    paths:
      - 'openspec/**'
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: '18'
      - name: Install OpenSpec
        run: npm install -g @fission-ai/openspec@latest
      - name: Validate changes
        run: |
          # find top-level change directories modified in the PR
          CHANGES=$(git diff --name-only ${{ github.event.before }} ${{ github.sha }} | grep '^openspec/changes/' | cut -d'/' -f3 | sort -u)
          for c in $CHANGES; do
            if [ -n "$c" ]; then
              openspec validate "$c" --strict || exit 1
            fi
          done
```

Notes:
- The workflow above runs on Linux and uses the npm-installed `openspec` executable directly. On Windows runners you may need to call the `.cmd` shims.
- If your CI environment blocks global npm installs, add a step to install via `npx` or a local node_modules install.

---

If you'd like, I can scaffold the `.github/workflows/openspec-validate.yml` file in this repo and add a small example change to exercise the CI.
