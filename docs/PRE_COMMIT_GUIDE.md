# Pre-commit Setup Guide

This document explains how to set up and use pre-commit hooks for the LibroBot VLA project.

## What is Pre-commit?

Pre-commit is a framework for managing git pre-commit hooks. It automatically runs checks on your code before you commit, ensuring code quality and consistency.

## Installation

### 1. Install pre-commit

Pre-commit is already included in the dev dependencies. Install it with:

```bash
pip install pre-commit
```

Or install all dev dependencies:

```bash
pip install -e ".[dev]"
```

### 2. Install the git hook scripts

Run this command once in your local repository:

```bash
pre-commit install
```

This will set up the git hooks to run automatically on every commit.

## Usage

### Automatic Checks on Commit

Once installed, pre-commit will automatically run all configured hooks when you try to commit:

```bash
git add .
git commit -m "Your commit message"
```

If any hook fails:
- The commit will be blocked
- Some hooks may auto-fix issues (black, ruff, isort)
- You'll need to review the changes and commit again

### Manual Execution

Run all hooks on all files:

```bash
pre-commit run --all-files
```

Run all hooks on staged files only:

```bash
pre-commit run
```

Run a specific hook:

```bash
pre-commit run black --all-files
pre-commit run ruff --all-files
```

### Skip Hooks (Not Recommended)

If you need to skip pre-commit checks (use sparingly):

```bash
git commit -m "Your message" --no-verify
```

## Configured Hooks

Our pre-commit configuration includes:

### 1. **General File Checks**
- Remove trailing whitespace
- Ensure files end with newline
- Check YAML/JSON/TOML syntax
- Prevent large files (>10MB)
- Check for merge conflicts
- Detect debug statements (pdb, ipdb)
- Prevent commits to main/master branch

### 2. **Python Formatting**
- **Black**: Auto-formats Python code to consistent style
- **isort**: Sorts and organizes imports

### 3. **Python Linting**
- **Ruff**: Fast Python linter with auto-fix
- Checks for common errors, style issues, and best practices

### 4. **Type Checking**
- **mypy**: Static type checker for Python
- Ensures type hints are correct

### 5. **Security**
- **Bandit**: Checks for common security issues
- Scans for vulnerabilities in code

### 6. **Documentation**
- **pydocstyle**: Checks docstring formatting
- Ensures Google-style docstrings

## Updating Hooks

Update all hooks to their latest versions:

```bash
pre-commit autoupdate
```

## Configuration Files

Pre-commit behavior is configured in:
- `.pre-commit-config.yaml` - Hook definitions and versions
- `pyproject.toml` - Tool-specific settings (black, ruff, isort, etc.)

## Troubleshooting

### Hook fails with import errors

Some hooks (like mypy) may fail if they can't import your code. Make sure your package is installed:

```bash
pip install -e .
```

### Want to temporarily disable a hook

Edit `.pre-commit-config.yaml` and comment out the hook you want to disable.

### Pre-commit is slow

First run can be slow as it sets up virtual environments. Subsequent runs are much faster. You can also:

```bash
# Run only on changed files instead of all files
pre-commit run
```

## Best Practices

1. **Install pre-commit early**: Set it up when you first clone the repository
2. **Run before pushing**: Use `pre-commit run --all-files` before pushing
3. **Don't skip checks**: The hooks are there to maintain code quality
4. **Update regularly**: Run `pre-commit autoupdate` monthly
5. **Check CI**: Even with pre-commit, CI should run the same checks

## IDE Integration

### VS Code

Install the "Pre-commit" extension to see hook results in your IDE.

### PyCharm

Enable pre-commit in Settings > Tools > Pre-commit

## Additional Resources

- [Pre-commit Documentation](https://pre-commit.com/)
- [Black Documentation](https://black.readthedocs.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [mypy Documentation](https://mypy.readthedocs.io/)
