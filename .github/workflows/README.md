# GitHub Actions Workflows

This directory contains the CI/CD workflows for the LibroBot VLA project.

## Workflows Overview

### 🧪 test.yml - Continuous Integration Tests

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

**Status Badge:**
```markdown
![Tests](https://github.com/wfhit/librobot_vla/workflows/Tests/badge.svg)
```

**Jobs:**

1. **Lint and Format Check**
   - Runs Black code formatter (check mode)
   - Runs Ruff linter
   - Python 3.12 on Ubuntu

2. **Type Checking**
   - Runs mypy type checker
   - Python 3.12 on Ubuntu

3. **Tests (Matrix)**
   - Python version: 3.12
   - Runs unit tests (`tests/unit/`)
   - Runs integration tests (`tests/integration/`)
   - Generates coverage reports
   - Uploads coverage to Codecov

4. **Minimal Install Test**
   - Tests basic package installation without optional dependencies
   - Verifies core imports work correctly

**Features:**
- Dependency caching for faster builds
- Coverage reports with HTML output
- Parallel testing across Python versions

---

### 🐳 docker.yml - Docker Image Build and Push

**Triggers:**
- Push to `main` branch
- Version tags (`v*`)
- Pull requests to `main` (build only, no push)

**Status Badge:**
```markdown
![Docker Build](https://github.com/wfhit/librobot_vla/workflows/Docker%20Build/badge.svg)
```

**Jobs:**

1. **Build Base Image**
   - Builds from `docker/Dockerfile.base`
   - CUDA 13.0 + PyTorch 2.9 foundation

2. **Build Train Image**
   - Builds from `docker/Dockerfile.train`
   - Includes training dependencies (DeepSpeed, Accelerate, Transformers)
   - Depends on base image

3. **Build Deploy Image**
   - Builds from `docker/Dockerfile.deploy`
   - Includes inference dependencies (FastAPI, ONNX Runtime)
   - Depends on base image

4. **Scan Images**
   - Runs Trivy security scanner
   - Checks for CRITICAL and HIGH severity vulnerabilities
   - Uploads results to GitHub Security tab

**Image Registries:**
- GitHub Container Registry (GHCR): `ghcr.io/<owner>/librobot-<type>:<tag>`
- Docker Hub (optional): `<username>/librobot-<type>:<tag>`

**Image Tags:**
- `latest` - Latest main branch build
- `<version>` - Semver version (e.g., `1.0.0`)
- `<major>.<minor>` - Minor version (e.g., `1.0`)
- `<major>` - Major version (e.g., `1`)
- `<branch>-<sha>` - Branch + commit SHA

**Features:**
- Multi-stage builds with layer caching
- Parallel building of train/deploy images
- Automatic tagging and versioning
- Security vulnerability scanning
- Support for both GHCR and Docker Hub

---

### 🚀 release.yml - Release and Publish

**Triggers:**
- Version tags matching `v*` (e.g., `v1.0.0`, `v0.1.0-rc1`)

**Status Badge:**
```markdown
![Release](https://github.com/wfhit/librobot_vla/workflows/Release/badge.svg)
```

**Jobs:**

1. **Run Tests Before Release**
   - Full test suite on Python 3.12
   - Linting and type checking
   - Must pass before release proceeds

2. **Build Distribution**
   - Verifies version in code matches git tag
   - Builds wheel and source distribution
   - Validates with `twine check`

3. **Publish to PyPI**
   - Production releases (non-RC tags)
   - Uses trusted publishing (OIDC)
   - Published to https://pypi.org/project/librobot/

4. **Publish to TestPyPI** (Optional)
   - RC/alpha/beta releases only
   - Published to https://test.pypi.org/project/librobot/

5. **Build and Push Docker Images**
   - Builds and tags all three images (base, train, deploy)
   - Tags with version numbers and `latest`
   - Pushes to GHCR and optionally Docker Hub

6. **Create GitHub Release**
   - Generates changelog from git commits
   - Attaches wheel and source distributions
   - Links to Docker images and PyPI
   - Marks as prerelease for RC/alpha/beta tags

**Features:**
- Automated version verification
- Changelog generation from commits
- Comprehensive release notes with installation instructions
- Trusted publishing to PyPI (no tokens needed)
- Parallel Docker image builds
- Draft/prerelease support

---

## Setup Instructions

### Required Secrets

For full functionality, configure these secrets in your repository settings:

#### PyPI Publishing (Recommended: Use Trusted Publishing)

**Option 1: Trusted Publishing (Recommended)**
1. Go to https://pypi.org/manage/account/publishing/
2. Add a new publisher:
   - PyPI Project: `librobot`
   - Owner: Your GitHub username/org
   - Repository: `librobot_vla`
   - Workflow: `release.yml`
   - Environment: `pypi`

No secrets needed! GitHub Actions will authenticate automatically.

**Option 2: API Token (Legacy)**
- `PYPI_TOKEN` - PyPI API token
- `TESTPYPI_TOKEN` - TestPyPI API token

#### Docker Hub (Optional)
- `DOCKERHUB_USERNAME` - Docker Hub username
- `DOCKERHUB_TOKEN` - Docker Hub access token

> **Note:** If Docker Hub secrets are not set, images will only be pushed to GitHub Container Registry (GHCR).

#### Code Coverage (Optional)
- `CODECOV_TOKEN` - Codecov token for coverage reports

### Required GitHub Settings

1. **Enable GitHub Container Registry**
   - Settings → Packages → Make package public (if desired)

2. **Create PyPI Environment** (for trusted publishing)
   - Settings → Environments → New environment → `pypi`
   - Add protection rules if desired

3. **Enable Actions Permissions**
   - Settings → Actions → General
   - Workflow permissions: Read and write permissions
   - Allow GitHub Actions to create and approve pull requests

### Status Badges

Add these badges to your README:

```markdown
[![Tests](https://github.com/wfhit/librobot_vla/workflows/Tests/badge.svg)](https://github.com/wfhit/librobot_vla/actions/workflows/test.yml)
[![Docker Build](https://github.com/wfhit/librobot_vla/workflows/Docker%20Build/badge.svg)](https://github.com/wfhit/librobot_vla/actions/workflows/docker.yml)
[![Release](https://github.com/wfhit/librobot_vla/workflows/Release/badge.svg)](https://github.com/wfhit/librobot_vla/actions/workflows/release.yml)
[![codecov](https://codecov.io/gh/wfhit/librobot_vla/branch/main/graph/badge.svg)](https://codecov.io/gh/wfhit/librobot_vla)
[![PyPI version](https://badge.fury.io/py/librobot.svg)](https://badge.fury.io/py/librobot)
```

---

## Usage Examples

### Running Tests Locally

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
make test

# Run specific test types
make test-unit
make test-integration

# Run linting
make lint

# Run type checking
make type-check
```

### Building Docker Images Locally

```bash
# Build all images
make docker-build

# Or build individually
docker build -f docker/Dockerfile.base -t librobot-base:local .
docker build -f docker/Dockerfile.train -t librobot-train:local .
docker build -f docker/Dockerfile.deploy -t librobot-deploy:local .
```

### Creating a Release

1. **Update version** in `librobot/version.py`:
   ```python
   __version__ = "1.0.0"
   ```

2. **Commit and push**:
   ```bash
   git add librobot/version.py
   git commit -m "Bump version to 1.0.0"
   git push origin main
   ```

3. **Create and push tag**:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

4. The release workflow will automatically:
   - Run tests
   - Build distributions
   - Publish to PyPI
   - Build and push Docker images
   - Create GitHub release with changelog

---

## Workflow Customization

### Modifying Python Versions

Edit the matrix in `test.yml` and `release.yml`:

```yaml
strategy:
  matrix:
    python-version: ["3.9", "3.10", "3.11", "3.12"]  # Add more versions
```

### Adding Additional Checks

Add new jobs or steps to `test.yml`:

```yaml
jobs:
  security-check:
    name: Security Audit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install safety
      - run: safety check
```

### Changing Docker Registries

Modify the `docker.yml` registry settings:

```yaml
env:
  REGISTRY: docker.io  # Use Docker Hub instead
  IMAGE_PREFIX: myorg/myproject
```

---

## Troubleshooting

### Test Failures

1. **Import errors**: Install dev dependencies with `pip install -e ".[dev]"`
2. **Coverage issues**: Ensure `pytest-cov` is installed
3. **Timeout errors**: Increase timeout in `pyproject.toml`

### Docker Build Failures

1. **Out of memory**: Use smaller base images or add memory limits
2. **Authentication errors**: Check GITHUB_TOKEN permissions
3. **Layer cache issues**: Clear with `docker builder prune`

### Release Failures

1. **Version mismatch**: Ensure version in `librobot/version.py` matches the git tag
2. **PyPI authentication**: Verify trusted publishing is set up correctly
3. **Test failures**: Fix tests before creating the release tag

---

## Best Practices

1. **Always run tests locally** before pushing
2. **Use feature branches** and create PRs for code changes
3. **Version bumps** should be in their own commit
4. **Tag format**: Use semantic versioning (`vMAJOR.MINOR.PATCH`)
5. **Release candidates**: Use `-rc1`, `-rc2` suffixes for pre-releases
6. **Security**: Never commit secrets; use GitHub Secrets or trusted publishing
7. **Review**: Check workflow runs in the Actions tab after pushing

---

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [Docker Build Push Action](https://github.com/docker/build-push-action)
- [Codecov Documentation](https://docs.codecov.com/)
- [Semantic Versioning](https://semver.org/)
