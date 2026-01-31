# GitHub Actions Workflows

CI/CD workflows for LibroBot VLA.

## Quick Reference

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `test.yml` | Push/PR to main | Lint, type check, unit tests |
| `docker.yml` | Push to main, tags | Build & push images to ghcr.io |
| `self-hosted.yml` | After docker.yml | GPU tests on self-hosted runner |
| `release.yml` | Version tags (v*) | PyPI publish, GitHub release |

## Docker Images

Published to GitHub Container Registry:

```
ghcr.io/wfhit/librobot-base    # CUDA 13.0 + PyTorch 2.9
ghcr.io/wfhit/librobot-train   # + Training deps (DeepSpeed, etc)
ghcr.io/wfhit/librobot-deploy  # + Inference (FastAPI)
```

**Tags:** `latest`, `1.0.0`, `1.0`, `<sha>`

---

## ⚙️ Setup Steps

### 1. Repository Actions Permissions

**Settings → Actions → General:**
- ✅ Workflow permissions: **Read and write permissions**
- ✅ Allow GitHub Actions to create and approve pull requests

### 2. Package Visibility (After First Build)

Packages are **private by default**. After first Docker build completes:

1. Go to your profile → **Packages** tab
2. Click each package (`librobot-base`, `librobot-train`, `librobot-deploy`)
3. **Package settings** → **Danger Zone** → **Change visibility** → **Public**

> **Note:** Packages auto-link to the repository via `org.opencontainers.image.source` labels in Dockerfiles, inheriting repo permissions.

### 3. Self-Hosted Runner (for GPU Tests)

On your GPU machine:

```bash
# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Install GitHub Actions Runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.321.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.321.0/actions-runner-linux-x64-2.321.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.321.0.tar.gz

# Configure (get token from Settings → Actions → Runners → New self-hosted runner)
./config.sh --url https://github.com/wfhit/librobot_vla --token YOUR_TOKEN --labels self-hosted,linux,x64

# Run as service
sudo ./svc.sh install && sudo ./svc.sh start
```

---

## Local Development

```bash
# Build images locally
docker build -f docker/Dockerfile.base -t librobot-base:latest .
docker build -f docker/Dockerfile.train -t librobot-train:latest --build-arg BASE_IMAGE=librobot-base:latest .

# Run with GPU
docker run --gpus all -it -v $(pwd):/workspace librobot-train:latest bash
```

---

## Status Badges

```markdown
[![Tests](https://github.com/wfhit/librobot_vla/workflows/Tests/badge.svg)](https://github.com/wfhit/librobot_vla/actions)
[![Docker](https://github.com/wfhit/librobot_vla/workflows/Docker%20Build/badge.svg)](https://github.com/wfhit/librobot_vla/actions)
```
