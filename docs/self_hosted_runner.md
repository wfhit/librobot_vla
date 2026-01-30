# Setting Up a Self-Hosted GitHub Actions Runner

This guide explains how to integrate your own computer with GitHub CI to run tests and workflows on your local machine or custom hardware.

## Why Use a Self-Hosted Runner?

Self-hosted runners are useful when you need:
- **GPU testing**: Run tests that require GPU hardware (CUDA, ROCm, etc.)
- **Custom hardware**: Test on specific robot hardware or embedded devices
- **Large resources**: More RAM, CPU, or disk space than GitHub-hosted runners
- **Network access**: Access to internal networks or services
- **Cost savings**: For heavy CI usage, self-hosted runners can be more economical

## Step 1: Prepare Your Computer

### System Requirements

- Linux, Windows, or macOS
- At least 2 GB RAM (4+ GB recommended)
- Internet connectivity
- Git installed

### For GPU Testing (LibroBot VLA)

If you're setting up a runner for GPU-accelerated tests:

```bash
# Install NVIDIA drivers and CUDA toolkit
# For Ubuntu:
sudo apt update
sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit

# Verify GPU is available
nvidia-smi
```

## Step 2: Add a Self-Hosted Runner on GitHub

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Actions** → **Runners**
3. Click **New self-hosted runner**
4. Select your operating system (Linux/Windows/macOS)
5. Follow the instructions displayed, which include:

### Linux Setup

```bash
# Create a folder for the runner
mkdir actions-runner && cd actions-runner

# Download the latest runner package
curl -o actions-runner-linux-x64-2.313.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.313.0/actions-runner-linux-x64-2.313.0.tar.gz

# Extract the installer
tar xzf ./actions-runner-linux-x64-2.313.0.tar.gz

# Configure the runner
./config.sh --url https://github.com/wfhit/librobot_vla --token YOUR_TOKEN_HERE

# Run the runner
./run.sh
```

### Configure Runner Labels

During configuration, you'll be asked to add labels. Use labels to identify your runner's capabilities:

- `self-hosted` (default)
- `linux`, `windows`, or `macos` (operating system)
- `gpu` (if GPU is available)
- `cuda` (if CUDA is installed)
- `robot-hardware` (if connected to physical robot)

Example:
```bash
./config.sh --url https://github.com/wfhit/librobot_vla --token YOUR_TOKEN --labels self-hosted,linux,gpu,cuda
```

## Step 3: Run as a Service (Recommended)

To keep the runner running in the background:

### Linux (systemd)

```bash
# Install the runner as a service
sudo ./svc.sh install

# Start the service
sudo ./svc.sh start

# Check status
sudo ./svc.sh status

# Stop the service
sudo ./svc.sh stop

# Uninstall the service
sudo ./svc.sh uninstall
```

### Windows (Service)

```powershell
# Install as a Windows service
.\config.cmd --url https://github.com/wfhit/librobot_vla --token YOUR_TOKEN
.\svc.cmd install
.\svc.cmd start
```

### macOS (launchd)

```bash
./svc.sh install
./svc.sh start
```

## Step 4: Create Workflows for Self-Hosted Runners

Create or modify workflow files to use your self-hosted runner. See the example workflow at `.github/workflows/self-hosted-test.yml`.

### Basic Example

```yaml
name: Self-Hosted Tests

on:
  push:
    branches: [main, develop]
  workflow_dispatch:  # Allow manual triggering

jobs:
  gpu-tests:
    runs-on: [self-hosted, linux, gpu]  # Match your runner labels
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        run: |
          python3 -m venv venv
          source venv/bin/activate
          pip install -e ".[dev]"
      
      - name: Run GPU tests
        run: |
          source venv/bin/activate
          pytest tests/ -v -m gpu
```

### Key Differences from GitHub-Hosted Runners

1. **Python setup**: Use system Python or create a virtual environment (actions/setup-python may not work)
2. **Clean environment**: Self-hosted runners don't automatically clean up, so clean build artifacts
3. **Security**: Be careful with secrets on self-hosted runners in public repos

## Step 5: Security Best Practices

### For Public Repositories

⚠️ **Warning**: Using self-hosted runners with public repositories can be a security risk.

- Only use self-hosted runners in **private repositories** or with **fork protection** enabled
- Never store credentials on the runner machine
- Use ephemeral runners when possible
- Keep the runner software updated

### For Private Repositories

- Use dedicated machines for CI (not production servers)
- Restrict who can modify workflows
- Monitor runner activity

### Network Security

```bash
# Minimal required outbound connections:
# - github.com (HTTPS)
# - api.github.com (HTTPS)
# - codeload.github.com (HTTPS)
# - objects.githubusercontent.com (HTTPS)
# - pipelines.actions.githubusercontent.com (HTTPS)
```

## Troubleshooting

### Runner Not Picking Up Jobs

1. Check runner status in GitHub Settings → Actions → Runners
2. Verify labels match workflow `runs-on` specification
3. Check runner logs: `_diag/Runner_*.log`

### Permission Issues

```bash
# Fix common permission issues on Linux
sudo chown -R $USER:$USER actions-runner
chmod -R 755 actions-runner
```

### GPU Not Available in Jobs

```bash
# Verify NVIDIA drivers are accessible
nvidia-smi

# Check CUDA installation
nvcc --version

# Ensure runner user has GPU access
sudo usermod -aG video $USER
```

### Docker Support

If using Docker in self-hosted runners:

```bash
# Install Docker
sudo apt install docker.io

# Add runner user to docker group
sudo usermod -aG docker $USER

# Restart the runner service
sudo ./svc.sh restart
```

## Removing a Runner

To remove a self-hosted runner:

1. Stop the runner service
2. On GitHub: Settings → Actions → Runners → Select runner → Remove
3. Delete the runner folder on your machine

```bash
# Stop and uninstall service
sudo ./svc.sh stop
sudo ./svc.sh uninstall

# Remove from GitHub (get token from GitHub)
./config.sh remove --token YOUR_REMOVE_TOKEN
```

## Additional Resources

- [GitHub Actions Self-Hosted Runner Documentation](https://docs.github.com/en/actions/hosting-your-own-runners)
- [Runner Releases](https://github.com/actions/runner/releases)
- [Security hardening for self-hosted runners](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions#hardening-for-self-hosted-runners)
