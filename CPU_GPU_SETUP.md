# DeepSeek-OCR CPU/GPU Flexible Setup Guide

This guide explains how to configure DeepSeek-OCR to work with either NVIDIA GPU or CPU, automatically detecting and using whichever is available.

## Quick Start

The Docker setup now automatically detects and uses:
- **NVIDIA GPU** if available and properly configured
- **CPU** as fallback if no GPU is available

## Prerequisites

### For GPU Mode
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime installed
- NVIDIA Container Toolkit

### For CPU Mode
- No special requirements - works on any system that supports Docker

## Configuration

### Environment Variables

The following environment variables in `docker-compose.yml` control the behavior:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0  # Set to "" for CPU-only mode
  - MAX_CONCURRENCY=5       # Reduce to 1-2 for CPU mode
  - GPU_MEMORY_UTILIZATION=0.85  # Ignored in CPU mode
```

### Automatic Detection

The system automatically:
1. Checks for CUDA availability
2. Configures vLLM for GPU or CPU mode accordingly
3. Adjusts memory utilization and concurrency settings
4. Falls back to CPU if GPU is not available

## Usage

### Standard Usage (Auto-detect)
```bash
docker-compose up -d
```

### Force CPU Mode
To force CPU mode even if GPU is available:
```bash
# Edit docker-compose.yml and set:
# CUDA_VISIBLE_DEVICES=""
# MAX_CONCURRENCY=1
docker-compose up -d
```

### Verify Mode
Check the logs to see which mode is being used:
```bash
docker-compose logs deepseek-ocr
```

Look for messages like:
- `"GPU detected: [GPU Name]"` - Using GPU mode
- `"No GPU detected, using CPU mode"` - Using CPU mode

## Performance Considerations

### GPU Mode
- **Recommended**: MAX_CONCURRENCY=5-10
- **GPU_MEMORY_UTILIZATION**: 0.85 (adjust based on GPU memory)
- **Performance**: Much faster processing

### CPU Mode
- **Recommended**: MAX_CONCURRENCY=1-2
- **Performance**: Slower but functional
- **Memory**: Uses system RAM instead of GPU memory

## Troubleshooting

### GPU Not Detected
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify Docker runtime: `docker info | grep nvidia`
3. Install NVIDIA Container Toolkit if needed

### CPU Mode Issues
1. Reduce MAX_CONCURRENCY to 1
2. Ensure sufficient system RAM (8GB+ recommended)
3. Check Docker memory limits

### Switching Between Modes
1. Stop the container: `docker-compose down`
2. Modify environment variables as needed
3. Restart: `docker-compose up -d`

## Health Check

The API provides health check endpoints:
- `http://localhost:8000/` - Basic health check
- `http://localhost:8000/health` - Detailed status including GPU/CPU mode