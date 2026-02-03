#!/usr/bin/env python
"""
CloudyView Benchmark Script

Benchmarks behold render performance with different ray tracing parameters.
Results are appended to benchmarking/results.md with hardware info and timestamps.

Usage:
    python benchmarking/benchmark.py          # Use CUDA (GPU) backend (default)
    python benchmarking/benchmark.py --llvm   # Use LLVM (CPU) backend

Test cases:
    1. 400x300, rr_depth=4, max_depth=8 (fast, lower quality)
    2. 400x300, rr_depth=64, max_depth=128 (higher quality)
    3. 1600x1200, rr_depth=64, max_depth=128 (high resolution)
"""

import argparse
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for cloudyview imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import mitsuba as mi

from cloudyview import behold

# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================
# Modify these values to change the benchmark parameters

# Data file (relative to project root)
DATA_FILE = "data/TWPICE_subvolume_256x256_5km.nc"

# Camera/sun settings (explicit defaults in case config changes)
CAMERA_POSITION = [0, 0, -0.9]
CAMERA_AZIMUTH = 90.0
CAMERA_ELEVATION = 35.0
CAMERA_FOV = 100.0
SUN_AZIMUTH = 70.0
SUN_ELEVATION = 55.0

# Render settings
BENCHMARK_SPP = 32

# Test cases: each dict defines a benchmark scenario
TEST_CASES = [
    {'name': '400x300 Fast', 'resolution': (400, 300), 'rr_depth': 4, 'max_depth': 8},
    {'name': '400x300 High Quality', 'resolution': (400, 300), 'rr_depth': 64, 'max_depth': 128},
    {'name': '1600x1200 Fast', 'resolution': (1600, 1200), 'rr_depth': 4, 'max_depth': 8},
]

# =============================================================================


def get_cpu_info() -> str:
    """Get CPU information."""
    processor = platform.processor()
    cpu_count = os.cpu_count()
    return f"{processor} ({cpu_count} cores)"


def get_gpu_info() -> str:
    """Get GPU information via nvidia-smi if available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            # Parse output: "GPU Name, Memory MB"
            lines = result.stdout.strip().split('\n')
            gpus = []
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 2:
                    name = parts[0].strip()
                    memory_mb = int(parts[1].strip())
                    gpus.append(f"{name} ({memory_mb // 1024} GB)")
            return ", ".join(gpus) if gpus else "Unknown GPU"
        return "No NVIDIA GPU detected"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "nvidia-smi not available"


def get_memory_info() -> str:
    """Get system memory info using psutil if available."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return f"{mem.total / (1024**3):.1f} GB total"
    except ImportError:
        return "psutil not installed"


def run_benchmark(data_file: str, backend: str, rr_depth: int, max_depth: int,
                  resolution: tuple, output_dir: Path) -> float:
    """
    Run a single benchmark case.

    Returns render time in seconds.
    """
    print(f"\n--- Benchmark: {resolution[0]}x{resolution[1]}, rr_depth={rr_depth}, max_depth={max_depth} ---")

    start = time.perf_counter()

    behold.main(
        filename=data_file,
        backend=backend,
        quality='custom',
        output=str(output_dir),
        custom_spp=BENCHMARK_SPP,
        custom_size=resolution,
        custom_max_depth=max_depth,
        custom_rr_depth=rr_depth,
        camera_position=CAMERA_POSITION,
        camera_azimuth=CAMERA_AZIMUTH,
        camera_elevation=CAMERA_ELEVATION,
        camera_fov=CAMERA_FOV,
        sun_azimuth=SUN_AZIMUTH,
        sun_elevation=SUN_ELEVATION,
    )

    elapsed = time.perf_counter() - start
    print(f"Completed in {elapsed:.2f} seconds")

    return elapsed


def write_result(results_file: Path, hardware_info: dict, result: dict):
    """Write a single self-contained result entry to the results file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create file with header if it doesn't exist
    if not results_file.exists():
        with open(results_file, 'w') as f:
            f.write("# CloudyView Benchmark Results\n\n")

    res = f"{result['resolution'][0]}x{result['resolution'][1]}"

    with open(results_file, 'a') as f:
        f.write(f"---\n\n")
        f.write(f"## {timestamp} - {result['name']}\n\n")

        # Hardware info
        f.write("### Hardware\n\n")
        f.write(f"- **OS**: {hardware_info['os']}\n")
        f.write(f"- **CPU**: {hardware_info['cpu']}\n")
        f.write(f"- **GPU**: {hardware_info['gpu']}\n")
        f.write(f"- **Memory**: {hardware_info['memory']}\n")
        f.write(f"- **Backend**: {hardware_info['backend']}\n\n")

        # Test configuration
        f.write("### Configuration\n\n")
        f.write(f"- **Data file**: {hardware_info['data_file']}\n")
        f.write(f"- **Resolution**: {res}\n")
        f.write(f"- **SPP**: {BENCHMARK_SPP}\n")
        f.write(f"- **rr_depth**: {result['rr_depth']}\n")
        f.write(f"- **max_depth**: {result['max_depth']}\n")
        f.write(f"- **Camera position**: {CAMERA_POSITION}\n")
        f.write(f"- **Camera azimuth/elevation**: {CAMERA_AZIMUTH}° / {CAMERA_ELEVATION}°\n")
        f.write(f"- **Camera FOV**: {CAMERA_FOV}°\n")
        f.write(f"- **Sun azimuth/elevation**: {SUN_AZIMUTH}° / {SUN_ELEVATION}°\n\n")

        # Result
        f.write(f"### Result\n\n")
        f.write(f"**Time: {result['time']:.2f} seconds**\n\n")




def main():
    """Run the benchmark suite."""
    parser = argparse.ArgumentParser(description="CloudyView Benchmark Script")
    parser.add_argument(
        "--llvm",
        action="store_true",
        help="Use LLVM (CPU) backend instead of CUDA (GPU)"
    )
    args = parser.parse_args()

    backend = 'llvm' if args.llvm else 'cuda'

    print("=" * 60)
    print("CloudyView Benchmark")
    print("=" * 60)

    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_file = project_dir / DATA_FILE
    output_dir = script_dir / "output"
    results_file = script_dir / "results.md"

    # Verify data file exists
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Gather hardware info
    print("\nGathering hardware information...")

    hardware_info = {
        'os': f"{platform.system()} {platform.release()}",
        'cpu': get_cpu_info(),
        'gpu': get_gpu_info(),
        'memory': get_memory_info(),
        'backend': backend,
        'data_file': data_file.name
    }

    print(f"  OS: {hardware_info['os']}")
    print(f"  CPU: {hardware_info['cpu']}")
    print(f"  GPU: {hardware_info['gpu']}")
    print(f"  Memory: {hardware_info['memory']}")
    print(f"  Backend: {backend}")

    # Run benchmarks
    print("\n" + "=" * 60)
    print("Running benchmarks...")
    print("=" * 60)

    print(f"Results will be saved to: {results_file}")

    benchmark_results = []
    for case in TEST_CASES:
        elapsed = run_benchmark(
            data_file=str(data_file),
            backend=backend,
            rr_depth=case['rr_depth'],
            max_depth=case['max_depth'],
            resolution=case['resolution'],
            output_dir=output_dir
        )
        result = {
            'name': case['name'],
            'resolution': case['resolution'],
            'rr_depth': case['rr_depth'],
            'max_depth': case['max_depth'],
            'time': elapsed
        }
        benchmark_results.append(result)

        # Save this result immediately (self-contained row)
        write_result(results_file, hardware_info, result)

    print(f"\nResults saved to: {results_file}")

    # Summary
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    for result in benchmark_results:
        res = f"{result['resolution'][0]}x{result['resolution'][1]}"
        print(f"  {result['name']} ({res}): {result['time']:.2f}s")


if __name__ == "__main__":
    main()
