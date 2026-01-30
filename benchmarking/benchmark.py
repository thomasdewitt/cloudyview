#!/usr/bin/env python
"""
CloudyView Benchmark Script

Benchmarks behold render performance with different ray tracing parameters.
Results are appended to benchmarking/results.md with hardware info and timestamps.

Usage:
    python benchmarking/benchmark.py

Test cases:
    1. rr_depth=4, max_depth=8 (fast, lower quality)
    2. rr_depth=64, max_depth=128 (slow, higher quality)
"""

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


def get_mitsuba_backends() -> str:
    """Get available Mitsuba backends."""
    variants = mi.variants()
    return ", ".join(variants) if variants else "None"


def detect_backend() -> str:
    """Detect best available backend (cuda if available, else llvm)."""
    variants = mi.variants()
    for v in variants:
        if 'cuda' in v:
            return 'cuda'
    return 'llvm'


def run_benchmark(data_file: str, backend: str, rr_depth: int, max_depth: int,
                  output_dir: Path) -> float:
    """
    Run a single benchmark case.

    Returns render time in seconds.
    """
    print(f"\n--- Benchmark: rr_depth={rr_depth}, max_depth={max_depth} ---")

    start = time.perf_counter()

    behold.main(
        filename=data_file,
        backend=backend,
        quality='custom',
        output=str(output_dir),
        custom_spp=2,
        custom_size=(400, 400),
        custom_max_depth=max_depth,
        custom_rr_depth=rr_depth
    )

    elapsed = time.perf_counter() - start
    print(f"Completed in {elapsed:.2f} seconds")

    return elapsed


def append_results(results_file: Path, hardware_info: dict, benchmark_results: list):
    """Append benchmark results to markdown file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create file with header if it doesn't exist
    if not results_file.exists():
        with open(results_file, 'w') as f:
            f.write("# CloudyView Benchmark Results\n\n")
            f.write("This file contains benchmark results for CloudyView behold renders.\n\n")

    with open(results_file, 'a') as f:
        f.write(f"---\n\n")
        f.write(f"## {timestamp}\n\n")

        # Hardware info
        f.write("### Hardware Configuration\n\n")
        f.write(f"- **OS**: {hardware_info['os']}\n")
        f.write(f"- **CPU**: {hardware_info['cpu']}\n")
        f.write(f"- **GPU**: {hardware_info['gpu']}\n")
        f.write(f"- **Memory**: {hardware_info['memory']}\n")
        f.write(f"- **Backend**: {hardware_info['backend']}\n")
        f.write(f"- **Mitsuba variants**: {hardware_info['mitsuba_variants']}\n\n")

        # Test configuration
        f.write("### Test Configuration\n\n")
        f.write(f"- **Data file**: {hardware_info['data_file']}\n")
        f.write(f"- **Resolution**: 400x400\n")
        f.write(f"- **SPP**: 128\n\n")

        # Results table
        f.write("### Results\n\n")
        f.write("| Test Case | rr_depth | max_depth | Time (s) |\n")
        f.write("|-----------|----------|-----------|----------|\n")
        for result in benchmark_results:
            f.write(f"| {result['name']} | {result['rr_depth']} | {result['max_depth']} | {result['time']:.2f} |\n")
        f.write("\n")

        # Speedup calculation
        if len(benchmark_results) >= 2:
            fast_time = benchmark_results[0]['time']
            slow_time = benchmark_results[1]['time']
            if fast_time > 0:
                speedup = slow_time / fast_time
                f.write(f"**Speedup (fast vs slow)**: {speedup:.2f}x\n\n")


def main():
    """Run the benchmark suite."""
    print("=" * 60)
    print("CloudyView Benchmark")
    print("=" * 60)

    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_file = project_dir / "data" / "QC_FIF_Square_512,512,256.nc"
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
    backend = detect_backend()

    hardware_info = {
        'os': f"{platform.system()} {platform.release()}",
        'cpu': get_cpu_info(),
        'gpu': get_gpu_info(),
        'memory': get_memory_info(),
        'backend': backend,
        'mitsuba_variants': get_mitsuba_backends(),
        'data_file': data_file.name
    }

    print(f"  OS: {hardware_info['os']}")
    print(f"  CPU: {hardware_info['cpu']}")
    print(f"  GPU: {hardware_info['gpu']}")
    print(f"  Memory: {hardware_info['memory']}")
    print(f"  Backend: {backend}")
    print(f"  Mitsuba variants: {hardware_info['mitsuba_variants']}")

    # Define benchmark cases
    test_cases = [
        {'name': 'Fast (lower quality)', 'rr_depth': 4, 'max_depth': 8},
        {'name': 'Slow (higher quality)', 'rr_depth': 64, 'max_depth': 128},
    ]

    # Run benchmarks
    print("\n" + "=" * 60)
    print("Running benchmarks...")
    print("=" * 60)

    benchmark_results = []
    for case in test_cases:
        elapsed = run_benchmark(
            data_file=str(data_file),
            backend=backend,
            rr_depth=case['rr_depth'],
            max_depth=case['max_depth'],
            output_dir=output_dir
        )
        benchmark_results.append({
            'name': case['name'],
            'rr_depth': case['rr_depth'],
            'max_depth': case['max_depth'],
            'time': elapsed
        })

    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)

    append_results(results_file, hardware_info, benchmark_results)
    print(f"Results appended to: {results_file}")

    # Summary
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    for result in benchmark_results:
        print(f"  {result['name']}: {result['time']:.2f}s")

    if len(benchmark_results) >= 2:
        speedup = benchmark_results[1]['time'] / benchmark_results[0]['time']
        print(f"\n  Speedup (fast vs slow): {speedup:.2f}x")


if __name__ == "__main__":
    main()
