import logging
import multiprocessing
import os
import signal
import sys
from multiprocessing import TimeoutError
from multiprocessing.pool import Pool
from typing import List, Optional, Tuple

import click
import pyopencl as cl

from core.config import DEFAULT_ITERATION_BITS, HostSetting
from core.opencl.manager import (
    get_all_gpu_devices,
    get_chosen_devices,
    get_device_by_index,
)
from core.utils.crypto import save_keypair
from core.searcher import multi_gpu_init
from core.utils.helpers import check_character, load_kernel_source

logging.basicConfig(level="INFO", format="[%(levelname)s %(asctime)s] %(message)s")


def _init_worker() -> None:
    """Ignore SIGINT in workers so the main process can handle Ctrl+C cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


@click.group()
def cli():
    pass


@cli.command(context_settings={"show_default": True})
@click.option(
    "--starts-with",
    type=str,
    default=[],
    help="Public key starts with the indicated prefix. Provide multiple arguments to search for multiple prefixes.",
    multiple=True,
)
@click.option(
    "--ends-with",
    type=str,
    default="",
    help="Public key ends with the indicated suffix.",
)
@click.option("--count", type=int, default=1, help="Count of pubkeys to generate.")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default="./",
    help="Output directory.",
)
@click.option(
    "--select-device",
    "select_device",
    type=int,
    default=None,
    help="Select a single OpenCL GPU device by its global index (0-based). Use `show_device` to see indices.",
)
@click.option(
    "--iteration-bits",
    type=int,
    default=DEFAULT_ITERATION_BITS,
    help="Iteration bits (e.g., 24, 26, 28, etc.)",
)
@click.option(
    "--is-case-sensitive", type=bool, default=True, help="Case sensitive search flag."
)
@click.option(
    "--starts-and-ends-with-file",
    "starts_ends_file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=None,
    help="Path to a file containing prefix/suffix pairs in the form <PREFIX>:<SUFFIX>, one per line.",
)
def search_pubkey(
    starts_with,
    ends_with,
    count,
    output_dir,
    select_device,
    iteration_bits,
    is_case_sensitive,
    starts_ends_file,
):
    """Search for Solana vanity pubkeys."""
    if starts_ends_file and (starts_with or ends_with):
        click.echo(
            "Use either --starts-and-ends-with-file or the --starts-with/--ends-with options, not both."
        )
        sys.exit(1)

    patterns: List[Tuple[List[str], str]] = []
    if starts_ends_file:
        raw_pairs: List[Tuple[str, str]] = []
        with open(starts_ends_file, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if ":" not in stripped:
                    click.echo(
                        f"Invalid line in {starts_ends_file}: {stripped!r}. Expected format <PREFIX>:<SUFFIX>."
                    )
                    sys.exit(1)
                prefix, suffix = stripped.split(":", 1)
                check_character("starts_with", prefix)
                check_character("ends_with", suffix)
                raw_pairs.append((prefix, suffix))

        seen = set()
        duplicate_count = 0
        for prefix, suffix in raw_pairs:
            key = (
                prefix if is_case_sensitive else prefix.lower(),
                suffix if is_case_sensitive else suffix.lower(),
            )
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            patterns.append(([prefix], suffix))

        if duplicate_count:
            logging.info(
                "Removed %s duplicate line(s) from %s (unique patterns=%s)",
                duplicate_count,
                starts_ends_file,
                len(patterns),
            )
        if not patterns:
            click.echo(
                f"No valid prefix/suffix pairs found in {starts_ends_file}. Nothing to search."
            )
            sys.exit(1)
    else:
        if not starts_with and not ends_with:
            click.echo("Please provide at least one of --starts-with or --ends-with.")
            ctx = click.get_current_context()
            click.echo(ctx.get_help())
            sys.exit(1)
        for prefix in starts_with:
            check_character("starts_with", prefix)
        check_character("ends_with", ends_with)
        if starts_with:
            for prefix in starts_with:
                patterns.append(([prefix], ends_with))
        else:
            patterns.append(([], ends_with))

    chosen_devices: Optional[Tuple[int, List[int]]] = None
    if select_device is not None:
        platform_id, device_id = get_device_by_index(select_device)
        chosen_devices = (platform_id, [device_id])
        gpu_counts = len(chosen_devices[1])
        logging.info(
            "Using selected GPU device %s (platform %s, device %s)",
            select_device,
            platform_id,
            device_id,
        )
    elif os.environ.get("CHOSEN_OPENCL_DEVICES"):
        chosen_devices = get_chosen_devices()
        gpu_counts = len(chosen_devices[1])
    else:
        gpu_counts = len(get_all_gpu_devices())

    logging.info(
        "Using %s OpenCL device(s) with case_sensitive=%s",
        gpu_counts,
        is_case_sensitive,
    )

    remaining_per_pattern = [count for _ in patterns]
    total_remaining = sum(remaining_per_pattern)

    with multiprocessing.Manager() as manager:
        kernel_source = load_kernel_source(patterns, is_case_sensitive)
        lock = manager.Lock()
        pool = Pool(processes=gpu_counts, initializer=_init_worker)

        try:
            while total_remaining > 0:
                stop_flag = manager.Value("i", 0)
                async_result = pool.starmap_async(
                    multi_gpu_init,
                    [
                        (
                            x,
                            HostSetting(kernel_source, iteration_bits),
                            gpu_counts,
                            stop_flag,
                            lock,
                            chosen_devices,
                        )
                        for x in range(gpu_counts)
                    ],
                )
                while True:
                    try:
                        results = async_result.get(timeout=1)
                        break
                    except TimeoutError:
                        continue

                for output in results:
                    if not output or int.from_bytes(bytes(output[0:4]), byteorder='little', signed=False) == 0:
                        continue
                    pattern_plus = int.from_bytes(bytes(output[0:4]), byteorder='little', signed=False)
                    if not pattern_plus:
                        continue
                    pattern_idx = pattern_plus - 1
                    if pattern_idx < 0 or pattern_idx >= len(remaining_per_pattern):
                        continue
                    if remaining_per_pattern[pattern_idx] <= 0:
                        continue
                    remaining_per_pattern[pattern_idx] -= 1
                    total_remaining -= 1
                    pv_bytes = bytes(output[8:40])
                    save_keypair(pv_bytes, output_dir)
        except KeyboardInterrupt:
            logging.info("Stopping search after receiving Ctrl+C.")
            pool.terminate()
        else:
            pool.close()
        finally:
            pool.join()


@cli.command(context_settings={"show_default": True})
def show_device():
    """Show available OpenCL devices."""
    platforms = cl.get_platforms()
    global_index = 0
    for p_index, platform in enumerate(platforms):
        click.echo(f"Platform {p_index}: {platform.name}")
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        for d_index, device in enumerate(devices):
            click.echo(
                f"  - Device {d_index}: {device.name} (global index {global_index})"
            )
            global_index += 1


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    cli()
