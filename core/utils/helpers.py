import logging
import platform
from pathlib import Path
from typing import Sequence, Tuple

import pyopencl as cl
from base58 import b58decode


def check_character(name: str, character: str) -> None:
    try:
        b58decode(character)
    except ValueError as e:
        logging.error(f"{str(e)} in {name}")
        raise SystemExit(1)
    except Exception as e:
        raise e


def load_kernel_source(
    patterns: Sequence[Tuple[Sequence[str], str]], is_case_sensitive: bool
) -> str:
    """
    Update OpenCL codes with parameters
    """
    prefix_entries = []
    suffix_entries = []
    for starts_with_list, ends_with in patterns:
        if starts_with_list:
            for prefix in starts_with_list:
                prefix_entries.append(list(prefix.encode()))
                suffix_entries.append(list(ends_with.encode()))
        else:
            prefix_entries.append([])
            suffix_entries.append(list(ends_with.encode()))

    max_prefix_len = max((len(p) for p in prefix_entries), default=0)
    max_suffix_len = max((len(s) for s in suffix_entries), default=0)

    for p in prefix_entries:
        p.extend([0] * (max_prefix_len - len(p)))

    for s in suffix_entries:
        s.extend([0] * (max_suffix_len - len(s)))

    kernel_path = Path(__file__).parent.parent / "opencl" / "kernel.cl"
    if not kernel_path.exists():
        raise FileNotFoundError("Kernel source file not found.")
    with kernel_path.open("r") as f:
        source_lines = f.readlines()

    for i, line in enumerate(source_lines):
        if line.startswith("#define N "):
            source_lines[i] = f"#define N {len(prefix_entries)}\n"
        elif line.startswith("#define L "):
            source_lines[i] = f"#define L {max_prefix_len}\n"
        elif line.startswith("#define S "):
            source_lines[i] = f"#define S {max_suffix_len}\n"
        elif line.startswith("constant uchar PREFIXES"):
            prefixes_str = "{"
            for prefix in prefix_entries:
                prefixes_str += "{" + ", ".join(map(str, prefix)) + "}, "
            prefixes_str = prefixes_str.rstrip(", ") + "}"
            source_lines[i] = f"constant uchar PREFIXES[N][L] = {prefixes_str};\n"
        elif line.startswith("constant uchar PREFIX_LENGTHS"):
            prefix_lengths = (
                "{" + ", ".join(str(len(p)) for p in prefix_entries) + "}"
            )
            source_lines[i] = (
                f"constant uchar PREFIX_LENGTHS[N] = {prefix_lengths};\n"
            )
        elif line.startswith("constant uchar SUFFIXES"):
            suffixes_str = "{"
            for suffix in suffix_entries:
                suffixes_str += "{" + ", ".join(map(str, suffix)) + "}, "
            suffixes_str = suffixes_str.rstrip(", ") + "}"
            source_lines[i] = f"constant uchar SUFFIXES[N][S] = {suffixes_str};\n"
        elif line.startswith("constant uchar SUFFIX_LENGTHS"):
            suffix_lengths = "{" + ", ".join(str(len(s)) for s in suffix_entries) + "}"
            source_lines[i] = (
                f"constant uchar SUFFIX_LENGTHS[N] = {suffix_lengths};\n"
            )
        elif line.startswith("constant bool CASE_SENSITIVE"):
            source_lines[i] = (
                f"constant bool CASE_SENSITIVE = {str(is_case_sensitive).lower()};\n"
            )

    source_str = "".join(source_lines)
    if "NVIDIA" in str(cl.get_platforms()) and platform.system() == "Windows":
        source_str = source_str.replace("#define __generic\n", "")
    if cl.get_cl_header_version()[0] != 1 and platform.system() != "Windows":
        source_str = source_str.replace("#define __generic\n", "")
    return source_str
