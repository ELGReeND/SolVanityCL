import logging
import platform
from pathlib import Path
from typing import Sequence, Tuple

import pyopencl as cl
from base58 import b58decode


BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
BASE58_INDEX = {c: i for i, c in enumerate(BASE58_ALPHABET)}

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
                prefix_entries.append([BASE58_INDEX[c] for c in prefix])
                suffix_entries.append([BASE58_INDEX[c] for c in ends_with])
        else:
            prefix_entries.append([])
            suffix_entries.append([BASE58_INDEX[c] for c in ends_with])

    max_prefix_len = max((len(p) for p in prefix_entries), default=0)
    max_suffix_len = max((len(s) for s in suffix_entries), default=0)
    prefix_lengths_list = [len(p) for p in prefix_entries]
    suffix_lengths_list = [len(s) for s in suffix_entries]
    suffix_max_dim = max(max_suffix_len, 1)

    for p in prefix_entries:
        p.extend([0] * (max_prefix_len - len(p)))

    for s in suffix_entries:
        s.extend([0] * (suffix_max_dim - len(s)))

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
        elif line.startswith("#define SUFFIX_MAX "):
            source_lines[i] = f"#define SUFFIX_MAX {suffix_max_dim}\n"
        elif line.startswith("constant uchar PREFIXES"):
            prefixes_str = "{" + ", ".join("{" + ", ".join(map(str, prefix)) + "}" for prefix in prefix_entries) + "}"
            source_lines[i] = f"constant uchar PREFIXES[N][L] = {prefixes_str};\n"
        elif line.startswith("constant uchar PREFIX_LENGTHS"):
            prefix_lengths = (
                "{" + ", ".join(str(l) for l in prefix_lengths_list) + "}"
            )
            source_lines[i] = (
                f"constant uchar PREFIX_LENGTHS[N] = {prefix_lengths};\n"
            )
        elif line.startswith("constant uchar SUFFIXES"):
            suffixes_str = "{" + ", ".join("{" + ", ".join(map(str, suffix)) + "}" for suffix in suffix_entries) + "}"
            source_lines[i] = (
                f"constant uchar SUFFIXES[N][SUFFIX_MAX] = {suffixes_str};\n"
            )
        elif line.startswith("constant uchar SUFFIX_LENGTHS"):
            suffix_lengths = "{" + ", ".join(str(l) for l in suffix_lengths_list) + "}"
            source_lines[i] = (
                f"constant uchar SUFFIX_LENGTHS[N] = {suffix_lengths};\n"
            )
        elif line.startswith("constant uchar PREFIX_FIRST"):
            prefix_first = "{" + ", ".join(
                str(p[0] if l > 0 else 255)
                for p, l in zip(prefix_entries, prefix_lengths_list)
            ) + "}"
            source_lines[i] = f"constant uchar PREFIX_FIRST[N] = {prefix_first};\n"
        elif line.startswith("constant uchar SUFFIX_LAST"):
            suffix_last = "{" + ", ".join(
                str(s[l - 1] if l > 0 else 255)
                for s, l in zip(suffix_entries, suffix_lengths_list)
            ) + "}"
            source_lines[i] = f"constant uchar SUFFIX_LAST[N] = {suffix_last};\n"

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
