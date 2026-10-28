#!/usr/bin/env python3
"""Fail-closed fixed-unit-seed patch for the frozen production fitting ISA.

This experiment keeps the production low-register kernel's instruction layout,
register allocation, launch ABI, barriers, and reverse traversal intact.  At the
two guarded seed-load sites it removes the FP64 global load/wait/convert and
materializes exact FP32 1.0 in the existing scale register.  The row bounds and
branches are unchanged, so inactive tail rows retain their existing zero scale.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


REFERENCE_INPUT_SHA256S = {
    "7bf5d87fb3545d70bc86d734518fcc9e23e5e97ac566bdf6a72af4ac89067a9c",
}
NOP = bytes.fromhex("0000000000000008")

BASE_PATH = Path(__file__).with_name("patch_lowreg_reverse.py")
SPEC = importlib.util.spec_from_file_location("lowreg_reverse_base", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {BASE_PATH}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


ARCH = {
    "ppu1.0": {
        "stable_fingerprint": (
            "8c154fc0396ba35657a5c2bdd682d1909d37b9185a309aa6cec1c4d8d620be13"
        ),
        "expected_text_offset": 0x33AD0,
        "sites": [
            {
                "row": 0,
                "load": (0xBF00, "40250000a08bcbc8"),
                "wait": (0xBF08, "000000000000c408"),
                "convert": (0xBF10, "0000009c804b8942"),
                "one": "0000803f20400950",
                "scale_register": "vreg37",
            },
            {
                "row": 1,
                "load": (0xBF58, "40250004a08acac8"),
                "wait": (0xBF60, "000000000000c408"),
                "convert": (0xBF68, "0000009c800a9842"),
                "one": "0000803f20001850",
                "scale_register": "vreg96",
            },
        ],
    },
    "ppu1.5": {
        "stable_fingerprint": (
            "2c84701f4a042cf8c90cdea06c7325bcc05f21bc79b29e97fedb887113f627a2"
        ),
        "expected_text_offset": 0x2E3E0,
        "sites": [
            {
                "row": 0,
                "load": (0x9F98, "40250000200acac8"),
                "wait": (0x9FA0, "000000000000c408"),
                "convert": (0x9FA8, "0000009c004a8942"),
                "one": "0000803f20400950",
                "scale_register": "vreg37",
            },
            {
                "row": 1,
                "load": (0x9FF8, "40250004200acac8"),
                "wait": (0xA000, "000000000000c408"),
                "convert": (0xA008, "0000009c000a8a42"),
                "one": "0000803f20000a50",
                "scale_register": "vreg40",
            },
        ],
    },
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    original = args.input.read_bytes()
    input_sha = sha256(original)

    data = bytearray(original)
    allowed: set[int] = set()
    records = []

    def patch(arch: str, identity: dict, address: int, before: bytes,
              after: bytes, role: str, expected_assembly: str) -> None:
        if len(before) != 8 or len(after) != 8:
            raise SystemExit(f"{arch} {address:#x}: patch is not one instruction")
        absolute = identity["elf_offset"] + identity["text_offset"] + address
        actual = bytes(data[absolute:absolute + 8])
        if actual != before:
            raise SystemExit(
                f"{arch} {address:#x}: {actual.hex()} != {before.hex()}"
            )
        data[absolute:absolute + 8] = after
        allowed.update(range(absolute, absolute + 8))
        records.append({
            "arch": arch,
            "role": role,
            "row": role.rsplit("_", 1)[-1],
            "isa_address": hex(address),
            "file_offset": hex(absolute),
            "before": before.hex(),
            "after": after.hex(),
            "expected_assembly": expected_assembly,
        })

    embedded = {}
    resolved_identities = {}
    for arch, raw_identity in ARCH.items():
        identity = dict(raw_identity)
        resolved = base.find_unique_embedded_elf(
            original, identity["stable_fingerprint"]
        )
        target_sections = [
            section for section in resolved["sections"]
            if section["name"] == base.TARGET_TEXT_SECTION
        ]
        if len(target_sections) != 1:
            raise SystemExit(f"{arch}: target text section count {len(target_sections)}")
        target = target_sections[0]
        if target["offset"] != identity["expected_text_offset"]:
            raise SystemExit(
                f"{arch}: target text offset {target['offset']:#x} != "
                f"{identity['expected_text_offset']:#x}"
            )
        identity.update(resolved)
        identity["text_offset"] = target["offset"]
        begin = identity["elf_offset"]
        end = begin + identity["elf_size"]
        elf_sha = sha256(original[begin:end])
        embedded[arch] = {
            "offset": hex(begin),
            "size": identity["elf_size"],
            "input_sha256": elf_sha,
            "text_offset": hex(identity["text_offset"]),
            "stable_fingerprint": identity["stable_fingerprint"],
        }
        resolved_identities[arch] = identity
        for site in identity["sites"]:
            row = site["row"]
            patch(arch, identity, site["load"][0],
                  bytes.fromhex(site["load"][1]), NOP,
                  f"remove_seed_load_row_{row}", "s.nop")
            patch(arch, identity, site["wait"][0],
                  bytes.fromhex(site["wait"][1]), NOP,
                  f"remove_seed_wait_row_{row}", "s.nop")
            patch(arch, identity, site["convert"][0],
                  bytes.fromhex(site["convert"][1]),
                  bytes.fromhex(site["one"]),
                  f"materialize_unit_seed_row_{row}",
                  f"v.mov.b32 {site['scale_register']}, 0x3f800000")

    changed = [i for i, (a, b) in enumerate(zip(original, data)) if a != b]
    if not set(changed) <= allowed:
        raise SystemExit("byte changed outside the audited instruction whitelist")
    if len(records) != 12:
        raise SystemExit(f"patch record count {len(records)} != 12")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    for arch, identity in resolved_identities.items():
        begin = identity["elf_offset"]
        end = begin + identity["elf_size"]
        embedded[arch]["output_sha256"] = sha256(data[begin:end])
    manifest = {
        "schema": 1,
        "status": "experiment-only-static-validation-required",
        "transformation": (
            "fixed-unit-seed load elimination in frozen production ISA"
        ),
        "input": str(args.input),
        "output": str(args.output),
        "input_sha256": input_sha,
        "reference_full_file_sha256_match": input_sha in REFERENCE_INPUT_SHA256S,
        "output_sha256": sha256(data),
        "architectures": list(ARCH),
        "embedded_elf_identities": embedded,
        "instruction_patch_count": len(records),
        "changed_byte_count": len(changed),
        "invariants": [
            "same launch ABI and kernel symbol",
            "same 252/248-vreg resource metadata and zero stack",
            "same reverse task traversal and physical-tail MMA pruning",
            "same bounds branches; inactive tail rows keep zero scale",
            "exact FP32 unit cotangent is materialized in existing registers",
            "no coordinate, force, energy, model, or topology data is cached",
        ],
        "production_requirements": [
            "restore audited operator-to-chain unit-seed pointer-range contract",
            "fail closed to the generic path for unregistered seed ranges",
        ],
        "patches": records,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "patches"}, indent=2))


if __name__ == "__main__":
    main()
