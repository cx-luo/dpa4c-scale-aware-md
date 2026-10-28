#!/usr/bin/env python3
"""Fail-closed fixed-DAG pruning for the low-register fitting kernel.

The compiler preserves the desired register allocation only for forward grid-stride
traversal. This tool rewrites an audited instruction whitelist so independent fitting
tasks are visited in reverse during the backward half, and removes only the layer-0
MMA outputs outside the coordinate VJP consumer range. Launch ABI, resource
metadata, barriers, fitting arithmetic and per-task accumulation order stay fixed.
The exact embedded device ELF identities and every before-image are verified; any
compiler or layout change fails closed.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path


REFERENCE_INPUT_SHA256S = {
    "9fbc7fe045733ee7b41dd0aef7bf01dc0c281ba7806f49e1fc49c9b98a631710",
    "aa5a08a3623269568f2957ce1cd0ba38240f99a298d2e4d59da6ed4576f9b74f",
}
NOP = bytes.fromhex("0000000000000008")


def cbr(origin: int, target: int) -> bytes:
    """Encode the observed PPU scalar unconditional relative branch."""
    delta = target - (origin + 8)
    if delta % 8:
        raise ValueError(f"unaligned branch {origin:#x} -> {target:#x}")
    displacement = delta // 8
    if not (-(1 << 19) <= displacement < (1 << 19)):
        raise ValueError(f"branch out of signed 20-bit range: {displacement}")
    encoded = displacement & 0xFFFFF
    return bytes((encoded & 0xFF, (encoded >> 8) & 0xFF,
                  (encoded >> 16) & 0x0F, 0x30, 0x20, 0, 0, 0x09))


TARGET_TEXT_SECTION = (
    ".text.kernel._ZN15lowreg_pipeline21block_pipeline_kernel"
    "ElPfPKlPKdS4_PKfPK5uint4S0_Pd"
)


ARCH = {
    "ppu1.0": {
        "reference_elf_sha256": "1d9da7074f6a0a9a4c916e1e2230e3defd7df13ef3fce1811cd26647abe949cf",
        "stable_fingerprint": "73442ddc84569374c181d8d1efa2fd63c82eeed81b149606cf5de8ee8350fbc1",
        "expected_text_offset": 0x33AD0,
        "entry_branch": 0x77C0,
        "trampoline": 0xCCA8,
        "sync": 0x77C8,
        "init": 0x7D38,
        "tail": 0xCC48,
    },
    "ppu1.5": {
        "reference_elf_sha256": "f031c649d4ec973130349dc1c84611f7b46cb53050554a331aeb643ee5134e60",
        "stable_fingerprint": "4ae04e2724a68c69f2716e3eb3658f1f4a5116bd762969bf5a0719889110f50c",
        "expected_text_offset": 0x2E3E0,
        "entry_branch": 0x67C0,
        "trampoline": 0xAD48,
        "sync": 0x67C8,
        "init": 0x6D38,
        "tail": 0xACE8,
    },
}


# Exact layer-0 fifth-tile MMA sites. The coordinate VJP consumes descriptor
# gradients 0..61, so dimensions 62..69 and padded lanes are mathematically
# dead. Keeping this as a post-link whitelist preserves the low-register
# allocation and all live accumulators.
PHYSICAL_TAIL_MMA = {
    "ppu1.0": {
        0xB218: "00004cc800198c70", 0xB280: "00004cce00188c70",
        0xB2F8: "00004cc800188c70", 0xB470: "00004cd400128c70",
        0xB4B0: "00004cbd801f8c70", 0xB4F8: "00004cd4801f8c70",
        0xB9C8: "000052e680249270", 0xB9F8: "000052ea80239270",
        0xBA10: "000052e680239270", 0xBAC8: "000052f6802a9270",
        0xBC68: "00004af280308a70", 0xBD40: "00004aee802d8a70",
        0xBD58: "00004af2802d8a70", 0xBD98: "000052ec80229270",
        0xBDD0: "000052f680229270", 0xBE68: "00004ae7802d8a70",
        0xBEA0: "00004af8802a8a70", 0xBEC0: "00004ae7802a8a70",
    },
    "ppu1.5": {
        0x95B0: "00844bdb85a28b70", 0x95D0: "00844bd985a28b70",
        0x9620: "00844be3049f8b70", 0x9660: "00844be7049e8b70",
        0x96B8: "00844be3049e8b70", 0x9970: "000454e5852a9470",
        0x99F0: "000454e3842a9470", 0x9B88: "000454eb842c9470",
        0x9BA8: "000454e3852b9470", 0x9BB8: "000454eb852b9470",
        0x9E08: "00044e94852f8e70", 0x9E58: "00044eef852f8e70",
        0x9F30: "00044eeb041c8e70", 0x9F58: "00044ef3050d8e70",
        0x9F80: "00044eeb040d8e70",
    },
}


# Encodings are derived only from identical instructions already present in
# the same frozen kernel.  hgobjdump must subsequently prove their assembly.
SUB_TASK_LO = bytes.fromhex("00200002000681b9")
SUB_TASK_HI = bytes.fromhex("00208002404601b9")
TRAMP_SUB_TASK_LO = bytes.fromhex("0020000c000681b9")
TRAMP_SUB_TASK_HI = bytes.fromhex("0020800c404601b9")
SUB_ROW_LO = bytes.fromhex("00200004800682b9")
SUB_ROW_HI = bytes.fromhex("00208004c04602b9")
SUB_CKPT_LO = bytes.fromhex("00200011008788b9")
SUB_CKPT_HI = bytes.fromhex("0020801140c708b9")
SHIFT_TASK_X16 = bytes.fromhex("04000000208181a3")
MOV_ROW_LO = bytes.fromhex("0000008080010250")
MOV_ROW_HI = bytes.fromhex("00000080c0410250")
CMP_FIRST = bytes.fromhex("008000020046d3bf")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_embedded_elf(data: bytes, offset: int):
    """Parse enough ELF64 metadata to fingerprint an embedded device image."""
    if offset + 64 > len(data) or data[offset:offset + 4] != b"\x7fELF":
        raise ValueError("not an ELF image")
    if data[offset + 4] != 2 or data[offset + 5] != 1:
        raise ValueError("expected little-endian ELF64")
    section_table = struct.unpack_from("<Q", data, offset + 0x28)[0]
    section_size = struct.unpack_from("<H", data, offset + 0x3A)[0]
    section_count = struct.unpack_from("<H", data, offset + 0x3C)[0]
    string_index = struct.unpack_from("<H", data, offset + 0x3E)[0]
    if section_size < 64 or not section_count or string_index >= section_count:
        raise ValueError("invalid section table")
    elf_size = section_table + section_size * section_count
    if elf_size <= 0 or offset + elf_size > len(data):
        raise ValueError("truncated embedded ELF")

    raw_sections = []
    for index in range(section_count):
        item = offset + section_table + index * section_size
        raw_sections.append(struct.unpack_from("<IIQQQQIIQQ", data, item))
    string_header = raw_sections[string_index]
    string_begin = offset + string_header[4]
    string_end = string_begin + string_header[5]
    if string_end > offset + elf_size:
        raise ValueError("invalid section-name table")
    names = data[string_begin:string_end]

    sections = []
    for fields in raw_sections:
        name_offset, section_type, flags, address, relative, size, _, _, align, _ = fields
        name_end = names.find(b"\0", name_offset)
        if name_end < 0:
            raise ValueError("unterminated section name")
        name = names[name_offset:name_end].decode("utf-8")
        begin = offset + relative
        end = begin + size
        if section_type != 8 and end > offset + elf_size:
            raise ValueError(f"section outside ELF: {name}")
        content = b"" if section_type == 8 else data[begin:end]
        sections.append({
            "name": name,
            "type": section_type,
            "flags": flags,
            "offset": relative,
            "size": size,
            "align": align,
            "content": content,
        })
    return elf_size, sections


def stable_section(section: dict) -> bool:
    name = section["name"]
    return (
        name in {".rodata", ".data", ".text", ".hg_info"}
        or name.startswith(".text.kernel.")
        or name.startswith(".hg_info.")
    )


def stable_fingerprint(sections) -> str:
    """Hash device code, constants and resource metadata, excluding provenance."""
    digest = hashlib.sha256()
    for section in sections:
        if not stable_section(section):
            continue
        digest.update(section["name"].encode())
        digest.update(b"\0")
        digest.update(struct.pack(
            "<IQQQ", section["type"], section["flags"],
            section["size"], section["align"],
        ))
        digest.update(section["content"])
    return digest.hexdigest()


def find_unique_embedded_elf(data: bytes, expected_fingerprint: str) -> dict:
    hits = []
    offset = 0
    while True:
        offset = data.find(b"ELF", offset)
        if offset < 0:
            break
        try:
            elf_size, sections = parse_embedded_elf(data, offset)
        except (ValueError, struct.error, UnicodeDecodeError):
            offset += 1
            continue
        fingerprint = stable_fingerprint(sections)
        if fingerprint == expected_fingerprint:
            hits.append({
                "elf_offset": offset,
                "elf_size": elf_size,
                "elf_sha256": sha256(data[offset:offset + elf_size]),
                "stable_fingerprint": fingerprint,
                "sections": sections,
            })
        offset += 1
    if len(hits) != 1:
        raise SystemExit(
            "embedded ELF semantic identity is not unique: "
            f"fingerprint={expected_fingerprint} hits="
            f"{[hex(hit['elf_offset']) for hit in hits]}"
        )
    return hits[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", type=Path,
        help="source-built low-register libdpa-chain.so",
    )
    parser.add_argument(
        "output", type=Path,
        help="output path for the patched library",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    original = args.input.read_bytes()
    actual_sha = sha256(original)
    data = bytearray(original)
    allowed = set()
    records = []

    def patch(arch: str, identity: dict, address: int, before: bytes,
              after: bytes, role: str, expected_asm: str) -> None:
        if len(before) != 8 or len(after) != 8:
            raise SystemExit(f"{arch} {address:#x}: non-instruction patch")
        absolute = identity["elf_offset"] + identity["text_offset"] + address
        actual = bytes(data[absolute:absolute + 8])
        if actual != before:
            raise SystemExit(
                f"{arch} {address:#x} mismatch: {actual.hex()} != {before.hex()}"
            )
        data[absolute:absolute + 8] = after
        allowed.update(range(absolute, absolute + 8))
        records.append({
            "arch": arch,
            "role": role,
            "isa_address": hex(address),
            "file_offset": hex(absolute),
            "before": before.hex(),
            "after": after.hex(),
            "expected_assembly": expected_asm,
        })

    resolved_offsets = {}
    resolved_identities = {}
    for arch, raw_identity in ARCH.items():
        identity = dict(raw_identity)
        resolved = find_unique_embedded_elf(
            original, identity["stable_fingerprint"]
        )
        target_sections = [
            section for section in resolved["sections"]
            if section["name"] == TARGET_TEXT_SECTION
        ]
        if len(target_sections) != 1:
            raise SystemExit(f"{arch} target text section count: {len(target_sections)}")
        target_text = target_sections[0]
        if target_text["offset"] != identity["expected_text_offset"]:
            raise SystemExit(
                f"{arch} target text offset changed: "
                f"{target_text['offset']:#x} != {identity['expected_text_offset']:#x}"
            )
        identity.update(resolved)
        identity["text_offset"] = target_text["offset"]
        resolved_offsets[arch] = identity["elf_offset"]
        resolved_identities[arch] = {
            "elf_offset": hex(identity["elf_offset"]),
            "elf_size": identity["elf_size"],
            "elf_sha256": identity["elf_sha256"],
            "reference_elf_sha256_match": (
                identity["elf_sha256"] == identity["reference_elf_sha256"]
            ),
            "stable_fingerprint": identity["stable_fingerprint"],
            "target_text_offset": hex(identity["text_offset"]),
        }

        entry = identity["entry_branch"]
        tramp = identity["trampoline"]
        sync = identity["sync"]
        init = identity["init"]
        tail = identity["tail"]

        patch(arch, identity, entry, NOP, cbr(entry, tramp),
              "enter_trampoline", f"s.cbr {tramp:#x}")
        for offset, instruction, asm in (
            (0x00, TRAMP_SUB_TASK_LO, "v.add.co.u32 vreg4, !sreg24, vreg24"),
            (0x08, TRAMP_SUB_TASK_HI, "v.add.ci.u32 vreg5, !sreg25, vreg25"),
            (0x10, SHIFT_TASK_X16, "v.shll.b64 vreg[6:7], vreg[4:5], 0x4"),
            (0x18, cbr(tramp + 0x18, sync), f"s.cbr {sync:#x}"),
        ):
            patch(arch, identity, tramp + offset, NOP, instruction,
                  "reverse_start_trampoline", asm)

        patch(arch, identity, init + 0x00,
              bytes.fromhex("00000004800782b9"), MOV_ROW_LO,
              "reuse_reverse_row_low", "v.mov.b32 vreg8, vreg6")
        patch(arch, identity, init + 0x08,
              bytes.fromhex("00008004804002b9"), MOV_ROW_HI,
              "reuse_reverse_row_high", "v.mov.b32 vreg9, vreg7")
        patch(arch, identity, init + 0x68,
              bytes.fromhex("04000000200282a3"), NOP,
              "remove_forward_row_shift", "s.nop")
        patch(arch, identity, init + 0x78,
              bytes.fromhex("00200004800682b9"), NOP,
              "remove_forward_row_decrement_low", "s.nop")
        patch(arch, identity, init + 0x80,
              bytes.fromhex("00208004c04602b9"), NOP,
              "remove_forward_row_decrement_high", "s.nop")

        original_tail = (
            bytes.fromhex("00000002000681b9"),
            bytes.fromhex("00008002404601b9"),
            bytes.fromhex("00000004800682b9"),
            bytes.fromhex("00008004c04602b9"),
            bytes.fromhex("00000011008788b9"),
            bytes.fromhex("0000801140c708b9"),
            bytes.fromhex("008000048041cfbf"),
            SHIFT_TASK_X16,
        )
        replacement_tail = (
            (CMP_FIRST, "v.cmp.gt.i64 sreg2, sreg[24:25], vreg[4:5]"),
            (SUB_TASK_LO, "v.add.co.u32 vreg4, !sreg24, vreg4"),
            (SUB_TASK_HI, "v.add.ci.u32 vreg5, !sreg25, vreg5"),
            (SUB_ROW_LO, "v.add.co.u32 vreg8, !sreg26, vreg8"),
            (SUB_ROW_HI, "v.add.ci.u32 vreg9, !sreg27, vreg9"),
            (SUB_CKPT_LO, "v.add.co.u32 vreg34, !sreg28, vreg34"),
            (SUB_CKPT_HI, "v.add.ci.u32 vreg35, !sreg29, vreg35"),
            (SHIFT_TASK_X16, "v.shll.b64 vreg[6:7], vreg[4:5], 0x4"),
        )
        for index, (before, replacement) in enumerate(zip(original_tail, replacement_tail)):
            after, asm = replacement
            patch(arch, identity, tail + index * 8, before, after,
                  "reverse_loop_tail", asm)

        for address, expected_hex in sorted(PHYSICAL_TAIL_MMA[arch].items()):
            patch(arch, identity, address, bytes.fromhex(expected_hex), NOP,
                  "remove_dead_physical_gradient_mma", "s.nop")

    changed = [i for i, (a, b) in enumerate(zip(original, data)) if a != b]
    if not set(changed) <= allowed:
        raise SystemExit("unexpected changed bytes outside audited instruction sites")
    if len(records) != 69:
        raise SystemExit(f"patch record count mismatch: {len(records)} != 69")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    manifest = {
        "status": "static-validation-required",
        "transformation": "fixed low-register fitting reverse traversal and dead physical-gradient tail prune",
        "input": str(args.input),
        "output": str(args.output),
        "input_sha256": actual_sha,
        "reference_full_file_sha256_match": actual_sha in REFERENCE_INPUT_SHA256S,
        "output_sha256": sha256(data),
        "architectures": list(ARCH),
        "embedded_elf_offsets": {k: hex(v) for k, v in resolved_offsets.items()},
        "embedded_elf_identities": resolved_identities,
        "semantic_identity_scope": [
            "all device text sections",
            "device constants and data",
            "all kernel resource metadata",
        ],
        "ignored_provenance_sections": [
            ".hg.build.commands",
            ".note.gnu.build-id",
            "debug, symbol and string tables",
        ],
        "instruction_patch_count": len(records),
        "changed_byte_count": len(changed),
        "invariants": [
            "same embedded ELF sizes and resource metadata",
            "same launch ABI and kernel symbol",
            "same per-task fitting arithmetic and accumulation order",
            "only independent grid-stride task traversal is reversed",
            "only the unconsumed descriptor-gradient tail MMA is removed",
        ],
        "patches": records,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "patches"}, indent=2))


if __name__ == "__main__":
    main()

