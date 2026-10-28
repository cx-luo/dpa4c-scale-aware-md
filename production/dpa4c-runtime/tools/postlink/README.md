# Post-link policy

The low-register fitting kernel is compiled from owned source. The fixed compiler
keeps the required register allocation only with forward grid-stride traversal, so
the build applies one fail-closed device-ISA transformation before packaging.

The patch_lowreg_reverse.py tool locates one exact PPU 1.0 and one exact PPU 1.5
device image by a stable fingerprint over all device code, constants and kernel
resource metadata. It deliberately excludes compiler command provenance, build-id,
debug, symbol and string tables. The tool then verifies every original instruction
byte, changes only the audited 69-instruction whitelist (36 traversal sites and 33
dead physical-gradient MMA sites), and records the full ELF input/output hashes, semantic identities and offsets. Any device code, resource or
instruction before-image change aborts the build.

The subsequent `patch_unit_seed_fixed_isa.py` transform targets the already
validated production low-register image. It removes two guarded FP64 seed loads, waits and
conversions and materializes exact FP32 one in the existing scale registers.
The host path admits this image only for a device-pointer range registered by
the direct executor; all unregistered seeds retain the generic fallback.

This is a build-time
transformation; the runtime never makes executable memory writable. ELF packaging
then sets SONAME and origin-relative RUNPATH as usual.
