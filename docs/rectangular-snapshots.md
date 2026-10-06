# Rectangular snapshots (plan version 2)

H5Carry 0.2.0a1 adds a narrow, declared transformation of trusted, quiescent local HDF5 files. It does not infer scientific coordinates or validate netCDF, CF, NWB or NeXus. Unchanged ordinary attributes may become scientifically invalid after a crop; review and repair them in the owning domain workflow.

Use `h5carry plan source.h5 --selection-json selection.json --out plan.json`. The option is mutually exclusive with repeated `--select`. Export, verify and inspect use their existing commands. Existing `--select` produces version-1 whole-object plans; the new option produces version 2 even when every requested object is whole. Old H5Carry binaries reject v2 plans.

## Exact request

```json
{
  "format": "h5carry-selection",
  "version": 1,
  "objects": [
    {"path": "/A", "selection": {"kind": "box", "start": [1, 1], "stop": [4, 3]}},
    {"path": "/B", "selection": {"kind": "box", "start": [1, 1], "stop": [4, 3]}},
    {"path": "/calibration", "selection": {"kind": "whole"}}
  ],
  "scale_mappings": [
    {"consumer": "/A", "axis": 0, "scale": "/time", "mapping": "index"},
    {"consumer": "/A", "axis": 1, "scale": "/channel", "mapping": "index"},
    {"consumer": "/B", "axis": 0, "scale": "/time", "mapping": "index"},
    {"consumer": "/B", "axis": 1, "scale": "/channel", "mapping": "index"}
  ],
  "storage_policy": "fixed-snapshot-v1"
}
```

JSON is bounded UTF-8 with exact keys, versions and integer types. Duplicate keys, paths or mapping assertions are errors. Bounds must satisfy `0 <= start <= stop <= source_extent`; stops are exclusive and vectors match source rank. There is no clipping, negative indexing, implicit endpoint, step field, squeeze, expression, point list, union or value-coordinate query. Integers are at most `2**63-1`. Rank, request size and other limits may only be lowered.

Only proven hard-linked simple nonscalar datasets of currently supported fixed primitive types accept boxes. References, scalars and null dataspaces remain whole-only. Canonical h5py complex and Boolean representations retain their existing narrow exceptions; arbitrary compound types remain unsupported. Full-shape primitive boxes normalize to whole, preserving original creation properties. Empty boxes retain rank.

## One source identity, one selection

Retained hard aliases name the same output object. Aliases requesting different intervals conflict even when output shapes or values are equal. Whole and proper-subset requirements conflict. There is no implicit cloning, union, intersection or override precedence.

Every scale attachment of a cropped consumer needs one explicit `index` mapping. The mapping is the requester's assertion of positional correspondence. H5Carry checks rank one and equal original axis length; these necessary structural checks cannot prove the assertion scientifically. Multiple scales per axis and one scale on multiple axes are admitted only when all induced original-coordinate intervals agree. A supplied mapping on a whole consumer must also satisfy the assertion's rank/length checks. Whole consumers without mappings retain legal non-positional-scale behavior.

Duplicate mapping identities through hard aliases are rejected. A mapping must identify an actual retained attachment via source hard paths. Display names do not identify scales. Reverse scale metadata is rebuilt from retained consumers only; an excluded sibling that shares a scale remains excluded.

Ordinary object references require whole targets. A retained reference into a cropped target therefore refuses. References from cropped data to unchanged calibration objects are remapped normally, including null references. References on root/ancestor groups participate in closure. Referencing a group requires its descendants whole and can conflict with a separately requested crop. Whole soft links preserve their original closure rules; soft-path boxes are refused.

## Explicit storage change

For every proper crop, including induced scale crops:

- Output shape and maximum shape are the crop extent, a fixed snapshot
- Contiguous layout stays contiguous
- Chunked layout stays chunked with each extent `min(old_chunk, max(1, output_extent))`
- Exact qualified low-level dtype, defined selected element bytes, filter pipeline, fill and other checked creation policies are retained

Plan v2 stores the canonical original request, whole-file source fingerprint, separate source descriptors, normalized selections, expected output graph/data digests, inclusion reasons and a transformation list. Source extents cannot be disguised as output extents. Reasons explain planning; they are not equality evidence.

HDF5 chunk-option flags require a separate native query because the qualified H5Pequal implementations do not compare them. Proper crops with nondefault flags are conservatively refused: resetting chunk geometry can silently clear them. Whole datasets preserve the flags, and verification compares them explicitly. The read-only query is tied to the exact two qualified h5py/HDF5 builds, uses the already loaded extension under its lock and fails closed if unavailable. No other library is searched for.

The snapshot does not preserve physical file layout, file size, compression ratio, object addresses, timestamps or actual attribute insertion order. Native creation-policy flags are checked; this does not promise historical provenance. Clamped chunk dimensions are deterministic, not claimed optimal.

## Resources and truth boundaries

Existing application ceilings remain unchanged: 10,000 objects, 50,000 edges, 8 MiB serialized plans, 256 MiB selected payload, 1 MiB logical read blocks, 256 KiB attributes, rank 32, path depth 64, 1,024-byte paths and 60-second discovery. Native children retain their existing 2 GiB address-space, CPU, wall-time and written-file caps.

Typed dataset payload reads additionally limit uncompressed native chunk exposure to the same `chunk_bytes` cap. A source chunk whose dimensions times dtype width exceed that cap refuses before a nonempty read. Planner/writer row tiles stop at original chunk boundaries. The independent verifier separately translates coordinates and coalesces adjacent segments within each C-order row only while both logical bytes and aggregate touched native-chunk exposure stay within the cap for both source and output. It does not share planner tiling, and its independent pre-read guard rechecks the actual selection. There are no per-index lists proportional to a rectangle. Empty selections do no element-payload reads. Source reference-array inventory reads are guarded even when their owners will later be excluded.

Typed file opens disable per-dataset raw chunk caches. A 1 MiB logical or expanded-chunk cap is not a 1 MiB process-memory promise: native metadata, open objects and other allocations remain, and process caps may stop a workload. Repeated small reads can decode the same chunk repeatedly. Separate ceiling calibrations do not establish simultaneous-maxima completion or predictable command duration.

Whole-file fingerprinting still reads the whole source. Full namespace/reference validation can dominate small selections and may reject an unsupported unselected object. There is no subset-only-I/O, terabyte-scale, speed or physical-file-size reduction claim. In the original small feasibility fixture the cropped file was larger despite fewer logical values.

Source-bound verification independently derives closure, original intervals, scale/ref/alias identities and storage transformation from source plus the original request. It checks both the plan and reopened output; coordinated plan/output omissions or changes do not bypass expected source truth. It compares exact byte representations, including signed zero and supported NaN payloads. It does not prove that the user-provided positional assertion is scientifically true. Output-only inspection proves neither source equality nor that assertion.

Keep all writers closed. Fingerprints detect observed changes but cannot prove quiescence, and native process controls are not a malicious-file sanitizer or filesystem sandbox. Destination plan/output/report paths are distinct and no-clobber. Exported output and report are individually published, not an atomic pair; a failed report publication can leave the verified output available.

## Existing alternatives

The original 23-line known-schema h5py recipe already solves the two-consumer/shared-scale/alias crop. Use such a maintained recipe when it fits a stable schema. NCO, xarray and NeXpy offer mature schema-aware subsetting. This extension integrates a reusable request, inspectable dependency plan, conservative conflict policy and independent source-bound check; it is not a novel extraction algorithm or a demonstrated performance/adoption advantage.
