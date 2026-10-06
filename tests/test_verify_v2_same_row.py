"""Independent same-row budget oracles and bounded native byte regressions.

Coordinate enumeration checks coverage/order and raw-chunk accounting only;
the separate real-HDF5 fixture checks endian, signed-zero and NaN payload bits.
No planner, writer, timing threshold, or product tiling helper supplies an oracle.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from h5carry.model import CarryError, Limits, fingerprint
from tests.test_verify_v2 import box, manual_copy, manual_plan, request


def coordinates(selection):
    return list(itertools.product(*(
        range(index.start, index.stop) if isinstance(index, slice) else (index,)
        for index in selection
    )))


def raw_exposure(selection, chunks, itemsize):
    """Count the explicit set of touched chunk coordinates, not a span formula."""
    if chunks is None:
        return 0
    touched = {
        tuple(value // chunk for value, chunk in zip(coordinate, chunks))
        for coordinate in coordinates(selection)
    }
    return len(touched) * math.prod(chunks) * itemsize


class SameRowVerifierTests(unittest.TestCase):
    def setUp(self):
        if os.environ.get("H5CARRY_TEST_NATIVE_CHILD") != "1":
            raise RuntimeError("Native fixtures require python -m tests.native_runner unittest ...")
        import h5py
        import numpy as np
        from h5carry import verify_v2
        self.h5, self.np, self.verifier = h5py, np, verify_v2
        self.tmp = tempfile.TemporaryDirectory(prefix="h5carry-same-row-")
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "source.h5"
        self.output = Path(self.tmp.name) / "output.h5"

    def check_coordinates(self, shape, origin, source_chunks, output_chunks, itemsize, cap):
        must_refuse = itemsize > cap or any(
            chunks is not None and math.prod(chunks) * itemsize > cap
            for chunks in (source_chunks, output_chunks)
        )
        iterator = self.verifier._tiles(shape, itemsize, cap, origin, source_chunks, output_chunks)
        if must_refuse:
            with self.assertRaises(CarryError) as caught:
                next(iterator)
            self.assertEqual(caught.exception.code, "RESOURCE")
            return
        expected = list(itertools.product(*(range(n) for n in shape)))
        seen = []
        for left, right in iterator:
            a, b = coordinates(left), coordinates(right)
            self.assertTrue(all(isinstance(index, int) for index in left[:-1] + right[:-1]))
            self.assertEqual(a, [tuple(value + offset for value, offset in zip(q, origin)) for q in b])
            self.assertGreater(len(b), 0)
            self.assertLessEqual(len(b) * itemsize, cap)
            self.assertLessEqual(raw_exposure(left, source_chunks, itemsize), cap)
            self.assertLessEqual(raw_exposure(right, output_chunks, itemsize), cap)
            seen.extend(b)
            if right[-1].stop < shape[-1]:
                left_more = left[:-1] + (slice(left[-1].start, left[-1].stop + 1),)
                right_more = right[:-1] + (slice(right[-1].start, right[-1].stop + 1),)
                self.assertTrue(
                    (len(b) + 1) * itemsize > cap
                    or raw_exposure(left_more, source_chunks, itemsize) > cap
                    or raw_exposure(right_more, output_chunks, itemsize) > cap,
                    "one more same-row element must exceed a comparison budget",
                )
        self.assertEqual(seen, expected)
        # Opaque coordinate encoding proves concatenation order, not native bytes.
        self.assertEqual(
            b"".join(hashlib.sha256(repr(q).encode()).digest()[:itemsize] for q in seen),
            b"".join(hashlib.sha256(repr(q).encode()).digest()[:itemsize] for q in expected),
        )

    def test_independent_coordinate_and_aggregate_cap_oracle(self):
        rng = random.Random(770119)
        for _ in range(2500):
            rank = rng.randrange(1, 5)
            shape = tuple(rng.randrange(1, 5) for _ in range(rank))
            origin = tuple(rng.randrange(0, 7) for _ in shape)
            source = None if rng.randrange(3) == 0 else tuple(rng.randrange(1, 5) for _ in shape)
            output = None if rng.randrange(3) == 0 else tuple(rng.randrange(1, 5) for _ in shape)
            self.check_coordinates(shape, origin, source, output,
                                   rng.choice((1, 2, 4, 8, 16)), rng.randrange(1, 1025))
        for case in (
            ((3, 19), (1, 3), (2, 4), (3, 7), 4, 169),
            ((4, 41), (3, 5), (2, 6), (4, 3), 8, 385),
            ((2, 3, 37), (1, 4, 9), (2, 2, 5), (1, 3, 7), 4, 253),
            ((11,), (5,), (3,), (4,), 8, 32),
            ((4, 33), (0, 1), None, (4, 2), 4, 65),
            ((4, 33), (0, 1), (2, 4), None, 4, 65),
            ((2, 16), (1, 8), (2, 8), (1, 4), 8, 256),
            ((2, 5), (1, 6), (2, 8), (2, 5), 8, 128),
        ):
            self.check_coordinates(*case)

    def test_empty_scalar_rank32_and_huge_prefix_are_lazy(self):
        tiles = self.verifier._tiles
        for shape in ((0,), (0, 4), (3, 0), (1, 0, 1), (0,) + (1,) * 31):
            self.assertEqual(list(tiles(shape, 8, 1, (0,) * len(shape), (100,) * len(shape))), [])
        self.assertEqual(list(tiles(None, 8, 1)), [])
        self.assertEqual(list(tiles((), 8, 8)), [((), ())])
        with self.assertRaises(CarryError) as caught:
            list(tiles((), 8, 7))
        self.assertEqual(caught.exception.code, "RESOURCE")
        self.check_coordinates((1,) * 29 + (2, 3, 13), (0,) * 29 + (4, 5, 2),
                               (1,) * 29 + (2, 3, 4), (1,) * 29 + (1, 2, 5), 4, 192)
        huge = tiles(((1 << 63) - 1, 3, 15), 4, 65,
                     (9, 1, 3), (1, 1, 3), (1, 2, 2))
        first = list(itertools.islice(huge, 8))
        self.assertEqual(len(first), 8)
        self.assertEqual([right[:-1] for _, right in first],
                         [(row, column) for row in range(2) for column in range(3)
                          for _ in range(2)][:8])
        # Four output chunks fit (4 * 16 bytes), allowing eight columns per read.
        for position, (left, right) in enumerate(first):
            self.assertEqual(left[:-1], (right[0] + 9, right[1] + 1))
            self.assertEqual((left[-1], right[-1]),
                             (slice(3, 11), slice(0, 8)) if position % 2 == 0
                             else (slice(11, 18), slice(8, 15)))
            self.assertLessEqual(raw_exposure(left, (1, 1, 3), 4), 65)
            self.assertLessEqual(raw_exposure(right, (1, 2, 2), 4), 65)

    def test_maximum_payload_four_pass_exact_read_count(self):
        counts, maxima = [], []
        for origin, output_chunks, reads_per_tile in (
            ((1, 0), None, 1), ((0, 0), None, 1), ((1, 0), (32, 256), 2),
        ):
            count = maximum_logical = maximum_raw = 0
            for left, right in self.verifier._tiles(
                    (65536, 1024), 4, 1 << 20, origin, (32, 256), output_chunks):
                count += 1
                maximum_logical = max(maximum_logical, (right[-1].stop - right[-1].start) * 4)
                chunk_columns = {column // 256 for column in range(left[-1].start, left[-1].stop, 256)}
                maximum_raw = max(maximum_raw, len(chunk_columns) * 32 * 256 * 4)
            counts.append(count * reads_per_tile)
            maxima.append((maximum_logical, maximum_raw))
        self.assertEqual(counts, [65536, 65536, 131072])
        self.assertEqual(sum(counts), 4 * 65536)
        self.assertEqual(maxima, [(4096, 131072)] * 3)

    def native_fixture(self):
        self.limits = Limits(chunk_bytes=256)
        bits = self.np.full((3, 16), 0x3FF0000000000000, dtype=">u8")
        bits[1, 6:11] = [0, 0x8000000000000000, 0x7FF8000000000011,
                         0x7FF8000000000042, 0x3FF0000000000000]
        bits[2, 6:11] = [0x8000000000000000, 0, 0x7FF8000000000042,
                         0x7FF8000000000011, 0xBFF0000000000000]
        self.source_bits, self.output_bits = bits, bits[1:3, 6:11].copy()
        with self.h5.File(self.source, "w") as source:
            source.create_dataset("A", data=bits.view(">f8"), chunks=(2, 8))
        paths = ["/", "/A"]
        boxes = {"/A": box("", [1, 6], [3, 11])["selection"]}
        manual_copy(self.source, self.output, paths, boxes)
        self.plan = manual_plan(self.source, self.output, request([box("/A", [1, 6], [3, 11])]),
                                paths, boxes, self.limits)
        with self.h5.File(self.source, "r") as source, self.h5.File(self.output, "r") as output:
            self.assertEqual(source["A"].dtype.str, ">f8")
            self.assertEqual(output["A"].dtype.str, ">f8")
            self.assertEqual(source["A"].chunks, (2, 8))
            self.assertEqual(output["A"].chunks, (2, 5))
            self.assertEqual(source["A"][()].tobytes(), self.source_bits.tobytes())
            self.assertEqual(output["A"][()].tobytes(), self.output_bits.tobytes())

    def test_native_big_endian_bits_and_all_four_passes_use_eight_reads(self):
        self.native_fixture()
        before = fingerprint(self.source)
        original = self.h5.Dataset.__getitem__
        reads = []

        def capture(dataset, selection):
            value = original(dataset, selection)
            source = Path(dataset.file.filename) == self.source
            expected = self.source_bits if source else self.output_bits
            self.assertEqual(self.np.asarray(value, dtype=dataset.dtype).tobytes(), expected[selection].tobytes())
            reads.append(("source" if source else "output", value.nbytes,
                          raw_exposure(selection, dataset.chunks, dataset.dtype.itemsize)))
            return value

        with patch.object(self.h5.Dataset, "__getitem__", capture):
            result = self.verifier.verify_export(self.source, self.output, self.plan, self.limits)
        self.assertEqual(result["status"], "verified", result)
        self.assertTrue(result["coverage"]["source_equality"])
        self.assertEqual(result["coverage"]["data_blocks"], 8)
        self.assertEqual(reads, [("source", 40, 256)] * 2 + [("output", 40, 80)] * 2
                         + [("source", 40, 256), ("output", 40, 80)] * 2)
        self.assertEqual(result["coverage"]["max_read_bytes"], 40)
        self.assertEqual(result["coverage"]["max_native_expansion_bytes"], 256)
        self.assertEqual(fingerprint(self.source), before)
        self.assertEqual(self.plan["graph"]["objects"][1]["metadata"]["payload_sha256"],
                         hashlib.sha256(self.output_bits.tobytes()).hexdigest())

    def check_native_mutation(self, index, replacement):
        self.native_fixture()
        before = fingerprint(self.source)
        mutated = self.output_bits.copy()
        mutated[index] = replacement
        self.assertNotEqual(mutated.tobytes(), self.output_bits.tobytes())
        with self.h5.File(self.output, "r+") as output:
            output["A"][()] = mutated.view(">f8")
            self.assertEqual(output["A"][()].tobytes(), mutated.tobytes())
        result = self.verifier.verify_export(self.source, self.output, self.plan, self.limits)
        self.assertEqual(result["status"], "mismatch", result)
        self.assertFalse(result["coverage"]["source_equality"])
        self.assertTrue(any(item["code"] == "MISMATCH" for item in result["diagnostics"]))
        self.assertEqual(fingerprint(self.source), before)

    def test_native_signed_zero_bit_change_mismatches(self):
        self.check_native_mutation((0, 1), 0)

    def test_native_nan_payload_bit_change_mismatches(self):
        self.check_native_mutation((0, 2), 0x7FF8000000000012)

    def test_aggregate_raw_oversized_chunk_and_never_refuse_before_getitem(self):
        limits = Limits(chunk_bytes=32)
        with self.h5.File(self.source, "w") as source:
            source.create_dataset("aggregate", shape=(6,), dtype="u8", chunks=(3,))
            source.create_dataset("oversized", shape=(5,), dtype="u8", chunks=(5,))
            dcpl = self.h5.h5p.create(self.h5.h5p.DATASET_CREATE)
            dcpl.set_fill_time(self.h5.h5d.FILL_TIME_NEVER)
            self.h5.h5d.create(source.id, b"never", self.h5.h5t.STD_U64BE,
                              self.h5.h5s.create_simple((5,)), dcpl=dcpl)
            coverage = self.verifier._coverage()
            inventory = self.verifier._Inventory(source, limits, coverage)
            self.assertEqual(raw_exposure((slice(2, 4),), (3,), 8), 48)
            for name, selection, code in (
                ("aggregate", (slice(2, 4),), "RESOURCE"),
                ("oversized", (slice(0, 1),), "RESOURCE"),
                ("never", (slice(0, 1),), "UNSUPPORTED"),
            ):
                with self.subTest(dataset=name):
                    with patch.object(self.h5.Dataset, "__getitem__",
                                      side_effect=AssertionError("refusal must precede native payload read")) as getitem:
                        with self.assertRaises(CarryError) as caught:
                            inventory.read(source[name], selection)
                        self.assertEqual(caught.exception.code, code)
                        getitem.assert_not_called()
            self.assertEqual(coverage["data_blocks"], 0)


if __name__ == "__main__":
    unittest.main()
