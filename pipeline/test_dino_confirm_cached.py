import tempfile
import unittest
from pathlib import Path
import gc

import numpy as np

from dataclasses import asdict

from dino_confirm_cached import _cached_confirmation_task, _init_confirmation_worker
from near_duplicates import AuditConfig, NormalizedRetina, classify_pair, confirm_pair


class CachedConfirmationTests(unittest.TestCase):
    def _release_globals(self):
        import dino_confirm_cached as module
        module._CACHE_GRAY = None
        module._CACHE_MASK = None
        module._CACHE_CONFIG = None
        gc.collect()

    def test_worker_opens_exact_memmap_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gray = np.lib.format.open_memmap(
                root / "gray.npy", mode="w+", dtype=np.float32, shape=(2, 4, 4))
            mask = np.lib.format.open_memmap(
                root / "mask.npy", mode="w+", dtype=np.bool_, shape=(2, 4, 4))
            gray[1] = np.arange(16, dtype=np.float32).reshape(4, 4)
            mask[1, 1:3, 1:3] = True
            gray.flush()
            mask.flush()
            del gray, mask
            _init_confirmation_worker(
                str(root / "gray.npy"), str(root / "mask.npy"),
                {"schema_version": 3, "hash_bits": 256})
            import dino_confirm_cached as module
            np.testing.assert_array_equal(
                module._CACHE_GRAY[1], np.arange(16, dtype=np.float32).reshape(4, 4))
            self.assertEqual(int(module._CACHE_MASK[1].sum()), 4)
            self._release_globals()

    def test_cached_confirmation_matches_direct_primitive(self):
        config = AuditConfig.from_json(Path(__file__).with_name(
            "near_duplicate_config.json"))
        size = int(config.normalized_size)
        yy, xx = np.mgrid[:size, :size]
        mask = ((xx - size / 2) ** 2 + (yy - size / 2) ** 2) < (size * .42) ** 2
        gray = np.zeros((size, size), dtype=np.float32)
        gray[mask] = (.25 + .65 * xx[mask] / size).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gray_map = np.lib.format.open_memmap(
                root / "gray.npy", mode="w+", dtype=np.float32,
                shape=(2, size, size))
            mask_map = np.lib.format.open_memmap(
                root / "mask.npy", mode="w+", dtype=np.bool_,
                shape=(2, size, size))
            gray_map[0] = gray
            gray_map[1] = gray
            mask_map[0] = mask
            mask_map[1] = mask
            gray_map.flush()
            mask_map.flush()
            del gray_map, mask_map
            _init_confirmation_worker(
                str(root / "gray.npy"), str(root / "mask.npy"), config.values)
            empty = np.empty((0,), dtype=np.uint8)
            direct = confirm_pair(
                NormalizedRetina(empty, gray, mask, (0, 0, size, size), .5),
                NormalizedRetina(empty, gray, mask, (0, 0, size, size), .5),
                config,
            )
            _, cached, cached_class = _cached_confirmation_task(("pair", 0, 1))
            self.assertEqual(asdict(direct), cached)
            self.assertEqual(cached_class, classify_pair(direct, config.confirmation))
            self._release_globals()


if __name__ == "__main__":
    unittest.main()
