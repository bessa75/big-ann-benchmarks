"""
Integration tests for the DINO 10B dataset.

Tests all methods of DINO10BDataset at multiple scales, plus FaissIVF
algorithm validation on small slices. Uses a fixed query subset so
test time is approximately constant regardless of dataset size.

Unit tests (no data download) verify contract compliance with other
DatasetCompetitionFormat subclasses, attribute completeness, and
edge-case handling.

Usage:
    PYTHONPATH=. python tests/test_dino.py                # default sizes
    PYTHONPATH=. python tests/test_dino.py dino-1M        # single size
    PYTHONPATH=. python tests/test_dino.py dino-1M dino-1B
"""
import sys
import os
import math
import time

import numpy as np

from benchmark.datasets import DATASETS, DINO10BDataset
from benchmark.datasets import DatasetCompetitionFormat

DEFAULT_SIZES = ['dino-1M', 'dino-10M']

NQ_TEST = 1000
K = 10


def compute_recall(gt_ids, result_ids, k):
    nq = gt_ids.shape[0]
    total = 0
    for i in range(nq):
        total += len(set(gt_ids[i, :k]) & set(result_ids[i, :k]))
    return total / (nq * k)


# ---- Unit tests (no data download required) ----

def test_init():
    print("\n--- __init__ ---")
    for name, nb_M, expect_ds_fn in [
        ('dino-1M', 1, True),
        ('dino-10M', 10, True),
        ('dino-1B', 1000, True),
        ('dino-2B', 2000, True),
        ('dino-5B', 5000, False),
        ('dino-10B', 10000, False),
    ]:
        ds = DATASETS[name]()
        assert ds.nb == nb_M * 10**6
        assert ds.nb_M == nb_M
        assert ds.d == 1024
        assert ds.nq == 100_000
        assert ds.dtype == "uint8"
        assert ds.num_chunks == -(-ds.nb // ds.VECTORS_PER_CHUNK)
        assert ds.distance() == "euclidean"
        assert ds.search_type() == "knn"
        assert ds.data_type() == "dense"
        assert ds.default_count() == 10
        if expect_ds_fn:
            assert ds.ds_fn == "dino_base_%d.u8bin" % ds.nb
        else:
            assert ds.ds_fn is None
        print("  %s: OK (nb=%d, chunks=%d, ds_fn=%s)"
              % (name, ds.nb, ds.num_chunks, ds.ds_fn))
    print("  PASSED")


def test_chunk_url():
    print("\n--- _chunk_url ---")
    ds = DATASETS['dino-1M']()
    url = ds._chunk_url(0)
    assert url.endswith("/chunk_0000.bvecs")
    assert "dino_vitl_10B" in url
    url42 = ds._chunk_url(42)
    assert url42.endswith("/chunk_0042.bvecs")
    print("  OK")


def test_chunk_path():
    print("\n--- _chunk_path ---")
    ds = DATASETS['dino-1M']()
    d = "/tmp/chunks"
    assert ds._chunk_path(d, 0) == "/tmp/chunks/chunk_0000.bvecs"
    assert ds._chunk_path(d, 3) == "/tmp/chunks/chunk_0003.bvecs"
    assert ds._chunk_path(d, 0, 200_000_000) == \
        "/tmp/chunks/chunk_0000.bvecs"
    assert ds._chunk_path(d, 0, 1_000_000) == \
        "/tmp/chunks/chunk_0000.bvecs.crop_nb_1000000"
    print("  OK")


def test_short_name():
    print("\n--- short_name ---")
    ds = DATASETS['dino-1M']()
    assert "DINO10BDataset" in ds.short_name()
    assert "1000000" in ds.short_name()
    print("  OK: %s" % ds.short_name())


def test_get_dataset_fn_overflow():
    print("\n--- get_dataset_fn (uint32 overflow) ---")
    for name in ['dino-5B', 'dino-10B']:
        ds = DATASETS[name]()
        try:
            ds.get_dataset_fn()
            assert False, "should have raised"
        except RuntimeError as e:
            assert "uint32" in str(e)
            print("  %s: correctly raises RuntimeError" % name)
    print("  OK")


def test_str():
    """Verify __str__ doesn't crash and contains expected info."""
    print("\n--- __str__ ---")
    for name in ['dino-1M', 'dino-10B']:
        ds = DATASETS[name]()
        s = str(ds)
        assert "DINO10BDataset" in s
        assert "euclidean" in s
        assert "knn" in s
        assert str(ds.nq) in s
        assert str(ds.nb) in s
        print("  %s: OK (%s)" % (name, s))
    print("  PASSED")


def test_dataset_registry():
    """Verify all DINO variants are in DATASETS dict and instantiate."""
    print("\n--- DATASETS registry ---")
    expected = [
        'dino-1M', 'dino-10M', 'dino-100M', 'dino-1B',
        'dino-2B', 'dino-5B', 'dino-10B',
    ]
    for name in expected:
        assert name in DATASETS, "missing from DATASETS: %s" % name
        ds = DATASETS[name]()
        assert isinstance(ds, DINO10BDataset)
        assert isinstance(ds, DatasetCompetitionFormat)
        print("  %s: OK" % name)
    print("  PASSED")


def test_chunk_count():
    """Verify chunk count is correct for all sizes."""
    print("\n--- chunk count ---")
    for name, nb_M, expected_chunks in [
        ('dino-1M', 1, 1),
        ('dino-10M', 10, 1),
        ('dino-100M', 100, 1),
        ('dino-1B', 1000, 5),
        ('dino-2B', 2000, 10),
        ('dino-5B', 5000, 25),
        ('dino-10B', 10000, 50),
    ]:
        ds = DATASETS[name]()
        assert ds.num_chunks == expected_chunks, \
            "%s: expected %d chunks, got %d" \
            % (name, expected_chunks, ds.num_chunks)
        assert ds.num_chunks == math.ceil(ds.nb / ds.VECTORS_PER_CHUNK)
        print("  %s: %d chunks OK" % (name, ds.num_chunks))
    print("  PASSED")


def test_attribute_completeness():
    """Verify DINO sets all attributes that runner.py and algorithms access."""
    print("\n--- attribute completeness ---")
    ds = DATASETS['dino-1M']()

    required_attrs = [
        'nb', 'nb_M', 'd', 'nq', 'dtype', 'basedir', 'base_url',
        'num_chunks', 'gt_fn', 'qs_fn', 'ds_fn',
        'private_qs_url', 'private_gt_url', 'private_nq',
    ]
    for attr in required_attrs:
        assert hasattr(ds, attr), "missing attribute: %s" % attr
        print("  %s = %s" % (attr, repr(getattr(ds, attr))))

    assert ds.private_nq == 0
    assert ds.private_qs_url is None
    assert ds.private_gt_url is None
    print("  PASSED")


def test_interface_methods():
    """Verify DINO implements all Dataset interface methods without error."""
    print("\n--- interface methods ---")
    ds = DATASETS['dino-1M']()

    assert ds.distance() == "euclidean"
    assert ds.search_type() == "knn"
    assert ds.data_type() == "dense"
    assert ds.default_count() == 10
    assert isinstance(ds.short_name(), str)
    assert isinstance(str(ds), str)

    print("  distance: %s" % ds.distance())
    print("  search_type: %s" % ds.search_type())
    print("  data_type: %s" % ds.data_type())
    print("  default_count: %d" % ds.default_count())
    print("  PASSED")


def test_cross_dataset_contract():
    """Compare DINO interface with a known-good dataset (BigANNDataset)."""
    print("\n--- cross-dataset contract ---")
    ref_name = 'bigann-10M'
    dino_name = 'dino-1M'

    if ref_name not in DATASETS:
        print("  SKIPPED (bigann-10M not in DATASETS)")
        return

    ref = DATASETS[ref_name]()
    dino = DATASETS[dino_name]()

    interface_methods = [
        'prepare', 'get_dataset_fn', 'get_dataset',
        'get_dataset_iterator', 'get_queries', 'get_private_queries',
        'get_groundtruth', 'search_type', 'distance', 'data_type',
        'default_count', 'short_name', 'get_data_in_range',
    ]

    for method in interface_methods:
        assert hasattr(ref, method), \
            "reference missing %s" % method
        assert hasattr(dino, method), \
            "DINO missing %s" % method
        assert callable(getattr(dino, method)), \
            "DINO.%s not callable" % method
        print("  %s: OK" % method)

    required_attrs = ['nb', 'd', 'nq', 'dtype', 'basedir', 'ds_fn',
                      'gt_fn', 'private_nq']
    for attr in required_attrs:
        assert hasattr(ref, attr), "reference missing attr %s" % attr
        assert hasattr(dino, attr), "DINO missing attr %s" % attr
        print("  attr %s: OK (ref=%s, dino=%s)"
              % (attr, repr(getattr(ref, attr)), repr(getattr(dino, attr))))

    assert isinstance(dino.distance(), type(ref.distance()))
    assert dino.distance() in ("euclidean", "ip", "angular")
    assert isinstance(dino.search_type(), type(ref.search_type()))
    assert dino.search_type() in ("knn", "range", "knn_filtered")
    assert isinstance(dino.data_type(), type(ref.data_type()))
    assert dino.data_type() in ("dense", "sparse")
    print("  return types match reference")
    print("  PASSED")


def test_uint32_boundary():
    """Verify the uint32 header limit is exactly at 2^32 - 1."""
    print("\n--- uint32 boundary ---")
    limit = np.iinfo(np.uint32).max
    assert limit == 4294967295

    ds_below = DINO10BDataset(4294)
    assert ds_below.nb == 4294 * 10**6
    assert ds_below.nb <= limit
    assert ds_below.ds_fn is not None
    print("  4294M (%d): ds_fn=%s (below limit)"
          % (ds_below.nb, ds_below.ds_fn))

    ds_above = DINO10BDataset(4295)
    assert ds_above.nb == 4295 * 10**6
    assert ds_above.nb > limit
    assert ds_above.ds_fn is None
    print("  4295M (%d): ds_fn=None (above limit)" % ds_above.nb)

    print("  PASSED")


def test_basedir_structure():
    """Verify basedir and URL patterns are consistent."""
    print("\n--- basedir structure ---")
    ds = DATASETS['dino-1M']()
    assert "dino_vitl_10B" in ds.basedir
    assert ds.basedir.startswith("data/")
    assert "dino_vitl_10B" in ds.base_url
    assert ds.base_url.startswith("http")
    print("  basedir: %s" % ds.basedir)
    print("  base_url: %s" % ds.base_url)
    print("  PASSED")


def test_inheritance():
    """Verify DINO correctly inherits from DatasetCompetitionFormat."""
    print("\n--- inheritance ---")
    ds = DATASETS['dino-1M']()
    from benchmark.datasets import Dataset
    assert isinstance(ds, Dataset)
    assert isinstance(ds, DatasetCompetitionFormat)
    assert isinstance(ds, DINO10BDataset)
    assert not hasattr(ds, 'filtered')
    print("  class hierarchy: %s" % [c.__name__ for c in type(ds).__mro__])
    print("  PASSED")


def test_nb_consistency():
    """Verify nb_M * 10^6 == nb for all sizes."""
    print("\n--- nb consistency ---")
    all_sizes = [
        'dino-1M', 'dino-10M', 'dino-100M',
        'dino-1B', 'dino-2B', 'dino-5B', 'dino-10B',
    ]
    for name in all_sizes:
        ds = DATASETS[name]()
        assert ds.nb == ds.nb_M * 10**6, \
            "%s: nb=%d != nb_M*1e6=%d" % (name, ds.nb, ds.nb_M * 10**6)
    print("  PASSED")


def test_gt_fn_format():
    """Verify ground truth filename format is consistent with nb."""
    print("\n--- gt_fn format ---")
    for name in ['dino-1M', 'dino-10M', 'dino-1B', 'dino-10B']:
        ds = DATASETS[name]()
        expected = "gts_dino_patch_%d_k10.bin" % ds.nb
        assert ds.gt_fn == expected, \
            "%s: gt_fn=%s, expected %s" % (name, ds.gt_fn, expected)
        print("  %s: %s OK" % (name, ds.gt_fn))
    print("  PASSED")


def test_resolve_chunk_logic():
    """Test _resolve_chunk fallback behavior."""
    print("\n--- _resolve_chunk ---")
    import tempfile
    ds = DINO10BDataset(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        full = ds._chunk_path(tmpdir, 0)
        crop = ds._chunk_path(tmpdir, 0, ds.nb)
        assert full != crop

        open(full, 'w').close()
        assert ds._resolve_chunk(tmpdir, 0) == full

        os.remove(full)
        assert ds._resolve_chunk(tmpdir, 0) == crop

    print("  PASSED")


def test_vectors_per_chunk():
    """Verify VECTORS_PER_CHUNK constant is correct."""
    print("\n--- VECTORS_PER_CHUNK ---")
    assert DINO10BDataset.VECTORS_PER_CHUNK == 200_000_000
    ds_10b = DINO10BDataset(10000)
    assert ds_10b.num_chunks == 50
    assert ds_10b.num_chunks * ds_10b.VECTORS_PER_CHUNK == ds_10b.nb
    print("  PASSED")


def test_qs_fn_none_intentional():
    """qs_fn=None is intentional: DINO queries are bvecs, not xbin."""
    print("\n--- qs_fn=None (bvecs queries) ---")
    ds = DATASETS['dino-1M']()
    assert ds.qs_fn is None, "qs_fn should be None for DINO"
    assert hasattr(ds, 'get_queries'), "must have get_queries override"
    # The parent's get_queries() uses qs_fn with xbin_mmap,
    # but DINO overrides to use bvecs_mmap directly.
    print("  PASSED")


def test_prepare_signature_compat():
    """prepare(skip_data) must accept positional bool from main.py."""
    print("\n--- prepare signature ---")
    import inspect
    ds = DATASETS['dino-1M']()
    sig = inspect.signature(ds.prepare)
    params = list(sig.parameters.keys())
    assert 'skip_data' in params
    # main.py calls: dataset.prepare(args.neurips23track == 'none')
    # This passes True/False as positional arg for skip_data
    assert len(params) == 1, \
        "prepare should take only skip_data, got: %s" % params
    print("  PASSED")


def test_get_dataset_10m_boundary():
    """get_dataset() should work at exactly 10M (nb == 10^7)."""
    print("\n--- get_dataset boundary ---")
    ds = DINO10BDataset(10)
    assert ds.nb == 10**7
    # The parent's get_dataset() asserts nb <= 10**7
    # This should NOT raise for dino-10M
    # (actual call requires data, just verify the condition)
    assert ds.nb <= 10**7
    ds100 = DINO10BDataset(100)
    assert ds100.nb > 10**7
    # This SHOULD raise if called
    print("  PASSED")


def test_streaming_build_attrs():
    """Streaming track build() accesses ds.d and ds.dtype."""
    print("\n--- streaming build attrs ---")
    ds = DATASETS['dino-1M']()
    assert isinstance(ds.d, int) and ds.d > 0
    assert isinstance(ds.dtype, str)
    assert ds.dtype in ("float32", "uint8", "int8")
    # streaming/run.py: algo.setup(ds.dtype, max_pts, ndims)
    print("  d=%d, dtype=%s" % (ds.d, ds.dtype))
    print("  PASSED")


def test_d_to_bytes():
    """streaming/compute_gt.py uses ds.d.to_bytes(4, 'little')."""
    print("\n--- d.to_bytes ---")
    ds = DATASETS['dino-1M']()
    b = ds.d.to_bytes(4, byteorder='little')
    assert len(b) == 4
    recovered = int.from_bytes(b, byteorder='little')
    assert recovered == ds.d
    print("  PASSED")


def test_gt_k10_limit():
    """DINO GT has k=10 — verify default_count matches."""
    print("\n--- GT k=10 limit ---")
    ds = DATASETS['dino-1M']()
    assert ds.default_count() == 10
    assert "k10" in ds.gt_fn, \
        "gt_fn should contain k10: %s" % ds.gt_fn
    print("  default_count=%d, gt_fn=%s"
          % (ds.default_count(), ds.gt_fn))
    print("  PASSED")


def test_dtype_consistency():
    """All DINO data paths should return uint8 dtype."""
    print("\n--- dtype consistency ---")
    ds = DATASETS['dino-1M']()
    assert ds.dtype == "uint8"
    assert np.dtype(ds.dtype) == np.uint8
    print("  PASSED")


# ---- Integration tests (require data download) ----

def test_prepare(ds):
    print("  prepare()...", end=" ")
    t0 = time.time()
    ds.prepare()
    print("%.1f s" % (time.time() - t0))


def test_u8bin_file(ds):
    if ds.ds_fn is None:
        print("  u8bin file... SKIPPED (>4B vectors, chunked bvecs path)")
        return
    print("  u8bin file...", end=" ")
    fn = ds.get_dataset_fn()
    assert os.path.exists(fn), "missing: %s" % fn
    expected_size = 8 + ds.nb * ds.d
    actual_size = os.path.getsize(fn)
    assert actual_size == expected_size, \
        "size mismatch: expected %d, got %d" % (expected_size, actual_size)
    n, d = np.fromfile(fn, dtype='uint32', count=2)
    assert int(n) == ds.nb, "header n=%d, expected %d" % (n, ds.nb)
    assert int(d) == ds.d, "header d=%d, expected %d" % (d, ds.d)
    print("OK (%s, %d bytes)" % (fn, actual_size))


def test_queries(ds):
    print("  get_queries()...", end=" ")
    q = ds.get_queries()
    assert q.shape == (ds.nq, ds.d)
    assert q.dtype == np.uint8
    assert q.flags['C_CONTIGUOUS'], "queries not contiguous"
    print("OK %s %s" % (q.shape, q.dtype))
    return q


def test_groundtruth(ds):
    print("  get_groundtruth()...", end=" ")
    gt_ids, gt_dists = ds.get_groundtruth()
    assert gt_ids.shape == (ds.nq, K)
    assert gt_dists.shape == (ds.nq, K)
    assert gt_ids.dtype == np.int32
    assert gt_dists.dtype == np.float32
    for i in range(min(100, ds.nq)):
        assert np.all(gt_dists[i, :-1] <= gt_dists[i, 1:]), \
            "distances not sorted for query %d" % i
    print("OK %s" % (gt_ids.shape,))
    return gt_ids, gt_dists


def test_groundtruth_k(ds):
    """Test get_groundtruth with explicit k parameter."""
    print("  get_groundtruth(k=5)...", end=" ")
    gt_ids, gt_dists = ds.get_groundtruth(k=5)
    assert gt_ids.shape == (ds.nq, 5)
    assert gt_dists.shape == (ds.nq, 5)
    print("OK")


def test_iterator(ds):
    print("  get_dataset_iterator()...", end=" ")
    t0 = time.time()
    count = 0
    first_batch = None
    bs = min(ds.nb, 500_000)
    for batch in ds.get_dataset_iterator(bs=bs):
        assert batch.shape[1] == ds.d
        assert batch.flags['C_CONTIGUOUS'], \
            "iterator batch not contiguous at offset %d" % count
        if first_batch is None:
            first_batch = batch[:10].copy()
        count += batch.shape[0]
    assert count == ds.nb, \
        "iterator yielded %d, expected %d" % (count, ds.nb)
    print("OK (%d vectors, %.1f s)" % (count, time.time() - t0))
    return first_batch


def test_iterator_split(ds):
    """Test that sharded iteration covers the full dataset."""
    print("  get_dataset_iterator(split)...", end=" ")
    n_splits = min(3, ds.num_chunks + 1)
    total = 0
    for part in range(n_splits):
        part_count = 0
        for batch in ds.get_dataset_iterator(bs=500_000,
                                             split=(n_splits, part)):
            assert batch.shape[1] == ds.d
            assert batch.flags['C_CONTIGUOUS']
            part_count += batch.shape[0]
        total += part_count
    assert total == ds.nb, \
        "split iterator yielded %d, expected %d" % (total, ds.nb)
    print("OK (%d splits, %d total)" % (n_splits, total))


def test_data_in_range(ds, first_batch_from_iter=None):
    print("  get_data_in_range()...", end=" ")
    head = ds.get_data_in_range(0, 1000)
    assert head.shape == (1000, ds.d)
    assert head.dtype == np.uint8
    assert head.flags['C_CONTIGUOUS'], "head not contiguous"
    tail = ds.get_data_in_range(ds.nb - 100, ds.nb)
    assert tail.shape == (100, ds.d)
    assert tail.flags['C_CONTIGUOUS'], "tail not contiguous"
    mid_start = ds.nb // 2
    mid = ds.get_data_in_range(mid_start, mid_start + 500)
    assert mid.shape == (500, ds.d)
    assert mid.flags['C_CONTIGUOUS'], "mid not contiguous"
    if first_batch_from_iter is not None:
        head10 = ds.get_data_in_range(0, 10)
        assert np.array_equal(
            head10, first_batch_from_iter.astype(np.uint8)
        ) or np.allclose(head10, first_batch_from_iter), \
            "get_data_in_range and iterator disagree on first 10 vectors"
    single = ds.get_data_in_range(0, 1)
    assert single.shape == (1, ds.d)
    assert single.flags['C_CONTIGUOUS'], "single vector not contiguous"
    empty = ds.get_data_in_range(0, 0)
    assert empty.shape == (0, ds.d), \
        "empty range should return (0, d), got %s" % (empty.shape,)
    print("OK")


def test_get_dataset_small(ds):
    if ds.nb > 10**7:
        print("  get_dataset()... SKIPPED (nb=%d > 10M)" % ds.nb)
        return
    print("  get_dataset()...", end=" ")
    data = ds.get_dataset()
    assert data.shape == (ds.nb, ds.d)
    assert data.flags['C_CONTIGUOUS']
    print("OK %s %s" % (data.shape, data.dtype))


def test_algorithm(ds, dataset_name, queries, gt_ids):
    from benchmark.algorithms.faiss_inmem import FaissIVF

    nq = min(NQ_TEST, len(queries))
    q_sub = queries[:nq].astype(np.float32)
    gt_sub = gt_ids[:nq]

    print("  FaissIVF fit()...", end=" ")
    algo = FaissIVF("euclidean", 256)
    t0 = time.time()
    algo.fit(dataset_name)
    print("%.1f s" % (time.time() - t0))

    for nprobe in [1, 10]:
        algo.set_query_arguments(nprobe)
        t0 = time.time()
        algo.query(q_sub, K)
        elapsed = time.time() - t0
        result_ids = algo.get_results()
        recall = compute_recall(gt_sub, result_ids, K)
        print("  nprobe=%2d: recall@%d=%.4f, %d QPS (%.2f s, %d queries)"
              % (nprobe, K, recall, nq / elapsed, elapsed, nq))
        if nprobe == 10:
            assert recall > 0.8, "recall too low: %.4f" % recall

    algo.done()


def test_dataset(dataset_name):
    print("\n=== %s ===" % dataset_name)
    ds = DATASETS[dataset_name]()
    print("  nb=%d, d=%d, nq=%d, chunks=%d, ds_fn=%s"
          % (ds.nb, ds.d, ds.nq, ds.num_chunks, ds.ds_fn))

    test_prepare(ds)
    test_u8bin_file(ds)
    queries = test_queries(ds)
    gt_ids, gt_dists = test_groundtruth(ds)
    test_groundtruth_k(ds)
    first_batch = test_iterator(ds)
    test_iterator_split(ds)
    test_data_in_range(ds, first_batch)
    test_get_dataset_small(ds)

    if ds.nb <= 10**7:
        test_algorithm(ds, dataset_name, queries, gt_ids)
    else:
        print("  (skipping algorithm test for nb=%d)" % ds.nb)

    print("  PASSED")


def main():
    sizes = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_SIZES

    for name in sizes:
        assert name in DATASETS, "unknown dataset: %s" % name

    print("=== DINO unit tests ===")
    test_init()
    test_chunk_url()
    test_chunk_path()
    test_short_name()
    test_get_dataset_fn_overflow()
    test_str()
    test_dataset_registry()
    test_chunk_count()
    test_attribute_completeness()
    test_interface_methods()
    test_cross_dataset_contract()
    test_uint32_boundary()
    test_basedir_structure()
    test_inheritance()
    test_nb_consistency()
    test_gt_fn_format()
    test_resolve_chunk_logic()
    test_vectors_per_chunk()
    test_qs_fn_none_intentional()
    test_prepare_signature_compat()
    test_get_dataset_10m_boundary()
    test_streaming_build_attrs()
    test_d_to_bytes()
    test_gt_k10_limit()
    test_dtype_consistency()
    print("\nUnit tests passed.")

    print("\n=== DINO integration tests ===")
    for name in sizes:
        test_dataset(name)

    print("\n=== All tests passed ===")


if __name__ == "__main__":
    main()
