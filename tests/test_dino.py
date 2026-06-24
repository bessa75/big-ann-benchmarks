"""
Integration tests for the DINO 10B dataset.

Tests all methods of DINO10BDataset at multiple scales, plus FaissIVF
algorithm validation on small slices. Uses a fixed query subset so
test time is approximately constant regardless of dataset size.

Usage:
    PYTHONPATH=. python tests/test_dino.py                # default sizes
    PYTHONPATH=. python tests/test_dino.py dino-1M        # single size
    PYTHONPATH=. python tests/test_dino.py dino-1M dino-1B
"""
import sys
import os
import time

import numpy as np

from benchmark.datasets import DATASETS

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


def test_iterator(ds):
    print("  get_dataset_iterator()...", end=" ")
    t0 = time.time()
    count = 0
    first_batch = None
    bs = min(ds.nb, 500_000)
    for batch in ds.get_dataset_iterator(bs=bs):
        assert batch.shape[1] == ds.d
        if first_batch is None:
            first_batch = batch[:10].copy()
        count += batch.shape[0]
    assert count == ds.nb, \
        "iterator yielded %d, expected %d" % (count, ds.nb)
    print("OK (%d vectors, %.1f s)" % (count, time.time() - t0))
    return first_batch


def test_data_in_range(ds, first_batch_from_iter=None):
    print("  get_data_in_range()...", end=" ")
    head = ds.get_data_in_range(0, 1000)
    assert head.shape == (1000, ds.d)
    assert head.dtype == np.uint8
    tail = ds.get_data_in_range(ds.nb - 100, ds.nb)
    assert tail.shape == (100, ds.d)
    mid_start = ds.nb // 2
    mid = ds.get_data_in_range(mid_start, mid_start + 500)
    assert mid.shape == (500, ds.d)
    if first_batch_from_iter is not None:
        head10 = ds.get_data_in_range(0, 10)
        assert np.array_equal(
            head10, first_batch_from_iter.astype(np.uint8)
        ) or np.allclose(head10, first_batch_from_iter), \
            "get_data_in_range and iterator disagree on first 10 vectors"
    print("OK")


def test_get_dataset_small(ds):
    if ds.nb > 10**7:
        print("  get_dataset()... SKIPPED (nb=%d > 10M)" % ds.nb)
        return
    print("  get_dataset()...", end=" ")
    data = ds.get_dataset()
    assert data.shape == (ds.nb, ds.d)
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
    first_batch = test_iterator(ds)
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
    print("\nUnit tests passed.")

    print("\n=== DINO integration tests ===")
    for name in sizes:
        test_dataset(name)

    print("\n=== All tests passed ===")


if __name__ == "__main__":
    main()
