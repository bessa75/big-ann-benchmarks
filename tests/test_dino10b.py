"""
Comprehensive test suite for DINO10BDataset.

Tests are split into two parts:
- Synthetic data tests: run anywhere, no downloads needed
- Live data tests: require real queries + GT files (skipped if not present)

Run:
    cd /path/to/big-ann-benchmarks
    PYTHONPATH="." python tests/test_dino10b.py
"""

import os
import sys
import shutil
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.datasets import (
    DINO10BDataset, DATASETS, Dataset, DatasetCompetitionFormat,
)
from benchmark.dataset_io import bvecs_mmap, knn_result_read, sanitize


PASS = 0
FAIL = 0


def check(condition, msg):
    global PASS, FAIL
    if condition:
        PASS += 1
        print("  PASS: %s" % msg)
    else:
        FAIL += 1
        print("  FAIL: %s" % msg)


def write_bvecs(fname, data):
    """Write uint8 array to .bvecs format."""
    n, d = data.shape
    with open(fname, "wb") as f:
        for i in range(n):
            np.array([d], dtype='<i4').tofile(f)
            data[i].tofile(f)


def write_gt_bin(fname, I, D):
    """Write GT in competition .bin format."""
    nq, k = I.shape
    with open(fname, "wb") as f:
        np.array([nq, k], dtype='uint32').tofile(f)
        I.astype('int32').tofile(f)
        D.astype('float32').tofile(f)


# =========================================================================
# 1. Class instantiation and registry
# =========================================================================

def test_instantiation():
    print("\n=== 1. Class instantiation and registry ===")

    sizes = {'dino-10M': 10, 'dino-100M': 100, 'dino-1B': 1000, 'dino-10B': 10000}

    for name, nb_M in sizes.items():
        check(name in DATASETS, "%s in DATASETS" % name)
        ds = DATASETS[name]()
        check(isinstance(ds, DINO10BDataset), "%s is DINO10BDataset" % name)
        check(isinstance(ds, DatasetCompetitionFormat),
              "%s is DatasetCompetitionFormat" % name)
        check(isinstance(ds, Dataset), "%s is Dataset" % name)

    ds = DINO10BDataset(10)
    check(ds.nb == 10_000_000, "nb=10M")
    check(ds.d == 1024, "d=1024")
    check(ds.nq == 100_000, "nq=100K")
    check(ds.dtype == "uint8", "dtype=uint8")
    check(ds.num_chunks == 1, "10M needs 1 chunk")

    ds = DINO10BDataset(10000)
    check(ds.num_chunks == 50, "10B needs 50 chunks")

    ds = DINO10BDataset(500)
    check(ds.num_chunks == 3, "500M needs 3 chunks")

    ds = DINO10BDataset(10)
    check(ds.gt_fn == "gts_dino_patch_10000000_k10.bin",
          "gt_fn correct for 10M")
    check(ds.qs_fn is None, "qs_fn is None")
    check(ds.ds_fn is None, "ds_fn is None")

    check(ds.distance() == "euclidean", "distance=euclidean")
    check(ds.search_type() == "knn", "search_type=knn (inherited)")
    check(ds.data_type() == "dense", "data_type=dense (inherited)")
    check(ds.default_count() == 10, "default_count=10 (inherited)")

    sn = ds.short_name()
    check("DINO10BDataset" in sn, "short_name contains class name")

    s = str(ds)
    check("1024" in s and "euclidean" in s, "__str__ works")


# =========================================================================
# 2. Inheritance chain
# =========================================================================

def test_inheritance():
    print("\n=== 2. Inheritance chain ===")

    ds = DINO10BDataset(10)

    check(DINO10BDataset.get_groundtruth is DatasetCompetitionFormat.get_groundtruth,
          "get_groundtruth NOT overridden (uses parent knn_result_read)")
    check(DINO10BDataset.search_type is DatasetCompetitionFormat.search_type,
          "search_type NOT overridden")
    check(DINO10BDataset.data_type is DatasetCompetitionFormat.data_type,
          "data_type NOT overridden")
    check(DINO10BDataset.get_queries is not DatasetCompetitionFormat.get_queries,
          "get_queries IS overridden (bvecs)")
    check(DINO10BDataset.get_dataset_iterator is not DatasetCompetitionFormat.get_dataset_iterator,
          "get_dataset_iterator IS overridden (chunked)")
    check(DINO10BDataset.prepare is not DatasetCompetitionFormat.prepare,
          "prepare IS overridden (chunk downloads)")


# =========================================================================
# 3. Format compatibility (bvecs + GT .bin round-trip)
# =========================================================================

def test_format_roundtrip():
    print("\n=== 3. Format round-trip ===")

    tmpdir = tempfile.mkdtemp()
    try:
        np.random.seed(42)

        # bvecs round-trip
        data = np.random.randint(0, 256, (1000, 64), dtype='uint8')
        bvecs_path = os.path.join(tmpdir, "test.bvecs")
        write_bvecs(bvecs_path, data)
        loaded = bvecs_mmap(bvecs_path)
        check(loaded.shape == (1000, 64), "bvecs shape correct")
        check(loaded.dtype == np.uint8, "bvecs dtype uint8")
        check(np.array_equal(data, loaded), "bvecs values match")

        # GT .bin round-trip
        nq, k = 50, 10
        I = np.random.randint(0, 1000, (nq, k), dtype='int32')
        D = np.sort(np.random.rand(nq, k).astype('float32'), axis=1)
        gt_path = os.path.join(tmpdir, "test_gt.bin")
        write_gt_bin(gt_path, I, D)
        I_loaded, D_loaded = knn_result_read(gt_path)
        check(I_loaded.shape == (nq, k), "GT I shape correct")
        check(D_loaded.shape == (nq, k), "GT D shape correct")
        check(I_loaded.dtype == np.int32, "GT I dtype int32")
        check(D_loaded.dtype == np.float32, "GT D dtype float32")
        check(np.array_equal(I, I_loaded), "GT I values match")
        check(np.allclose(D, D_loaded), "GT D values match")

        # sanitize
        x = np.random.rand(10, 5).astype('float32')[:, ::2]
        check(not x.flags['C_CONTIGUOUS'], "non-contiguous input")
        xs = sanitize(x)
        check(xs.flags['C_CONTIGUOUS'], "sanitize makes contiguous")

    finally:
        shutil.rmtree(tmpdir)


# =========================================================================
# 4. Functional tests with synthetic data
# =========================================================================

def make_synthetic_dataset(tmpdir, nb, nq, d, n_chunks=1):
    """Create synthetic DINO-like dataset files."""
    np.random.seed(123)

    # Queries
    queries = np.random.randint(0, 256, (nq, d), dtype='uint8')
    write_bvecs(os.path.join(tmpdir, "queries_clean.bvecs"), queries)

    # Base vectors (chunked)
    chunk_dir = os.path.join(tmpdir, "chunked_base_10B")
    os.makedirs(chunk_dir, exist_ok=True)
    vecs_per_chunk = nb // n_chunks
    all_vecs = np.random.randint(0, 256, (nb, d), dtype='uint8')
    for i in range(n_chunks):
        start = i * vecs_per_chunk
        end = start + vecs_per_chunk if i < n_chunks - 1 else nb
        write_bvecs(
            os.path.join(chunk_dir, "chunk_%04d.bvecs" % i),
            all_vecs[start:end])

    # GT: pick 10 nearest from base for each query (fake but monotonic)
    I = np.zeros((nq, 10), dtype='int32')
    D = np.zeros((nq, 10), dtype='float32')
    for q in range(nq):
        qvec = queries[q].astype('float32')
        dists = np.sum((all_vecs[:min(nb, 10000)].astype('float32') - qvec) ** 2, axis=1)
        idx = np.argsort(dists)[:10]
        I[q] = idx
        D[q] = dists[idx]

    gt_fn = "gts_dino_patch_%d_k10.bin" % nb
    write_gt_bin(os.path.join(tmpdir, gt_fn), I, D)

    return queries, all_vecs, I, D


def make_test_dataset(tmpdir, nb, nq, d, n_chunks=1):
    """Create synthetic data and return a patched DINO10BDataset."""
    queries, all_vecs, I, D = make_synthetic_dataset(
        tmpdir, nb, nq, d, n_chunks)

    ds = DINO10BDataset.__new__(DINO10BDataset)
    ds.nb = nb
    ds.d = d
    ds.nq = nq
    ds.dtype = "uint8"
    ds.basedir = tmpdir
    ds.base_url = "http://fake"
    ds.num_chunks = n_chunks
    ds.gt_fn = "gts_dino_patch_%d_k10.bin" % nb
    ds.qs_fn = None
    ds.ds_fn = None
    ds.private_qs_url = None
    ds.private_gt_url = None
    ds.VECTORS_PER_CHUNK = nb // n_chunks

    return ds, queries, all_vecs, I, D


def test_get_queries():
    print("\n=== 4a. get_queries() ===")

    tmpdir = tempfile.mkdtemp()
    try:
        ds, queries, _, _, _ = make_test_dataset(tmpdir, 1000, 50, 32)
        result = ds.get_queries()
        check(result.shape == (50, 32), "queries shape (50, 32)")
        check(result.flags['C_CONTIGUOUS'], "queries contiguous")
        check(np.array_equal(result, queries[:50]), "queries values match")
    finally:
        shutil.rmtree(tmpdir)


def test_get_groundtruth():
    print("\n=== 4b. get_groundtruth() ===")

    tmpdir = tempfile.mkdtemp()
    try:
        ds, _, _, I_expected, D_expected = make_test_dataset(
            tmpdir, 1000, 50, 32)

        I, D = ds.get_groundtruth()
        check(isinstance(I, np.ndarray), "I is ndarray")
        check(isinstance(D, np.ndarray), "D is ndarray")
        check(I.shape == (50, 10), "I shape (50, 10)")
        check(D.shape == (50, 10), "D shape (50, 10)")
        check(I.dtype == np.int32, "I dtype int32")
        check(D.dtype == np.float32, "D dtype float32")
        check(np.array_equal(I, I_expected), "I values match")
        check(np.allclose(D, D_expected), "D values match")

        # Truncation
        I5, D5 = ds.get_groundtruth(k=5)
        check(I5.shape == (50, 5), "k=5: I shape (50, 5)")
        check(D5.shape == (50, 5), "k=5: D shape (50, 5)")
        check(np.array_equal(I5, I_expected[:, :5]), "k=5: I values match")

        # Monotonicity
        monotonic = np.all(D[:, :-1] <= D[:, 1:])
        check(monotonic, "GT distances monotonically non-decreasing")
    finally:
        shutil.rmtree(tmpdir)


def test_dataset_iterator():
    print("\n=== 4c. get_dataset_iterator() ===")

    tmpdir = tempfile.mkdtemp()
    try:
        ds, _, all_vecs, _, _ = make_test_dataset(tmpdir, 1000, 50, 32)

        # Full iteration
        batches = list(ds.get_dataset_iterator(bs=300))
        total = sum(b.shape[0] for b in batches)
        check(total == 1000, "iterator yields exactly nb=%d vectors" % total)
        check(batches[0].shape == (300, 32), "first batch shape (300, 32)")
        check(batches[-1].shape[1] == 32, "last batch has correct dim")

        reconstructed = np.vstack(batches)
        check(np.array_equal(reconstructed, all_vecs),
              "iterator values match source data")

        for b in batches:
            check(b.flags['C_CONTIGUOUS'], "batch is contiguous")
            break  # just check first
    finally:
        shutil.rmtree(tmpdir)


def test_iterator_splits():
    print("\n=== 4d. get_dataset_iterator() with splits ===")

    tmpdir = tempfile.mkdtemp()
    try:
        ds, _, all_vecs, _, _ = make_test_dataset(tmpdir, 1000, 50, 32)

        # Split in 2
        half0 = np.vstack(list(ds.get_dataset_iterator(bs=200, split=(2, 0))))
        half1 = np.vstack(list(ds.get_dataset_iterator(bs=200, split=(2, 1))))
        check(half0.shape[0] == 500, "split(2,0) yields 500 vectors")
        check(half1.shape[0] == 500, "split(2,1) yields 500 vectors")
        combined = np.vstack([half0, half1])
        check(np.array_equal(combined, all_vecs),
              "two halves combine to full dataset")

        # Split in 4
        parts = []
        for p in range(4):
            part = np.vstack(list(ds.get_dataset_iterator(bs=100, split=(4, p))))
            parts.append(part)
        total = sum(p.shape[0] for p in parts)
        check(total == 1000, "4 splits yield 1000 total")
        combined4 = np.vstack(parts)
        check(np.array_equal(combined4, all_vecs),
              "four quarters combine to full dataset")
    finally:
        shutil.rmtree(tmpdir)


def test_iterator_multi_chunk():
    print("\n=== 4e. get_dataset_iterator() across chunk boundaries ===")

    tmpdir = tempfile.mkdtemp()
    try:
        ds, _, all_vecs, _, _ = make_test_dataset(
            tmpdir, 1000, 50, 32, n_chunks=2)

        batches = list(ds.get_dataset_iterator(bs=300))
        total = sum(b.shape[0] for b in batches)
        check(total == 1000, "multi-chunk: yields all %d vectors" % total)

        reconstructed = np.vstack(batches)
        check(np.array_equal(reconstructed, all_vecs),
              "multi-chunk: values match across boundary")

        # Splits across chunks
        half0 = np.vstack(list(ds.get_dataset_iterator(bs=200, split=(2, 0))))
        half1 = np.vstack(list(ds.get_dataset_iterator(bs=200, split=(2, 1))))
        combined = np.vstack([half0, half1])
        check(np.array_equal(combined, all_vecs),
              "multi-chunk splits combine correctly")
    finally:
        shutil.rmtree(tmpdir)


def test_get_dataset():
    print("\n=== 4f. get_dataset() ===")

    tmpdir = tempfile.mkdtemp()
    try:
        ds, _, all_vecs, _, _ = make_test_dataset(tmpdir, 1000, 50, 32)
        result = ds.get_dataset()
        check(result.shape == (1000, 32), "get_dataset shape (1000, 32)")
        check(np.array_equal(result, all_vecs), "get_dataset values match")
    finally:
        shutil.rmtree(tmpdir)


# =========================================================================
# 5. GT monotonicity test (recall_tests.py pattern)
# =========================================================================

def test_gt_monotonicity_synthetic():
    print("\n=== 5. GT monotonicity (synthetic) ===")

    tmpdir = tempfile.mkdtemp()
    try:
        ds, _, _, _, _ = make_test_dataset(tmpdir, 1000, 50, 32)
        I, D = ds.get_groundtruth()

        check(len((I, D)) == 2, "get_groundtruth returns tuple of 2")
        check(I.shape[1] == D.shape[1], "I and D have same k")
        check(I.shape[1] >= 10, "k >= 10")

        for i in range(I.shape[0]):
            if not np.all(D[i, :-1] <= D[i, 1:]):
                check(False, "monotonicity at query %d" % i)
                return
        check(True, "all queries have monotonically non-decreasing distances")
    finally:
        shutil.rmtree(tmpdir)


# =========================================================================
# 6. prepare() tests
# =========================================================================

def test_prepare():
    print("\n=== 6. prepare() URL construction ===")

    ds = DINO10BDataset(10)
    check(ds.base_url == "http://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B",
          "base_url correct")
    check(ds.gt_fn == "gts_dino_patch_10000000_k10.bin",
          "gt_fn correct for 10M")

    ds100 = DINO10BDataset(100)
    check(ds100.gt_fn == "gts_dino_patch_100000000_k10.bin",
          "gt_fn correct for 100M")

    ds10b = DINO10BDataset(10000)
    check(ds10b.gt_fn == "gts_dino_patch_10000000000_k10.bin",
          "gt_fn correct for 10B")
    check(ds10b.num_chunks == 50, "10B prepare would download 50 chunks")


# =========================================================================
# 7. Download URL reachability
# =========================================================================

def test_download_urls():
    print("\n=== 7. Download URL reachability ===")

    from urllib.request import urlopen, Request

    base = "http://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B"

    urls = [
        ("queries", base + "/queries_clean.bvecs"),
        ("GT 10M", base + "/gts_bin/gts_dino_patch_10000000_k10.bin"),
        ("GT 100M", base + "/gts_bin/gts_dino_patch_100000000_k10.bin"),
        ("GT 1B", base + "/gts_bin/gts_dino_patch_1000000000_k10.bin"),
        ("GT 10B", base + "/gts_bin/gts_dino_patch_10000000000_k10.bin"),
        ("chunk_0000", base + "/chunked_base_10B/chunk_0000.bvecs"),
        ("chunk_0049", base + "/chunked_base_10B/chunk_0049.bvecs"),
    ]

    for name, url in urls:
        try:
            req = Request(url, method='HEAD')
            resp = urlopen(req, timeout=10)
            code = resp.getcode()
            size = int(resp.headers.get('Content-Length', 0))
            check(code == 200, "%s URL reachable (HTTP %d, %.1f MB)" % (
                name, code, size / 1e6))
        except Exception as e:
            check(False, "%s URL reachable (%s)" % (name, e))


# =========================================================================
# 8. Live data validation (skipped if data not present)
# =========================================================================

def test_live_data():
    print("\n=== 8. Live data validation ===")

    basedir = os.path.join("data", "dino_vitl_10B")
    qs_path = os.path.join(basedir, "queries_clean.bvecs")
    gt_path = os.path.join(basedir, "gts_dino_patch_10000000_k10.bin")
    chunk_path = os.path.join(basedir, "chunked_base_10B", "chunk_0000.bvecs")

    if not os.path.exists(qs_path):
        print("  SKIP: queries not downloaded (%s)" % qs_path)
        return

    # Queries
    queries = bvecs_mmap(qs_path)
    check(queries.shape == (100000, 1024),
          "real queries shape (100000, 1024)")
    check(queries.dtype == np.uint8, "real queries dtype uint8")

    if not os.path.exists(gt_path):
        print("  SKIP: GT not downloaded (%s)" % gt_path)
        return

    # GT
    I, D = knn_result_read(gt_path)
    check(I.shape == (100000, 10), "real GT I shape (100000, 10)")
    check(D.shape == (100000, 10), "real GT D shape (100000, 10)")
    check(I.dtype == np.int32, "real GT I dtype int32")
    check(D.dtype == np.float32, "real GT D dtype float32")
    check(I.max() < 10_000_000,
          "real GT max index %d < 10M" % I.max())
    check(I.min() >= 0, "real GT min index >= 0")

    monotonic = np.all(D[:, :-1] <= D[:, 1:])
    check(monotonic, "real GT distances monotonically non-decreasing")

    if not monotonic:
        bad = np.where(~np.all(D[:, :-1] <= D[:, 1:], axis=1))[0]
        print("    %d bad queries, first: %d" % (len(bad), bad[0]))
        print("    distances: %s" % D[bad[0]])

    # Test via class interface
    ds = DINO10BDataset(10)
    ds_queries = ds.get_queries()
    check(ds_queries.shape == (100000, 1024),
          "DINO10BDataset.get_queries() shape correct")
    check(ds_queries.flags['C_CONTIGUOUS'],
          "DINO10BDataset.get_queries() contiguous")

    I_ds, D_ds = ds.get_groundtruth()
    check(np.array_equal(I_ds, I),
          "DINO10BDataset.get_groundtruth() I matches direct read")
    check(np.allclose(D_ds, D),
          "DINO10BDataset.get_groundtruth() D matches direct read")

    I5, D5 = ds.get_groundtruth(k=5)
    check(I5.shape == (100000, 5), "get_groundtruth(k=5) truncates I")
    check(D5.shape == (100000, 5), "get_groundtruth(k=5) truncates D")
    check(np.array_equal(I5, I[:, :5]), "k=5 truncation correct")

    if not os.path.exists(chunk_path):
        print("  SKIP: chunk_0000 not downloaded (%s)" % chunk_path)
        return

    # Dataset iterator with real data
    print("  Testing iterator on 10M vectors (may take a moment)...")
    total = 0
    for batch in ds.get_dataset_iterator(bs=500_000):
        check(batch.shape[1] == 1024,
              "real iterator batch dim 1024 (got %d)" % batch.shape[1])
        total += batch.shape[0]
        break  # just check first batch to save time

    check(total > 0, "real iterator yields vectors (first batch: %d)" % total)

    # Split test on real data
    split0 = []
    for batch in ds.get_dataset_iterator(bs=1_000_000, split=(2, 0)):
        split0.append(batch.shape[0])
    split1 = []
    for batch in ds.get_dataset_iterator(bs=1_000_000, split=(2, 1)):
        split1.append(batch.shape[0])
    total0 = sum(split0)
    total1 = sum(split1)
    check(total0 == 5_000_000,
          "real split(2,0) yields 5M (got %d)" % total0)
    check(total1 == 5_000_000,
          "real split(2,1) yields 5M (got %d)" % total1)


# =========================================================================
# Main
# =========================================================================

def main():
    global PASS, FAIL

    test_instantiation()
    test_inheritance()
    test_format_roundtrip()
    test_get_queries()
    test_get_groundtruth()
    test_dataset_iterator()
    test_iterator_splits()
    test_iterator_multi_chunk()
    test_get_dataset()
    test_gt_monotonicity_synthetic()
    test_prepare()
    test_download_urls()
    test_live_data()

    print("\n" + "=" * 50)
    print("Results: %d PASSED, %d FAILED" % (PASS, FAIL))
    print("=" * 50)

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
