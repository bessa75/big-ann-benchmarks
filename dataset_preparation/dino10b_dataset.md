# DINO 10B Dataset

This file describes a public vector benchmark consisting of 10 billion 1024-dimensional uint8 vectors extracted from image patches in the [YFCC100M dataset](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/) using a [DINOv3 ViT-L/16 model](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).

The dataset is one of the largest publicly available dense vector benchmarks, designed for evaluating approximate nearest neighbor search at extreme scale.


## License

The original YFCC100M images are licensed under various [Creative Commons Licenses](http://www.creativecommons.org/) (see the metadata for each image). The embedding vectors and auxiliary files are released by Meta.


## Dataset Properties

| Property | Value |
|----------|-------|
| Base vectors | 10,000,000,000 (10B) |
| Dimensions | 1024 |
| Data type | uint8 |
| Query vectors | 100,000 |
| Training vectors | 99,000,000 |
| Distance metric | L2 (Euclidean) |
| Ground truth | Top-10 nearest neighbors |


## Files and Format

The dataset uses the `.bvecs` binary format. In this format, each vector is stored as a 4-byte little-endian integer (the dimension), followed by `d` bytes of uint8 vector data. This per-vector header is repeated for every vector in the file.

The base vectors are split across 50 chunk files (`chunk_0000.bvecs` through `chunk_0049.bvecs`), each containing 200 million vectors (~200GB per chunk). The total dataset size is approximately 10 TB.


### Download URLs

**Queries and ground truth:**
```bash
wget http://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/queries_clean.bvecs
wget http://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/gts/gts_dino_patch_10000000000_k10.npy
```

**Base vectors (50 chunks):**
```bash
wget http://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/chunked_base_10B/chunk_0000.bvecs
wget http://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/chunked_base_10B/chunk_0001.bvecs
# ... through chunk_0049.bvecs
```

A complete file list is available at:
```bash
wget http://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/file_list.txt
```


### Accelerated Download

Due to the large size of the data (~10 TB), we recommend using the [Axel download accelerator](https://github.com/axel-download-accelerator/axel):

```bash
wget http://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/file_list.txt

base_dir="data/dino_vitl_10B"
mkdir -p "$base_dir/chunked_base_10B" "$base_dir/gts"

while read -r url; do
  path=$(echo "$url" | sed -E 's|https?://[^/]+/||')
  rel=$(echo "$path" | cut -d/ -f3-)
  [ -z "$rel" ] && rel=$(basename "$path")
  out="$base_dir/$rel"
  mkdir -p "$(dirname "$out")"
  axel -n 16 -a -o "$out" "$url"
done < file_list.txt
rm file_list.txt
```


### Supported Subset Sizes

Ground truth files are pre-computed for the following dataset sizes (always using the first N vectors from the chunked files):

100K, 200K, 500K, 1M, 2M, 5M, 10M, 20M, 50M, 100M, 200M, 500M, 1B, 2B, 5B, 10B

For smaller subsets, only the necessary chunk files need to be downloaded (each chunk contains 200M vectors).


### Ground Truth Format

Ground truth files are stored in the standard competition binary format: a header of two `uint32` values (`nq`, `k`), followed by `nq × k` `int32` nearest neighbor indices, followed by `nq × k` `float32` L2 distances. Each file contains 100,000 queries × 10 neighbors.

Download URLs follow the pattern:
```bash
wget http://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/gts_bin/gts_dino_patch_{nb}_k10.bin
```


## Development

### Embedding Generation

Image patches from the YFCC100M dataset were processed through a DINOv3 ViT-L/16 model ([facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)). The resulting embeddings were quantized to uint8 and stored in chunked `.bvecs` format for efficient streaming access.

### Notes

- Each chunk file contains exactly 200 million vectors. When using a dataset size that is not a multiple of 200M, only a prefix of the last required chunk is used.
- For large batch sizes, prefer batch sizes that are divisors of 200,000,000 to avoid splitting batches across chunk boundaries, which causes overhead from concatenation.
- The training set (`train_queries_99M.bvecs`, ~400GB) contains 99 million vectors and is available for training quantizers or other index structures.
