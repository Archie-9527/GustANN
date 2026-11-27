# GPU Direct-IO Support

Note: this setup is mainly for research purpose.

This document shows how to run GustANN with BaM GPU-SSD direct access engine[^1].

## Build

First, prepare BaM as shown in the [repository](https://github.com/ZaidQureshi/bam).

``` shell-session
$ git clone https://github.com/ZaidQureshi/bam.git deps/bam
$ # Then build BaM and its Linux kernel module as instructed in original repo
```

We tested BaM with git commit hash `2edd33c`. You may checkout this version. 
To compile and link successfully, you may need to add an `inline` at the beginning of line 584 in `dep/bam/include/page_cache.h`

Then, to enable BaM in GustANN, please recompile GustANN with:

``` shell-session
$ # in ./build
$ cmake -DGUSTANN_USE_BAM=ON ..
$ make -j
```

## Run GustANN-BaM

**Root previllege is required!**

``` shell-session
# ./bin/search_disk --query <query_file>  --index <index_file> --ground_truth <ground_truth>   --pq_data <pq_file> --nav_graph <nav_graph> --data_type <data_type> --topk <topk> --ef_search <L> --num_ctrls <num_ctrls> --cache_page <cache_pages> [--copy_data] [--copy_only] 
```

Most of the arguments are the same as original GustANN, here lists the difference:

+ `num_ctrls`: Number of SSDs to use. By default, BaM maps SSDs to `/dev/libnvm*`. If it is not the case, you can modify line 29 in `src/bam.cu`.
+ `cache_pages`: The number of SSD pages cached in GPU Memory.
+ `copy_data`: Copies the DiskANN index in `index_file` to SSD.
+ `copy_only`: Only copies index to SSD without searching it.

## SIFT1M 功能与性能对比

`scripts/benchmark_sift1m.py` 默认测试 DiskANN、GustANN libaio、GustANN liburing、GustANN BaM 仿真盘 4 KiB 和 512 B。前三组读取同一个主机文件系统上的索引，BaM 两组读取同一个仿真盘上的数据区。两组 BaM 默认使用同样的 64 MiB GPU 缓存数据容量。

在 Linux CUDA 主机上构建。BaM 的内核模块和 `deps/bam/build/lib/libnvm.so` 需要提前准备。构建 GustANN 时也需要安装 libaio 和 liburing 的开发库。

```bash
cmake -S deps/DiskANN -B deps/DiskANN/build -DCMAKE_BUILD_TYPE=Release
cmake --build deps/DiskANN/build --target build_disk_index search_disk_index -j
cmake -S . -B build -DGUSTANN_USE_BAM=ON -DGUSTANN_USE_AIO=ON -DGUSTANN_USE_URING=ON
cmake --build build --target search_disk search_disk_hybrid -j
```

SIFT1M 目录放置 `sift_base.fvecs`、`sift_query.fvecs` 和 `sift_groundtruth.ivecs`。已有 float/L2 DiskANN 索引时可以加 `--index-prefix /path/to/index/sift1m`，脚本会复用对应的 `_disk.index` 和两个 PQ 文件。首次运行加 `--load-bam`，脚本会确认仿真盘报告 512 B LBA，然后从 `/dev/libnvm0` 的 namespace 1、LBA 0 写入索引数据。后续使用同一份已装载索引时省略 `--load-bam`。`--bam-launcher ''` 可用于当前账户已有设备权限的环境。

```bash
python3 scripts/benchmark_sift1m.py \
  --sift-dir /data/sift1m --index-prefix /data/index/sift1m \
  --out-dir results/sift1m-sim --load-bam
```

每条记录是一批 10,000 条查询，Recall@10 使用同一份 ground truth。`measurements.jsonl` 保留原始批次，`sweep.json` 汇总每个 `L`，`recall_targets.json` 为每个目标选取达到该 Recall 的实测点中 QPS 最高的一条。`--trials` 可重复独立进程，吞吐使用批次时长中位数，达标按最低实测 Recall 判断。达不到的目标标为 `unreached`，可用 `--search-list` 扩大或加密扫描。延迟字段 `batch_latency_ms` 表示整批完成时间，DiskANN 另外保留工具输出的单查询平均和 P99.9 延迟。I/O 指标是搜索阶段应用发起的读命令和字节数。DiskANN 使用工具输出的 `Mean IOs` 估算，GustANN 直接计数。BaM 的总量包含补充读，报告同时给出主读和补充读。仿真器的设备侧数据可用命令 trace 另行核对。

### BaM 读取和装载

`--bam-io-size 4096` 仍按 DiskANN 的 4 KiB 数据页访问。在仿真盘上，每次缓存缺失提交一条覆盖 8 个 LBA 的主读。`--bam-io-size 512` 将 BaM 缓存槽和主读改为单 LBA，GPU 根据节点的向量字节数和实际度数，只获取所需扇区，并在块内共享内存中拼接。脚本为仿真盘启用 `--bam-sim`，要求一块 512 B LBA 的控制器。两组的 BaM 补充 READ 都读取 1 个 LBA，用于协调完成顺序。4 KiB 组的设备 trace 因此也会出现 512 B 补充读。两种模式共用同一盘上字节布局。装载器跳过完整索引文件的第一个 4 KiB 元数据页，从目标 LBA 0 开始写节点数据。

### 真实物理盘补测

默认测试列表跳过物理盘。将专用物理盘绑定到 BaM、确认 `/dev/libnvm*` 映射后，用下面的命令补测 4 KiB 路径。物理盘可以使用 512 B 或 4 KiB LBA，设备尚未装载该索引时再加入 `--load-bam`。加 `--dry-run` 可以查看命令而不访问设备。

```bash
python3 scripts/benchmark_sift1m.py \
  --sift-dir /data/sift1m --index-prefix /data/index/sift1m \
  --out-dir results/sift1m-physical --scenarios bam-physical-4k --physical-ctrls 1
```

## Reference

[^1]: Zaid Qureshi, Vikram Sharma Mailthody, Isaac Gelado, Seungwon Min, Amna Masood, Jeongmin Park, Jinjun Xiong, C. J. Newburn, Dmitri Vainbrand, I-Hsin Chung, Michael Garland, William Dally, and Wen-mei Hwu. 2023. GPU-Initiated On-Demand High-Throughput Storage Access in the BaM System Architecture. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS 2023). Association for Computing Machinery, New York, NY, USA, 325–339. https://doi.org/10.1145/3575693.3575748
