#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <thread>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>


#include "common.hpp"
#include "common_cuda.cuh"
#include "nav_graph.hpp"
#include "bam.hpp"
#include "page_cache.h"
#include "ssd_search.hpp"
//#include "ssd_search_kernel.hpp"

#include "impl/bam/def.cuh"
#include "impl/bam/transfer.cuh"
//#include "impl/bam/search_v1.cuh"
#include "impl/bam/search.cuh"
#include "impl/nav.cuh"



namespace gustann {

  const char *const ctrls_paths[6] = {"/dev/libnvm0","/dev/libnvm1","/dev/libnvm2","/dev/libnvm3","/dev/libnvm4","/dev/libnvm5"};

  BaMExecutor::BaMContext::~BaMContext() {
    for (auto* controller : ctrls) {
      auto* bar = static_cast<volatile char*>(controller->ctrl->mm_ptr);
      auto* config = reinterpret_cast<volatile uint32_t*>(bar + 0x14);
      *config &= ~1u;  // Clear NVMe CC.EN before releasing DMA mappings.
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    for (auto* controller : ctrls) {
      const auto timeout = std::chrono::milliseconds(std::max<uint64_t>(1000, controller->ctrl->timeout));
      const auto deadline = std::chrono::steady_clock::now() + timeout;
      while (*controller->controller_status() & 1u) {
        if (std::chrono::steady_clock::now() >= deadline) {
          ERROR("Timed out waiting for BaM controller to stop; DMA mappings were left pinned");
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    delete a;
    delete h_range;
    delete h_pc;
    for (auto* controller : ctrls) {
      delete controller;
    }
  }

  BaMExecutor::BaMExecutor(const std::string &fpath, const Layout &layout,
                           const DataType& data_type,
                           const BaMConfig &config, bool copy_data)
    : layout_(layout) {
    block_cnt_ = 112 * 100;
    block_dim_ = 32;
    visited_list_size_ = 8192 * 8;
    visited_table_size_ = visited_list_size_ * 2;
    data_type_ = data_type;
    
    init(config);
    if (copy_data) {
      copy_to_bam(fpath);
    }
#ifdef _USE_MEM
    read_to_mem(fpath);
#endif
  }

  void BaMExecutor::copy_to_bam(const std::string &fpath) {
    FILE* input = std::fopen(fpath.c_str(), "rb");
    if (!input || std::fseek(input, DISKANN_PAGE_SIZE, SEEK_SET) != 0) {
      ERROR("Cannot open DiskANN graph data in {}", fpath);
      exit(EXIT_FAILURE);
    }

    const uint64_t data_pages = layout_.num_pages * DISKANN_PAGE_SIZE / page_size_;
    const uint64_t batch_pages = std::min<uint64_t>(data_pages, 64 * 1024 * 1024 / page_size_);
    const uint64_t copy_blocks = std::min<uint64_t>(112 * 4, bam_data_.h_pc->pdt.n_pages);
    uint8_t* buffer;
    CHECK_CUDA(cudaMallocHost(&buffer, batch_pages * page_size_));
    const double start = elapsed();
    for (uint64_t i = 0; i < data_pages; i += batch_pages) {
      const uint64_t count = std::min(batch_pages, data_pages - i);
      if (std::fread(buffer, page_size_, count, input) != count) {
        ERROR("Truncated DiskANN graph data in {} at BaM page {}", fpath, i);
        exit(EXIT_FAILURE);
      }
      copy_page_to_ssd<<<copy_blocks, block_dim_>>>(
          bam_data_.h_pc->pdt.d_ctrls, bam_data_.h_pc->d_pc_ptr,
          bam_data_.ctrls.size(), buffer, i, count, page_size_);
      CHECK_CUDA(cudaDeviceSynchronize());
    }
#ifdef ASYNC_READ
    bam_data_.h_pc->clear_cache();
    CHECK_CUDA(cudaDeviceSynchronize());
#endif
    CHECK_CUDA(cudaFreeHost(buffer));
    std::fclose(input);
    INFO("Copied {} graph bytes to BaM in {:.3f} s", data_pages * page_size_, elapsed() - start);
  }

  void BaMExecutor::init(const BaMConfig &config) {
    if (config.page_size != 512 && config.page_size != DISKANN_PAGE_SIZE) {
      ERROR("BaM I/O size must be 512 or {} bytes", DISKANN_PAGE_SIZE);
      exit(EXIT_FAILURE);
    }
    if (config.num_ctrls < 1 || config.num_ctrls > sizeof(ctrls_paths) / sizeof(ctrls_paths[0]) ||
        ((config.page_size == 512 || config.simulator_mode) && config.num_ctrls != 1)) {
      ERROR("Invalid BaM controller count {} for the selected mode", config.num_ctrls);
      exit(EXIT_FAILURE);
    }
    if (config.num_page <= 0) {
      ERROR("BaM cache page count must be positive");
      exit(EXIT_FAILURE);
    }
    if (config.num_queues <= 0 || config.num_queues > UINT16_MAX) {
      ERROR("BaM I/O queue count must be between 1 and {}", UINT16_MAX);
      exit(EXIT_FAILURE);
    }
    const uint64_t data_size = layout_.num_pages * DISKANN_PAGE_SIZE;
    const uint64_t ssd_pages = data_size / config.page_size;
    for (int i = 0; i < config.num_ctrls; i++) {
      bam_data_.ctrls.push_back
        (new Controller(ctrls_paths[i],
                        config.nvm_namespace,
                        config.cuda_device,
                        config.queue_depth,
                        config.num_queues));
      const auto lba_size = bam_data_.ctrls.back()->blk_size;
      if (lba_size == 0 || config.page_size % lba_size != 0 ||
          lba_size != bam_data_.ctrls.front()->blk_size ||
          (config.simulator_mode && lba_size != 512)) {
        ERROR("BaM I/O size {} is incompatible with namespace LBA size {} on {}",
              config.page_size, lba_size, ctrls_paths[i]);
        exit(EXIT_FAILURE);
      }
      const uint64_t controller_pages = (ssd_pages + config.num_ctrls - 1 - i) / config.num_ctrls;
      if (controller_pages > bam_data_.ctrls.back()->ns.size / (config.page_size / lba_size)) {
        ERROR("The DiskANN graph data exceeds the namespace size on {}", ctrls_paths[i]);
        exit(EXIT_FAILURE);
      }
    }
    
    int num_pages = config.num_page;
    
    DEBUG("Cache page: {}", num_pages);
    DEBUG("Page size: {}", config.page_size);
    DEBUG("Data page: {}", ssd_pages);
    
    bam_data_.h_pc = new page_cache_t(config.page_size,
                                      num_pages,
                                      config.cuda_device,
                                      bam_data_.ctrls[0][0],
                                      64,
                                      bam_data_.ctrls);
    page_size_ = config.page_size;
    INFO("BaM I/O: {} bytes, {} LBA(s) of {} bytes, {} GPU cache slots, 1 LBA per supplemental READ",
         page_size_, bam_data_.h_pc->pdt.n_blocks_per_page, bam_data_.ctrls.front()->blk_size, num_pages);

    if (config.use_simple_cache) {
      this->use_simple_cache = true;
    } else {
      this->use_simple_cache = false;
      bam_data_.h_range = new range_t<uint8_t>
        (0, data_size, 0,
         ssd_pages,
         0, config.page_size, bam_data_.h_pc,
         config.cuda_device,
         STRIPE);
      bam_data_.vr.push_back(bam_data_.h_range);
      bam_data_.a = new array_t<uint8_t> (data_size, 0, bam_data_.vr, config.cuda_device);
    }
    
    INFO("Inited BaM device");
  }

  void BaMExecutor::read_to_mem(const std::string &fpath) {
    FILE* input = fopen(fpath.c_str(), "rb");
    if (!input) {
      ERROR("Failed to open file: {}", fpath);
      exit(-1);
    }
    fseek(input, DISKANN_PAGE_SIZE, SEEK_SET);
    CHECK_CUDA(cudaHostAlloc(&mem_data_, layout_.num_pages * DISKANN_PAGE_SIZE, cudaHostAllocPortable));
    uint8_t* mem_data_host = mem_data_; //new uint8_t[num_pages_ * page_size_];
    ASSERT(fread(mem_data_host, 1, layout_.num_pages * DISKANN_PAGE_SIZE, input) ==
           layout_.num_pages * DISKANN_PAGE_SIZE);
    //CHECK_CUDA(cudaMalloc(&mem_data_, num_pages_ * page_size_));
    //CHECK_CUDA(cudaMemcpy(mem_data_, mem_data_host, num_pages_ * page_size_, cudaMemcpyHostToDevice));
    fclose(input);
    //delete [] mem_data_host;
  }

  void BaMExecutor::search(const float *qdata, const int num_queries_,
                           const int topk, const int ef_search, int *nns,
                           float *distances, int *found_cnt,
                           PQSearch *pq, NavGraph *nav_graph,
                           GustANNStats *stats) {
    int num_queries = num_queries_;
    //num_queries = 10;   
    
    thrust::device_vector<float> d_qdata(num_queries * layout_.num_dims);
    thrust::copy(qdata, qdata + num_queries * layout_.num_dims, d_qdata.begin());

    int aligned_ef = (ef_search + layout_.max_m0 + 31) / 32 * 32;

    thrust::device_vector<int> d_entries(num_queries);
    thrust::device_vector<int> d_nns(num_queries * topk);
    thrust::device_vector<float> d_distances(num_queries * topk);
    thrust::device_vector<int> d_found_cnt(num_queries);
    thrust::device_vector<uint32_t> d_neighbors_id(aligned_ef * block_cnt_);
    thrust::device_vector<float> d_neighbors_dist(aligned_ef * block_cnt_);


    DEBUG("Start Search");

    //fetch_all_data<<<1, 32>>>(bam_data_.a->d_array_ptr, num_pages_);
    if (pq) {
      pq->init_device(layout_.num_dims, layout_.num_data, block_cnt_,
                      ef_search);
    } else {
      ERROR("PQ is not inited!");
      throw;
    }
    bam_data_.a->print_reset_stats();
    uint64_t extra_reads_before = 0;
    if (stats) {
      CHECK_CUDA(cudaMemcpy(&extra_reads_before, bam_data_.h_pc->pdt.extra_reads,
                            sizeof(extra_reads_before), cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    double start = elapsed();


    int init_ef = std::min(ef_search, 5);

    if (nav_graph) {
      get_entry_kernel(data_type_)<<<(num_queries + 1) / 2, 64, 0>>>(
          thrust::raw_pointer_cast(d_qdata.data()), nav_graph->data_dev,
          nav_graph->graph_dev, num_queries, nav_graph->num_node,
          layout_.num_dims, nav_graph->max_m, init_ef, nav_graph->start,
          thrust::raw_pointer_cast(d_entries.data()),
          thrust::raw_pointer_cast(d_neighbors_id.data()),
          thrust::raw_pointer_cast(d_neighbors_dist.data()));
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<int> entries(num_queries);
    thrust::copy(d_entries.begin(), d_entries.end(), entries.begin());
    if (nav_graph) {
      nav_graph->translate(entries.data(), entries.size());
    } else {
      std::fill(entries.begin(), entries.end(), layout_.enter_point);
    }
    thrust::copy(entries.begin(), entries.end(), d_entries.begin());
    double t1 = elapsed();
    INFO("Init: {}", t1 - start);

    auto kernel_func = search_disk_graph_kernel<uint8_t>;
#pragma GCC diagnostics push
#pragma GCC diagnostics error "-Wswtich"
    switch (data_type_) {
    case UINT8:
      break;
    case FLOAT:
      kernel_func = search_disk_graph_kernel<float>;
      break;
    }
#pragma GCC diagnostics pop
    const size_t scratch_bytes = page_size_ == DISKANN_PAGE_SIZE ? 0 : layout_.node_size + alignof(int) - 1;
    const int search_threads = std::max<int>(64, (layout_.max_m0 + 31) / 32 * 32);
    kernel_func<<<
      block_cnt_, search_threads,
      (sizeof(int) * 3 + sizeof(float) * 2) * (ef_search + layout_.max_m0) + scratch_bytes
        >>>
      (
#ifdef _USE_MEM
        mem_data_,
#else
        bam_data_.a->d_array_ptr,
#endif
        thrust::raw_pointer_cast(d_qdata.data()),
        layout_.num_dims,
        pq->get_device_ptr(),
        layout_.nodes_per_page, layout_.node_size, layout_.data_size, page_size_,
        thrust::raw_pointer_cast(d_entries.data()),
        layout_.max_m0, ef_search, topk,
        thrust::raw_pointer_cast(d_nns.data()), 
        thrust::raw_pointer_cast(d_distances.data()), 
        thrust::raw_pointer_cast(d_found_cnt.data()), 
        num_queries
      );

    CHECK_CUDA(cudaDeviceSynchronize());
    double end = elapsed();

    DEBUG("End Search");
    INFO("Use time: {}", end - start);
    if (stats) {
      stats->run_time = end - start;
      range_d_t<uint8_t> range_stats;
      // The search kernel updates the range copied into the array at initialization.
      CHECK_CUDA(cudaMemcpy(&range_stats, bam_data_.a->adt.d_ranges, sizeof(range_stats), cudaMemcpyDeviceToHost));
      uint64_t extra_reads;
      CHECK_CUDA(cudaMemcpy(&extra_reads, bam_data_.h_pc->pdt.extra_reads,
                            sizeof(extra_reads), cudaMemcpyDeviceToHost));
      stats->read_commands = range_stats.miss_cnt;
      stats->read_bytes = stats->read_commands * page_size_;
      stats->extra_read_commands = extra_reads - extra_reads_before;
      stats->extra_read_bytes = stats->extra_read_commands * bam_data_.ctrls.front()->blk_size;
    }
    std::vector<int64_t> acc_visited_cnt(block_cnt_);
    thrust::copy(d_nns.begin(), d_nns.end(), nns);
    thrust::copy(d_distances.begin(), d_distances.end(), distances);
    thrust::copy(d_found_cnt.begin(), d_found_cnt.end(), found_cnt);
    CHECK_CUDA(cudaDeviceSynchronize());

    bam_data_.a->print_reset_stats();

  }

}
