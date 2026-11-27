#pragma once
#include <cassert>
#include "def.cuh"
#include "../../layout.hpp"

namespace gustann {

  struct ReadCtx {
    int64_t r;
    int64_t start_idx;
    data_page_t* page;
  };

  template <class T>
  struct GraphData {
    int deg;
    int* edge;
    T* data;
    __inline__ __device__
    int operator [] (int x) const { return edge[x]; }
  };

  
  struct PageData {
    int nodes_per_page;
    int node_len;
    int data_len;
    DiskData* disk;
    int page_size = DISKANN_PAGE_SIZE;
    __inline__ __device__ void init() {}
    __inline__ __device__ int get_block(int nodeid) {
      return nodeid / nodes_per_page;
    }
    __inline__ __device__ int get_offset(int nodeid) {
      return nodeid % nodes_per_page * node_len;
    }
    __inline__ __device__ uint8_t* read_node(int nodeid, ReadCtx& ctx) {
      int block = get_block(nodeid);
      //if (threadIdx.x == 0) printf("B %d\n", block);
      ctx.start_idx = (int64_t) block * DISKANN_PAGE_SIZE;
      size_t s, e;
      return (uint8_t*) disk->acquire_page(ctx.start_idx, ctx.page, s, e, ctx.r);
    }
    __inline__ __device__ int copy_page_fragment(int64_t offset, int remaining, uint8_t* dest) {
      ReadCtx ctx;
      ctx.start_idx = offset;
      size_t begin, end;
      const auto* page = static_cast<const uint8_t*>(disk->acquire_page(offset, ctx.page, begin, end, ctx.r));
      const int available = static_cast<int>(end - offset);
      const int count = remaining < available ? remaining : available;
      for (int i = 0; i < count; ++i) {
        dest[i] = page[offset - begin + i];
      }
      drop(ctx);
      return count;
    }

    __inline__ __device__ uint8_t* read_partial_node(int nodeid, uint8_t* scratch) {
      const int64_t node_start = (int64_t)get_block(nodeid) * DISKANN_PAGE_SIZE + get_offset(nodeid);
      const int prefix = data_len + sizeof(int);
      // Align the degree and neighbors even for a byte vector with an odd dimension.
      uint8_t* node = scratch + ((-data_len) & (alignof(int) - 1));
      int copied = 0;
      while (copied < prefix) {
        copied += copy_page_fragment(node_start + copied, node_len - copied, node + copied);
      }
      const int degree = *reinterpret_cast<const int*>(node + data_len);
      assert(degree >= 0 && degree <= (node_len - prefix) / static_cast<int>(sizeof(int)));
      const int required = prefix + degree * static_cast<int>(sizeof(int));
      while (copied < required) {
        copied += copy_page_fragment(node_start + copied, required - copied, node + copied);
      }
      return node;
    }

    template<class T>
    __inline__ __device__ GraphData<T> get_graph(int nodeid, ReadCtx& ctx, uint8_t* scratch = nullptr) {
      uint8_t* tmp;
      if (page_size == DISKANN_PAGE_SIZE) {
        tmp = read_node(nodeid, ctx) + get_offset(nodeid);
      } else {
        ctx.r = -1;
        tmp = read_partial_node(nodeid, scratch);
      }
      T* data = (T*) tmp;
      int* result = (int*)(tmp + data_len);
      return (GraphData<T>) { result[0], result + 1, data };
    }
    // TODO: template
    template<class T>
    __inline__ __device__ T* get_data(int nodeid, ReadCtx& ctx) {
      int offset = get_offset(nodeid);
      //if (threadIdx.x == 0) printf("GetData %d\n", nodeid);
      return (T*) (read_node(nodeid, ctx) + offset);      
    }    
    __inline__ __device__ void prefetch(int nodeid) {
      int block = get_block(nodeid);
      // printf("Prefetch %d %d\n", nodeid, block);
#ifdef ASYNC_READ
      disk->prefetch((uint64_t)block * DISKANN_PAGE_SIZE);
#endif
    }
    __inline__ __device__ void drop(const ReadCtx& ctx) {
      if (ctx.r != -1) {
        disk->release_page(ctx.page, ctx.r, ctx.start_idx);
      }
    }
                                                         
  };
}
