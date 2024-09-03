/* Copyright (c) 2018-2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
#include <cstdlib>
using namespace std;
#else
#include <stdlib.h>
#endif

// roctx header file
#include <roctx.h>
// roctracer extension API
#include <roctracer_ext.h>

#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <filesystem>
#include <link.h>
#include <unistd.h>

#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>

#ifdef __cplusplus
static thread_local const size_t msg_size = 512;
static thread_local char* msg_buf = NULL;
static thread_local char* message = NULL;
#else
static const size_t msg_size = 512;
static char* msg_buf = NULL;
static char* message = NULL;
#endif
void SPRINT(const char* fmt, ...) {
  if (msg_buf == NULL) {
    msg_buf = (char*)calloc(msg_size, 1);
    message = msg_buf;
  }

  va_list args;
  va_start(args, fmt);
  message += vsnprintf(message, msg_size - (message - msg_buf), fmt, args);
  va_end(args);
}
void SFLUSH() {
  if (msg_buf == NULL) abort();
  message = msg_buf;
  msg_buf[msg_size - 1] = 0;
  fprintf(stdout, "%s", msg_buf);
  fflush(stdout);
}

#if HIP_TEST
// hip header file
#include <hip/hip_runtime.h>
// Macro to call HIP API
#define CALL_HIP(call)                                                                             \
  do {                                                                                             \
    call;                                                                                          \
  } while (0);
#define CHECK_HIP(call)                                                                            \
  do {                                                                                             \
    hipError_t err = call;                                                                         \
    if (err != hipSuccess) {                                                                       \
      fprintf(stderr, "%s\n", hipGetErrorString(err));                                             \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)
#else
#define CALL_HIP(call)                                                                             \
  do {                                                                                             \
  } while (0)
#define CHECK_HIP(call)                                                                            \
  do {                                                                                             \
  } while (0)
#endif

#ifndef ITERATIONS
#define ITERATIONS 101
#endif
#define WIDTH 1024
#define NUM (WIDTH * WIDTH)
#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

#if HIP_TEST
// Device (Kernel) function, it must be void
__global__ void matrixTranspose(float* out, float* in, const int width) {
  int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

  out[y * width + x] = in[x * width + y];
}
#endif

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float* output, float* input, const unsigned int width) {
  for (unsigned int j = 0; j < width; j++) {
    for (unsigned int i = 0; i < width; i++) {
      output[i * width + j] = input[j * width + i];
    }
  }
}

// int iterations = ITERATIONS;
int iterations = 1000000;

void init_tracing();
void start_tracing();
void stop_tracing();

int main() {
  float* Matrix;
  float* TransposeMatrix;
  // float* cpuTransposeMatrix;

#if HIP_TEST
  float* gpuMatrix;
  float* gpuTransposeMatrix;
#endif

  int i;
  int errors = 0;

  init_tracing();

#if HIP_TEST
  int gpuCount = 1;
#if MGPU_TEST
  hipGetDeviceCount(&gpuCount);
  fprintf(stderr, "Number of GPUs: %d\n", gpuCount);
#endif
  iterations *= gpuCount;
#endif

  while (iterations-- > 0) {
    start_tracing();

#if HIP_TEST
    // set GPU
    const int devIndex = iterations % gpuCount;
    hipSetDevice(devIndex);

    hipDeviceProp_t devProp;
    CHECK_HIP(hipGetDeviceProperties(&devProp, 0));
    fprintf(stderr, "Device %d name: %s\n", devIndex, devProp.name);
#endif

    Matrix = (float*)malloc(NUM * sizeof(float));
    TransposeMatrix = (float*)malloc(NUM * sizeof(float));
    // cpuTransposeMatrix = (float*)malloc(NUM * sizeof(float));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
      Matrix[i] = (float)i * 10.0f;
    }

    // allocate the memory on the device side
    CHECK_HIP(hipMalloc((void**)&gpuMatrix, NUM * sizeof(float)));
    CHECK_HIP(hipMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float)));

    // correlation reagion32
    roctracer_activity_push_external_correlation_id(31);
    // correlation reagion32
    roctracer_activity_push_external_correlation_id(32);

    // Memory transfer from host to device
    CHECK_HIP(hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice));

    // correlation reagion33
    roctracer_activity_push_external_correlation_id(33);

    roctxMark("before hipLaunchKernel");
    roctxRangePush("hipLaunchKernel");

    // Lauching kernel from host
    CALL_HIP(hipLaunchKernelGGL(matrixTranspose,
                                dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                                dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0,
                                gpuTransposeMatrix, gpuMatrix, WIDTH));

    roctxMark("after hipLaunchKernel");

    // correlation reagion end
    roctracer_activity_pop_external_correlation_id(NULL);

    // Memory transfer from device to host
    roctxRangePush("hipMemcpy");

    CHECK_HIP(
        hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost));

    roctxRangePop();  // for "hipMemcpy"
    roctxRangePop();  // for "hipLaunchKernel"

    // correlation reagion end
    roctracer_activity_pop_external_correlation_id(NULL);

    /*
    // CPU MatrixTranspose computation
#if HIP_TEST
    matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, WIDTH);

    // verify the results
    double eps = 1.0E-6;
    for (i = 0; i < NUM; i++) {
      if (abs((double)TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
        errors++;
      }
    }
    if (errors != 0) {
      fprintf(stderr, "FAILED: %d errors\n", errors);
    } else {
      errors = 0;
      fprintf(stderr, "PASSED!\n");
    }
#endif
    */

    // free the resources on device side
    CHECK_HIP(hipFree(gpuMatrix));
    CHECK_HIP(hipFree(gpuTransposeMatrix));

    // correlation reagion end
    roctracer_activity_pop_external_correlation_id(NULL);
    // correlation reagion end
    roctracer_activity_pop_external_correlation_id(NULL);

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);
    // free(cpuTransposeMatrix);
  }

  stop_tracing();

  return errors;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HIP Callbacks/Activity tracing
//
#if 1
#include <roctracer_hip.h>
#include <roctracer_hsa.h>
#include <roctracer_roctx.h>

#include <unistd.h>
#include <sys/syscall.h> /* For SYS_xxx definitions */

// Macro to check ROC-tracer calls status
#define CHECK_ROCTRACER(call)                                                                      \
  do {                                                                                             \
    int err = call;                                                                                \
    if (err != 0) {                                                                                \
      fprintf(stderr, "%s\n", roctracer_error_string());                                           \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

static inline uint32_t GetTid() { return syscall(__NR_gettid); }
static inline uint32_t GetPid() { return syscall(__NR_getpid); }


// Runtime API callback function
void api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) {
  (void)arg;

  if (domain == ACTIVITY_DOMAIN_ROCTX) {
    const roctx_api_data_t* data = (const roctx_api_data_t*)(callback_data);
    fprintf(stdout, "rocTX <\"%s pid(%d) tid(%d)\">\n", data->args.message, GetPid(), GetTid());
    return;
  }
  const hip_api_data_t* data = (const hip_api_data_t*)(callback_data);
  SPRINT("<%s id(%u)\tcorrelation_id(%lu) %s pid(%d) tid(%d)> ",
         roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0), cid, data->correlation_id,
         (data->phase == ACTIVITY_API_PHASE_ENTER) ? "on-enter" : "on-exit", GetPid(), GetTid());
  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    switch (cid) {
      case HIP_API_ID_hipMemcpy:
        SPRINT("dst(%p) src(%p) size(0x%x) kind(%u)", data->args.hipMemcpy.dst,
               data->args.hipMemcpy.src, (uint32_t)(data->args.hipMemcpy.sizeBytes),
               (uint32_t)(data->args.hipMemcpy.kind));
        break;
      case HIP_API_ID_hipMalloc:
        SPRINT("ptr(%p) size(0x%x)", data->args.hipMalloc.ptr,
               (uint32_t)(data->args.hipMalloc.size));
        break;
      case HIP_API_ID_hipFree:
        SPRINT("ptr(%p)", data->args.hipFree.ptr);
        break;
      case HIP_API_ID_hipModuleLaunchKernel:
        SPRINT("kernel(\"%s\") stream(%p)", hipKernelNameRef(data->args.hipModuleLaunchKernel.f),
               data->args.hipModuleLaunchKernel.stream);
        break;
      default:
        break;
    }
  } else {
    switch (cid) {
      case HIP_API_ID_hipMalloc:
        SPRINT("*ptr(0x%p)", *(data->args.hipMalloc.ptr));
        break;
      default:
        break;
    }
  }
  SPRINT("\n");
  SFLUSH();
}
// Activity tracing callback
//   hipMalloc id(3) correlation_id(1): begin_ns(1525888652762640464) end_ns(1525888652762877067)
void activity_callback(const char* begin, const char* end, void* arg) {
  const roctracer_record_t* record = (const roctracer_record_t*)(begin);
  const roctracer_record_t* end_record = (const roctracer_record_t*)(end);

  SPRINT("\tActivity records:\n");
  while (record < end_record) {
    const char* name = roctracer_op_string(record->domain, record->op, record->kind);
    SPRINT("\t%s\tcorrelation_id(%lu) time_ns(%lu:%lu:%ld)", name, record->correlation_id,
           record->begin_ns, record->end_ns, record->end_ns - record->begin_ns);
    if (record->domain == ACTIVITY_DOMAIN_HIP_API) {
      SPRINT(" process_id(%u) thread_id(%u)", record->process_id, record->thread_id);
    } else if (record->domain == ACTIVITY_DOMAIN_HIP_OPS) {
      SPRINT(" device_id(%d) queue_id(%lu)", record->device_id, record->queue_id);
      if (record->op == HIP_OP_ID_DISPATCH) SPRINT(" kernel_name(%s)", record->kernel_name);
      if (record->op == HIP_OP_ID_COPY) SPRINT(" bytes(0x%zx)", record->bytes);
    } else if (record->domain == ACTIVITY_DOMAIN_HSA_OPS) {
      SPRINT(" se(%u) cycle(%lu) pc(%lx)", record->pc_sample.se, record->pc_sample.cycle,
             record->pc_sample.pc);
    } else if (record->domain == ACTIVITY_DOMAIN_EXT_API) {
      SPRINT(" external_id(%lu)", record->external_id);
    } else {
      fprintf(stdout, "Bad domain %d\n\n", record->domain);
      abort();
    }
    SPRINT("\n");
    SFLUSH();

    CHECK_ROCTRACER(roctracer_next_record(record, &record));
  }
}

struct MemoryPool
{
  MemoryPool(const roctracer_properties_t& properties) : properties_(properties) {
    // Pool definition: The memory pool is split in 2 buffers of equal size. When first initialized,
    // the write pointer points to the first element of the first buffer. When a buffer is full,  or
    // when Flush() is called, the write pointer moves to the other buffer.
    // Each buffer should be large enough to hold at least 2 activity records, as record pairs may
    // be written when external correlation ids are used.
    const size_t allocation_size =
        2 * std::max(2 * sizeof(roctracer_record_t), properties_.buffer_size);
    pool_begin_ = nullptr;
    AllocateMemory(&pool_begin_, allocation_size);
    assert(pool_begin_ != nullptr && "pool allocator failed");

    pool_end_ = pool_begin_ + allocation_size;
    buffer_begin_ = pool_begin_;
    buffer_end_ = buffer_begin_ + properties_.buffer_size;
    record_ptr_ = buffer_begin_;
    data_ptr_ = buffer_end_;

    // Create a consumer thread and wait for it to be ready to accept work.
    std::promise<void> ready;
    std::future<void> future = ready.get_future();
    consumer_thread_ = std::thread(&MemoryPool::ConsumerThreadLoop, this, std::move(ready));
    future.wait();
  }

  ~MemoryPool() {
    Flush();

    // Wait for the previous flush to complete, then send the exit signal.
    NotifyConsumerThread(nullptr, nullptr);
    consumer_thread_.join();

    // Free the pool's buffer memory.
    AllocateMemory(&pool_begin_, 0);
  }

  MemoryPool(const MemoryPool&) = delete;
  MemoryPool& operator=(const MemoryPool&) = delete;

  template <typename Record, typename Functor = std::function<void(Record& record, const void*)>>
  void Write(Record&& record, const void* data, size_t data_size, Functor&& store_data = {}) {
    assert(data != nullptr || data_size == 0);  // If data is null, then data_size must be 0

    std::lock_guard producer_lock(producer_mutex_);

    // The amount of memory reserved in the buffer to store data. If the data cannot fit because it
    // is larger than the buffer size minus one record, then the data won't be copied into the
    // buffer.
    size_t reserve_data_size =
        data_size <= (properties_.buffer_size - sizeof(Record)) ? data_size : 0;

    std::byte* next_record = record_ptr_ + sizeof(Record);
    if (next_record > (data_ptr_ - reserve_data_size)) {
      NotifyConsumerThread(buffer_begin_, record_ptr_);
      SwitchBuffers();
      next_record = record_ptr_ + sizeof(Record);
      assert(next_record <= buffer_end_ && "buffer size is less then the record size");
    }

    // Store data in the record. Copy the data first if it fits in the buffer
    // (reserve_data_size != 0).
    if (reserve_data_size) {
      data_ptr_ -= data_size;
      ::memcpy(data_ptr_, data, data_size);
      store_data(record, data_ptr_);
    } else if (data != nullptr) {
      store_data(record, data);
    }

    // Store the record into the buffer, and increment the write pointer.
    ::memcpy(record_ptr_, &record, sizeof(Record));
    record_ptr_ = next_record;

    // If the data does not fit in the buffer, flush the buffer with the record as is. We don't copy
    // the data so we make sure that the record and its data are processed by waiting until the
    // flush is complete.
    if (data != nullptr && reserve_data_size == 0) {
      NotifyConsumerThread(buffer_begin_, record_ptr_);
      SwitchBuffers();
      {
        std::unique_lock consumer_lock(consumer_mutex_);
        consumer_cond_.wait(consumer_lock, [this]() { return !consumer_arg_.valid; });
      }
    }
  }
  template <typename Record> void Write(Record&& record) {
    using DataPtr = void*;
    Write(std::forward<Record>(record), DataPtr(nullptr), 0, {});
  }

  // Flush the records and block until they are all made visible to the client.
  void Flush() {
    {
      std::lock_guard producer_lock(producer_mutex_);
      if (record_ptr_ == buffer_begin_) return;

      NotifyConsumerThread(buffer_begin_, record_ptr_);
      SwitchBuffers();
    }
    {
      // Wait for the current operation to complete.
      std::unique_lock consumer_lock(consumer_mutex_);
      consumer_cond_.wait(consumer_lock, [this]() { return !consumer_arg_.valid; });
    }
  }

  void SwitchBuffers() {
    buffer_begin_ = (buffer_end_ == pool_end_) ? pool_begin_ : buffer_end_;
    buffer_end_ = buffer_begin_ + properties_.buffer_size;
    record_ptr_ = buffer_begin_;
    data_ptr_ = buffer_end_;
  }

  void ConsumerThreadLoop(std::promise<void> ready) {
    std::unique_lock consumer_lock(consumer_mutex_);

    // This consumer is now ready to accept work.
    ready.set_value();

    while (true) {
      consumer_cond_.wait(consumer_lock, [this]() { return consumer_arg_.valid; });

      // begin == end == nullptr means the thread needs to exit.
      if (consumer_arg_.begin == nullptr && consumer_arg_.end == nullptr) break;

      properties_.buffer_callback_fun(reinterpret_cast<const char*>(consumer_arg_.begin),
                                      reinterpret_cast<const char*>(consumer_arg_.end),
                                      properties_.buffer_callback_arg);

      // Mark this operation as complete (valid=false) and notify all producers that may be
      // waiting for this operation to finish, or to start a new operation. See comment below in
      // NotifyConsumerThread().
      consumer_arg_.valid = false;
      consumer_cond_.notify_all();
    }
  }

  void NotifyConsumerThread(const std::byte* data_begin, const std::byte* data_end) {
    std::unique_lock consumer_lock(consumer_mutex_);

    // If consumer_arg_ is still in use (valid=true), then wait for the consumer thread to finish
    // processing the current operation. Multiple producers may wait here, one will be allowed to
    // continue once the consumer thread is idle and valid=false. This prevents a race condition
    // where operations would be lost if multiple producers could enter this critical section
    // (sequentially) before the consumer thread could re-acquire the consumer_mutex_ lock.
    consumer_cond_.wait(consumer_lock, [this]() { return !consumer_arg_.valid; });

    consumer_arg_.begin = data_begin;
    consumer_arg_.end = data_end;

    consumer_arg_.valid = true;
    consumer_cond_.notify_all();
  }

  void AllocateMemory(std::byte** ptr, size_t size) const {
    if (properties_.alloc_fun != nullptr) {
      // Use the custom allocator provided in the properties.
      properties_.alloc_fun(reinterpret_cast<char**>(ptr), size, properties_.alloc_arg);
      return;
    }

    // No custom allocator was provided so use the default malloc/realloc/free allocator.
    if (*ptr == nullptr) {
      *ptr = static_cast<std::byte*>(malloc(size));
    } else if (size != 0) {
      *ptr = static_cast<std::byte*>(realloc(*ptr, size));
    } else {
      free(*ptr);
      *ptr = nullptr;
    }
  }

  // Properties used to create the memory pool.
  const roctracer_properties_t properties_;

  // Pool definition
  std::byte* pool_begin_;
  std::byte* pool_end_;
  std::byte* buffer_begin_;
  std::byte* buffer_end_;
  std::byte* record_ptr_;
  std::byte* data_ptr_;
  std::mutex producer_mutex_;

  // Consumer thread
  std::thread consumer_thread_;
  struct {
    const std::byte* begin;
    const std::byte* end;
    bool valid = false;
  } consumer_arg_;

  std::mutex consumer_mutex_;
  std::condition_variable consumer_cond_;
};

static MemoryPool* default_memory_pool = nullptr;

template <typename Loader>
class BaseLoader {
 protected:
  BaseLoader(const char* pattern) {
    // Iterate through the process' loaded shared objects and try to dlopen the first entry with a
    // file name starting with the given 'pattern'. This allows the loader to acquire a handle
    // to the target library iff it is already loaded. The handle is used to query symbols
    // exported by that library.

    auto callback = [this, pattern](dl_phdr_info* info) {
      namespace fs = std::filesystem;
      if (handle_ == nullptr &&
          fs::path(info->dlpi_name).filename().string().rfind(pattern, 0) == 0)
        handle_ = ::dlopen(info->dlpi_name, RTLD_LAZY);
    };
    dl_iterate_phdr(
        [](dl_phdr_info* info, size_t size, void* data) {
          (*reinterpret_cast<decltype(callback)*>(data))(info);
          return 0;
        },
        &callback);
  }

  ~BaseLoader() {
    if (handle_ != nullptr) ::dlclose(handle_);
  }

  BaseLoader(const BaseLoader&) = delete;
  BaseLoader& operator=(const BaseLoader&) = delete;

 public:
  template <typename FunctionPtr> FunctionPtr GetFun(const char* symbol) const {
    auto function_ptr = reinterpret_cast<FunctionPtr>(::dlsym(handle_, symbol));
    return function_ptr;
  }

  static inline Loader& Instance() {
    static Loader instance;
    return instance;
  }

 private:
  void* handle_;
};

class HipLoader : public BaseLoader<HipLoader> {
 private:
  friend HipLoader& BaseLoader::Instance();
  HipLoader() : BaseLoader("libamdhip64.so") {}

 public:
  int GetStreamDeviceId(hipStream_t stream) const {
    static auto function = GetFun<int (*)(hipStream_t stream)>("hipGetStreamDeviceId");
    return function(stream);
  }

  const char* KernelNameRef(const hipFunction_t f) const {
    static auto function = GetFun<const char* (*)(const hipFunction_t f)>("hipKernelNameRef");
    return function(f);
  }

  const char* KernelNameRefByPtr(const void* host_function, hipStream_t stream = nullptr) const {
    static auto function = GetFun<const char* (*)(const void* hostFunction, hipStream_t stream)>(
        "hipKernelNameRefByPtr");
    return function(host_function, stream);
  }

  const char* GetOpName(unsigned op) const {
    static auto function = GetFun<const char* (*)(unsigned op)>("hipGetCmdName");
    return function(op);
  }

  const char* ApiName(uint32_t id) const {
    static auto function = GetFun<const char* (*)(uint32_t id)>("hipApiName");
    return function(id);
  }

  void RegisterTracerCallback(int (*callback)(activity_domain_t domain, uint32_t operation_id,
                                              void* data)) const {
    static auto function = GetFun<void (*)(int (*callback)(
        activity_domain_t domain, uint32_t operation_id, void* data))>("hipRegisterTracerCallback");
    return function(callback);
  }
};

void TracerCallbackHipOpsKernel(activity_record_t* record) {
  default_memory_pool->Write(
      *record, record->kernel_name, strlen(record->kernel_name) + 1,
      [](auto& record, const void* data) {
        record.kernel_name = static_cast<const char*>(data);
      });
  SPRINT("!!! %s\n", record->kernel_name);
  SFLUSH();
}

void TracerCallbackHipOpsOther1(activity_record_t* record) {
  default_memory_pool->Write(*record);
}

void TracerCallbackHipOpsOther2() {
}

int TracerCallbackHipOps(activity_domain_t domain, uint32_t operation_id, void* data) {
  if (auto record = static_cast<activity_record_t*>(data)) {
    // If the record is for a kernel dispatch, write the kernel name in the pool's data,
    // and make the record point to it. Older HIP runtimes do not provide a kernel
    // name, so record.kernel_name might be null.
    if (operation_id == HIP_OP_ID_DISPATCH && record->kernel_name != nullptr) {
      TracerCallbackHipOpsKernel(record);
    } else {
      TracerCallbackHipOpsOther1(record);
    }
  } else {
    TracerCallbackHipOpsOther2();
  }
  return 0;
}

int TracerCallback(activity_domain_t domain, uint32_t operation_id, void* data) {
  switch (domain) {
    case ACTIVITY_DOMAIN_HIP_OPS:
      return TracerCallbackHipOps(domain, operation_id, data);
    default:
      break;
  }
  return -1;
}

void init_tracing() {
  roctracer_properties_t properties;
  memset(&properties, 0, sizeof(roctracer_properties_t));
  properties.buffer_size = 0x1000;
  properties.buffer_callback_fun = activity_callback;
  default_memory_pool = new MemoryPool(properties);
  HipLoader::Instance().RegisterTracerCallback(TracerCallback);
}

void start_tracing() {}
void stop_tracing() {}

/*
// Init tracing routine
void init_tracing() {
  fprintf(stderr, "# INIT #############################\n");
  // roctracer properties
  roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, NULL);
  // Allocating tracing pool
  roctracer_properties_t properties;
  memset(&properties, 0, sizeof(roctracer_properties_t));
  properties.buffer_size = 0x1000;
  properties.buffer_callback_fun = activity_callback;
  CHECK_ROCTRACER(roctracer_open_pool(&properties));
  // Enable HIP API callbacks
  CHECK_ROCTRACER(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, api_callback, NULL));
  // Enable HIP activity tracing
#if HIP_API_ACTIVITY_ON
  CHECK_ROCTRACER(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
#endif
  CHECK_ROCTRACER(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS));
  // Enable PC sampling
  CHECK_ROCTRACER(roctracer_enable_op_activity(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_RESERVED1));
  // Enable rocTX
  CHECK_ROCTRACER(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX, api_callback, NULL));
}

// Start tracing routine
void start_tracing() {
  fprintf(stderr, "# START (%d) #############################\n", iterations);
  // Start
  if ((iterations & 1) == 1)
    roctracer_start();
  else
    roctracer_stop();
  fprintf(stderr, "# START #############################\n");
  roctracer_start();
}

// Stop tracing routine
void stop_tracing() {
  CHECK_ROCTRACER(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API));
#if HIP_API_ACTIVITY_ON
  CHECK_ROCTRACER(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
#endif
  CHECK_ROCTRACER(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS));
  CHECK_ROCTRACER(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HSA_OPS));
  CHECK_ROCTRACER(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX));
  CHECK_ROCTRACER(roctracer_flush_activity());
  fprintf(stderr, "# STOP  #############################\n");
}
*/
#else
void init_tracing() {}
void start_tracing() {}
void stop_tracing() {}
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
