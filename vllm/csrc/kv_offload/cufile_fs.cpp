// SPDX-License-Identifier: Apache-2.0
// cuFile 파일시스템 KV 오프로드 전송기(native).
//   GPU KV 블록 ↔ 파일을 cuFile로 직접 옮긴다. CPU 메모리를 거치지 않는다.
//   IO 스레드는 C++ 스레드라 GIL을 잡지 않고, submit/wait/get_finished는 GIL을 놓고 돈다.
//   chunk 파일 레이아웃 = [tensor0: bpc 페이지][tensor1: bpc 페이지]... (expfs와 동일)
//   store: <path>.tmp 에 쓰고 rename → 파일 존재 = 로드 가능(원자적). 실패 시 tmp 삭제.
//   pause_writes/resume_writes: 가중치를 SSD에서 스트리밍하는 동안 쓰기 풀만 멈춘다.
//   pause_reads/resume_reads: 같은 방식으로 읽기 풀만 멈춘다(split-source KV의 tail 적재를 SSD 창 밖으로 미룰 때).
#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <atomic>
#include <chrono>
#include <memory>
#include <set>
#include <condition_variable>
#include <deque>
#include <mutex>
#ifdef CUFILE_FS_NVTX
#include <nvtx3/nvToolsExt.h>
#define NVTX_PUSH(x) nvtxRangePushA(x)
#define NVTX_POP() nvtxRangePop()
#define NVTX_MARK(x) nvtxMarkA(x)
#else
#define NVTX_PUSH(x)
#define NVTX_POP()
#define NVTX_MARK(x)
#endif
#include <algorithm>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Span { int t; size_t gpu_off; size_t size; size_t file_off; };
struct Chunk { std::string path; std::vector<Span> spans; };
struct Job {
  int64_t id; bool is_store; std::vector<Chunk> chunks; cudaEvent_t ev = nullptr;
  std::atomic<int> remaining{0}; std::atomic<bool> ok{true};
};

class CuFileFs {
 public:
  CuFileFs(std::vector<uintptr_t> base_ptrs, std::vector<size_t> tensor_bytes,
           std::vector<size_t> page_sizes, int bpc, int n_read, int n_write, bool register_tensors)
      : base_(std::move(base_ptrs)), tbytes_(std::move(tensor_bytes)), page_(std::move(page_sizes)), bpc_(bpc) {
    CUfileError_t e = cuFileDriverOpen();
    if (e.err != 0 && e.err != CU_FILE_DRIVER_ALREADY_OPEN) throw std::runtime_error("cuFileDriverOpen err=" + std::to_string(e.err));
    region_off_.resize(page_.size()); size_t off = 0;
    for (size_t t = 0; t < page_.size(); ++t) { region_off_[t] = off; off += page_[t] * bpc_; }
    chunk_bytes_ = off;
    if (register_tensors) {
      for (size_t t = 0; t < base_.size(); ++t) {
        CUfileError_t r = cuFileBufRegister((void*)base_[t], tbytes_[t], 0);
        if (r.err != 0) { for (size_t u = 0; u < t; ++u) cuFileBufDeregister((void*)base_[u]); registered_ = 0; register_err_ = r.err; break; }
        registered_++;
      }
    }
    for (int i = 0; i < n_read; ++i) threads_.emplace_back([this] { loop(read_q_, read_cv_, false); });
    for (int i = 0; i < n_write; ++i) threads_.emplace_back([this] { loop(write_q_, write_cv_, true); });
  }
  ~CuFileFs() { shutdown(); }

  size_t chunk_bytes() const { return chunk_bytes_; }
  int registered() const { return registered_; }
  int register_err() const { return register_err_; }

  // bids[c] = chunk c의 GPU 블록 id 목록(논리 페이지 j0[c]부터), paths[c] = chunk 파일 경로
  void submit(int64_t id, bool is_store, std::vector<std::string> paths, std::vector<std::vector<int64_t>> bids,
              std::vector<int> j0s, uintptr_t event) {
    auto job = std::make_shared<Job>(); job->id = id; job->is_store = is_store; job->ev = (cudaEvent_t)event;
    for (size_t c = 0; c < paths.size(); ++c) job->chunks.push_back(Chunk{paths[c], spans(bids[c], j0s[c])});
    job->remaining = (int)job->chunks.size();
    { std::lock_guard<std::mutex> g(mu_); pending_.insert(id); }
    if (job->chunks.empty()) { finish(job); return; }
    auto& q = is_store ? write_q_ : read_q_; auto& cv = is_store ? write_cv_ : read_cv_;
    { std::lock_guard<std::mutex> g(qmu_); for (size_t c = 0; c < job->chunks.size(); ++c) q.push_back({job, c});
      if (is_store && wpaused_.load() && !pause_counting_) { pause_counting_ = true; pause_since_ = std::chrono::steady_clock::now(); }
      if (!is_store && rpaused_.load() && !r_pause_counting_) { r_pause_counting_ = true; r_pause_since_ = std::chrono::steady_clock::now(); } }
    cv.notify_all();
  }
  // 쓰기 풀만 멈춘다. 큐에 쌓인 job은 그대로 남고, resume 후 이어서 처리된다.
  void pause_writes() {
    NVTX_MARK("kv_writes_pause");
    std::lock_guard<std::mutex> g(qmu_);
    if (wpaused_.load()) return;
    wpaused_.store(true);
    // "큐에 일이 있는데 멈춰 있던 시간"만 센다. 지금 비어 있으면 submit 때부터 센다.
    pause_counting_ = !write_q_.empty();
    if (pause_counting_) pause_since_ = std::chrono::steady_clock::now();
  }
  void resume_writes() {
    NVTX_MARK("kv_writes_resume");
    { std::lock_guard<std::mutex> g(qmu_); resume_locked(); }
    write_cv_.notify_all();
  }
  bool writes_paused() const { return wpaused_.load(); }
  // 읽기 풀만 멈춘다. 큐에 쌓인 job은 그대로 남고, resume 후 이어서 처리된다.
  void pause_reads() {
    NVTX_MARK("kv_reads_pause");
    std::lock_guard<std::mutex> g(qmu_);
    if (rpaused_.load()) return;
    rpaused_.store(true);
    r_pause_counting_ = !read_q_.empty();
    if (r_pause_counting_) r_pause_since_ = std::chrono::steady_clock::now();
  }
  void resume_reads() {
    NVTX_MARK("kv_reads_resume");
    { std::lock_guard<std::mutex> g(qmu_); resume_reads_locked(); }
    read_cv_.notify_all();
  }
  bool reads_paused() const { return rpaused_.load(); }

  std::vector<std::pair<int64_t, bool>> get_finished() {
    std::lock_guard<std::mutex> g(mu_); auto out = std::move(finished_); finished_.clear(); return out;
  }
  void wait(std::vector<int64_t> ids) {
    for (;;) {
      { std::lock_guard<std::mutex> g(mu_); bool any = false; for (auto i : ids) if (pending_.count(i)) { any = true; break; } if (!any) return; }
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  }
  py::dict stats() {
    py::dict d; d["reads"] = reads_.load(); d["writes"] = writes_.load(); d["read_bytes"] = rbytes_.load(); d["write_bytes"] = wbytes_.load();
    d["read_busy_ns"] = rns_.load(); d["write_busy_ns"] = wns_.load(); d["errors"] = errors_.load(); d["registered_tensors"] = registered_;
    d["outstanding_reads"] = out_r_.load(); d["outstanding_writes"] = out_w_.load();
    d["writes_paused"] = wpaused_.load(); d["reads_paused"] = rpaused_.load();
    d["w_ev_ns"] = w_ev_ns_.load(); d["w_open_ns"] = w_open_ns_.load(); d["w_io_ns"] = w_io_ns_.load(); d["w_fin_ns"] = w_fin_ns_.load(); d["w_calls"] = w_calls_.load();
    d["r_open_ns"] = r_open_ns_.load(); d["r_io_ns"] = r_io_ns_.load(); d["r_fin_ns"] = r_fin_ns_.load(); d["r_calls"] = r_calls_.load();
    { std::lock_guard<std::mutex> g(qmu_); long p = paused_ns_.load(); long rp = r_paused_ns_.load();
      auto now = std::chrono::steady_clock::now();
      if (pause_counting_) p += std::chrono::duration_cast<std::chrono::nanoseconds>(now - pause_since_).count();
      if (r_pause_counting_) rp += std::chrono::duration_cast<std::chrono::nanoseconds>(now - r_pause_since_).count();
      d["paused_ns"] = p; d["read_paused_ns"] = rp; }
    { std::lock_guard<std::mutex> g(mu_); d["last_error"] = last_err_; } return d;
  }
  void shutdown() {
    if (stopped_) return; stopped_ = true;
    { std::lock_guard<std::mutex> g(qmu_); resume_locked(); resume_reads_locked(); } read_cv_.notify_all(); write_cv_.notify_all();
    for (auto& t : threads_) if (t.joinable()) t.join();
    for (int t = 0; t < registered_; ++t) cuFileBufDeregister((void*)base_[t]);
    registered_ = 0;
  }

 private:
  using Task = std::pair<std::shared_ptr<Job>, size_t>;
  std::vector<Span> spans(const std::vector<int64_t>& bids, int j0) {
    std::vector<Span> out;
    for (size_t t = 0; t < page_.size(); ++t) {
      size_t page = page_[t], region = region_off_[t]; size_t k = 0;
      while (k < bids.size()) {
        int64_t b0 = bids[k]; size_t n = 1;
        while (k + n < bids.size() && bids[k + n] == b0 + (int64_t)n) ++n;
        out.push_back(Span{(int)t, (size_t)b0 * page, n * page, region + (j0 + k) * page});
        k += n;
      }
    }
    return out;
  }
  void loop(std::deque<Task>& q, std::condition_variable& cv, bool is_write) {
    for (;;) {
      Task task;
      { std::unique_lock<std::mutex> lk(qmu_);
        // pause 중인 풀의 스레드는 큐를 건드리지 않고 기다린다(다른 풀은 영향 없음).
        auto blocked = [&] { return is_write ? wpaused_.load() : rpaused_.load(); };
        cv.wait(lk, [&] { return stopped_ || (!q.empty() && !blocked()); });
        if (q.empty() || blocked()) { if (stopped_) return; continue; }
        task = std::move(q.front()); q.pop_front(); }
      run(task.first, task.second);
    }
  }
  // qmu_를 잡은 채로 호출한다. 진행 중이던 pause 구간을 누적하고 플래그를 푼다.
  void resume_locked() {
    if (!wpaused_.load()) return;
    wpaused_.store(false);
    if (pause_counting_) {
      paused_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - pause_since_).count();
      pause_counting_ = false;
    }
  }
  // qmu_를 잡은 채로 호출한다.
  void resume_reads_locked() {
    if (!rpaused_.load()) return;
    rpaused_.store(false);
    if (r_pause_counting_) {
      r_paused_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - r_pause_since_).count();
      r_pause_counting_ = false;
    }
  }
  static std::string errstr(const char* what, long n) { return std::string(what) + " ret=" + std::to_string(n) + " errno=" + std::to_string(errno); }
  void run(const std::shared_ptr<Job>& job, size_t c) {
    const Chunk& ch = job->chunks[c]; bool ok = true; std::string tmp = ch.path + ".tmp";
    // 타임라인에서 어느 KV 파일인지 보이게: 방향, job id, chunk 순번, 파일 이름(블록 해시), 크기
    { size_t tot = 0; for (const Span& sp : ch.spans) tot += sp.size;
      std::string nm = std::string(job->is_store ? "kv_store" : "kv_load") + " job=" + std::to_string(job->id) + " c=" + std::to_string(c)
                       + " " + ch.path.substr(ch.path.find_last_of('/') + 1) + " " + std::to_string(tot >> 20) + "MiB";
      NVTX_PUSH(nm.c_str()); }
    using clk = std::chrono::steady_clock; auto t0 = clk::now();
    auto lap = [&](clk::time_point& t) { auto n = clk::now(); long d = std::chrono::duration_cast<std::chrono::nanoseconds>(n - t).count(); t = n; return d; };
    if (job->is_store) out_w_++; else out_r_++;
    if (job->ok.load()) {
      clk::time_point tp = t0;
      if (job->is_store && job->ev) { cudaError_t ce = cudaEventSynchronize(job->ev); if (ce != cudaSuccess) ok = false; }
      w_ev_ns_ += job->is_store ? lap(tp) : 0;
      int fd = -1;
      if (ok) {
        if (job->is_store) { std::string d = ch.path.substr(0, ch.path.find_last_of('/')); mkdir_p(d); fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644); }
        else fd = open(ch.path.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) ok = false;
      }
      CUfileHandle_t fh = nullptr;
      if (ok) {
        CUfileDescr_t desc{}; desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; desc.handle.fd = fd;
        CUfileError_t e = cuFileHandleRegister(&fh, &desc); if (e.err != 0) ok = false;
      }
      (job->is_store ? w_open_ns_ : r_open_ns_) += lap(tp);
      if (ok) {
        for (const Span& s : ch.spans) {
          void* p = (void*)(base_[s.t] + s.gpu_off);
          ssize_t n = job->is_store ? cuFileWrite(fh, p, s.size, (off_t)s.file_off, 0) : cuFileRead(fh, p, s.size, (off_t)s.file_off, 0);
          if (n != (ssize_t)s.size) { ok = false; std::lock_guard<std::mutex> g(mu_); last_err_ = errstr(job->is_store ? "cuFileWrite" : "cuFileRead", (long)n); break; }
          if (job->is_store) wbytes_ += s.size; else rbytes_ += s.size;
          (job->is_store ? w_calls_ : r_calls_)++;
        }
      }
      (job->is_store ? w_io_ns_ : r_io_ns_) += lap(tp);
      if (fh) cuFileHandleDeregister(fh);
      if (fd >= 0) close(fd);
      if (job->is_store) { if (ok && rename(tmp.c_str(), ch.path.c_str()) != 0) ok = false; if (!ok) unlink(tmp.c_str()); }
      (job->is_store ? w_fin_ns_ : r_fin_ns_) += lap(tp);
    } else ok = false;
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0).count();
    if (job->is_store) { writes_++; wns_ += ns; out_w_--; } else { reads_++; rns_ += ns; out_r_--; }
    NVTX_POP();
    if (!ok) { errors_++; job->ok = false; }
    if (--job->remaining == 0) finish(job);
  }
  void finish(const std::shared_ptr<Job>& job) {
    std::lock_guard<std::mutex> g(mu_); pending_.erase(job->id); finished_.emplace_back(job->id, job->ok.load());
  }
  static void mkdir_p(const std::string& d) {
    for (size_t i = 1; i < d.size(); ++i) if (d[i] == '/') mkdir(d.substr(0, i).c_str(), 0755);
    mkdir(d.c_str(), 0755);
  }

  std::vector<uintptr_t> base_; std::vector<size_t> tbytes_, page_, region_off_; int bpc_; size_t chunk_bytes_ = 0;
  int registered_ = 0, register_err_ = 0;
  std::vector<std::thread> threads_; std::deque<Task> read_q_, write_q_; std::condition_variable read_cv_, write_cv_;
  std::mutex qmu_, mu_; std::set<int64_t> pending_; std::vector<std::pair<int64_t, bool>> finished_; std::atomic<bool> stopped_{false}, wpaused_{false}, rpaused_{false};
  bool pause_counting_ = false; std::chrono::steady_clock::time_point pause_since_{};  // qmu_ 보호
  bool r_pause_counting_ = false; std::chrono::steady_clock::time_point r_pause_since_{};  // qmu_ 보호
  std::atomic<long> reads_{0}, writes_{0}, rbytes_{0}, wbytes_{0}, rns_{0}, wns_{0}, errors_{0}, out_r_{0}, out_w_{0}, paused_ns_{0}, r_paused_ns_{0};
  std::atomic<long> w_ev_ns_{0}, w_open_ns_{0}, w_io_ns_{0}, w_fin_ns_{0}, w_calls_{0}, r_open_ns_{0}, r_io_ns_{0}, r_fin_ns_{0}, r_calls_{0};  // 파일 1개 처리의 구간별 합
  std::string last_err_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<CuFileFs>(m, "CuFileFs")
      .def(py::init<std::vector<uintptr_t>, std::vector<size_t>, std::vector<size_t>, int, int, int, bool>())
      .def("submit", &CuFileFs::submit, py::call_guard<py::gil_scoped_release>())
      .def("pause_writes", &CuFileFs::pause_writes, py::call_guard<py::gil_scoped_release>())
      .def("resume_writes", &CuFileFs::resume_writes, py::call_guard<py::gil_scoped_release>())
      .def("writes_paused", &CuFileFs::writes_paused)
      .def("pause_reads", &CuFileFs::pause_reads, py::call_guard<py::gil_scoped_release>())
      .def("resume_reads", &CuFileFs::resume_reads, py::call_guard<py::gil_scoped_release>())
      .def("reads_paused", &CuFileFs::reads_paused)
      .def("get_finished", &CuFileFs::get_finished, py::call_guard<py::gil_scoped_release>())
      .def("wait", &CuFileFs::wait, py::call_guard<py::gil_scoped_release>())
      .def("stats", &CuFileFs::stats)
      .def("shutdown", &CuFileFs::shutdown, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("chunk_bytes", &CuFileFs::chunk_bytes)
      .def_property_readonly("registered", &CuFileFs::registered)
      .def_property_readonly("register_err", &CuFileFs::register_err);
}
