# Phase 1: cuFile 마이크로벤치 (vLLM 밖) — Phase 2 투자 판정 게이트.
#
# 행렬: geometry(모델 3 × chunk-tokens 3 × span 구조 3) × transport 4 × op(write/read)
#   gds_reg     — op별 필요한 쪽 span만 cuFileBufRegister (쓰기=src, 읽기=dst;
#                 BAR1 실패는 정상 분기로 기록)
#   gds_unreg   — 미등록 포인터 직접 (nvidia-fs 내부 bounce buffer 경로)
#   gds_staging — 16MiB 등록 staging GPU buffer + D2D 경유
#   posix       — pinned host bounce + 코얼레싱 단일 O_DIRECT pwrite/preadv (vLLM 현행 모사)
#   (cuFile batch API는 v1 제외 — README 기록)
#
# 지표: p50/p95/p99(ms), GiB/s, cpu_util, 등록시간, nvidia-fs Mmap/Bar1 델타, checksum.
# checksum은 데이터 정합성 검증(경로 순수성은 path_classify.py TRACE 런에서).
import csv
import os
import statistics
import time

import torch

from gdslib import Gds, GdsError

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

GEOMS = [("opt125m", 589824, 12), ("llama8b", 2 << 20, 32), ("opt67b", 8 << 20, 32)]
TOKEN_MULT = [1, 4, 16]  # 16/64/256 tokens
REPS, WARMUP = 12, 2
GAP = 64 << 10
STAGING_BYTES = 16 << 20


def nvfs_map_counts():
    m = b1 = 0
    for line in open("/proc/driver/nvidia-fs/stats"):
        if line.startswith("Mmap"):
            m = int(line.split("n=")[1].split()[0])
        elif line.startswith("Bar1-map"):
            b1 = int(line.split("n=")[1].split()[0])
    return m, b1


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


class SpanBuf:
    def __init__(self, chunk, n_spans):
        self.chunk, self.n = chunk, n_spans
        self.span = chunk // n_spans
        assert self.span % 4096 == 0, (chunk, n_spans)
        self.stride = self.span + GAP
        self.t_src = torch.empty(n_spans * self.stride, dtype=torch.uint8, device="cuda")
        self.t_dst = torch.empty(n_spans * self.stride, dtype=torch.uint8, device="cuda")
        self.src = [self.t_src.data_ptr() + i * self.stride for i in range(n_spans)]
        self.dst = [self.t_dst.data_ptr() + i * self.stride for i in range(n_spans)]

    def fill_pattern(self):
        self.t_src.random_(0, 251)
        self.t_dst.zero_()
        torch.cuda.synchronize()

    def views(self, t):
        return [t[i * self.stride: i * self.stride + self.span] for i in range(self.n)]

    def check(self):
        return all(bool((a == b).all()) for a, b in zip(self.views(self.t_src), self.views(self.t_dst)))


def run_cell(g, meta, transport, fh, fd, buf, pin, pin_mv, staging, rows):
    span, chunk = buf.span, buf.chunk

    def d2d(dst_t, dst_off, src_t, src_off, n):
        dst_t[dst_off:dst_off + n].copy_(src_t[src_off:src_off + n])

    def do_write(rep):
        off0 = rep * chunk
        if transport == "posix":
            for i, v in enumerate(buf.views(buf.t_src)):
                pin[i * span:(i + 1) * span].copy_(v, non_blocking=True)
            torch.cuda.synchronize()
            os.pwrite(fd, pin_mv[:chunk], off0)
        elif transport == "gds_staging":
            for i, p in enumerate(buf.src):
                rel, done = p - buf.t_src.data_ptr(), 0
                while done < span:
                    piece = min(STAGING_BYTES, span - done)
                    d2d(staging, 0, buf.t_src, rel + done, piece)
                    torch.cuda.synchronize()
                    g.write(fh, staging.data_ptr(), piece, off0 + i * span + done)
                    done += piece
        else:
            for i, p in enumerate(buf.src):
                g.write(fh, p, span, off0 + i * span)

    def do_read(rep):
        off0 = rep * chunk
        if transport == "posix":
            os.preadv(fd, [pin_mv[:chunk]], off0)
            for i, v in enumerate(buf.views(buf.t_dst)):
                v.copy_(pin[i * span:(i + 1) * span], non_blocking=True)
            torch.cuda.synchronize()
        elif transport == "gds_staging":
            for i, p in enumerate(buf.dst):
                rel, done = p - buf.t_dst.data_ptr(), 0
                while done < span:
                    piece = min(STAGING_BYTES, span - done)
                    g.read(fh, staging.data_ptr(), piece, off0 + i * span + done)
                    d2d(buf.t_dst, rel + done, staging, 0, piece)
                    torch.cuda.synchronize()
                    done += piece
        else:
            for i, p in enumerate(buf.dst):
                g.read(fh, p, span, off0 + i * span)

    def register(ptrs):
        t0 = time.perf_counter()
        done = []
        try:
            for p in ptrs:
                g.buf_register(p, span)
                done.append(p)
            return done, time.perf_counter() - t0, ""
        except GdsError as e:
            for p in done:
                g.buf_deregister(p)
            return None, None, str(e)

    for op, fn, ptrs in (("write", do_write, buf.src), ("read", do_read, buf.dst)):
        reg_time, reg_fail, registered = None, "", None
        m0, b0 = nvfs_map_counts()
        if transport == "gds_reg":
            registered, reg_time, reg_fail = register(ptrs)
            if registered is None:
                rows.append({**meta, "transport": transport, "op": op, "p50": None,
                             "p95": None, "p99": None, "gibps": None, "cpu_util": None,
                             "reg_s": None, "reg_fail": reg_fail, "mmap_d": 0,
                             "bar1_d": 0, "checksum": None})
                continue
        laps, cpu = [], []
        for rep in range(REPS):
            c0, t0 = time.process_time(), time.perf_counter()
            fn(rep)
            dt = time.perf_counter() - t0
            if rep >= WARMUP:
                laps.append(dt)
                cpu.append((time.process_time() - c0) / dt if dt > 0 else 0)
        m1, b1 = nvfs_map_counts()
        med = statistics.median(laps)
        rows.append({**meta, "transport": transport, "op": op,
                     "p50": round(med * 1e3, 3), "p95": round(pct(laps, 95) * 1e3, 3),
                     "p99": round(pct(laps, 99) * 1e3, 3),
                     "gibps": round(chunk / med / (1 << 30), 3),
                     "cpu_util": round(statistics.median(cpu), 3),
                     "reg_s": round(reg_time, 4) if reg_time is not None else None,
                     "reg_fail": reg_fail, "mmap_d": m1 - m0, "bar1_d": b1 - b0,
                     "checksum": None})
        if registered:
            for p in registered:
                g.buf_deregister(p)

    # 정합성 검증 (비계측, 미등록 경로)
    buf.fill_pattern()
    do_write(0)
    do_read(0)
    rows[-1]["checksum"] = buf.check()


def main():
    torch.cuda.init()
    torch.empty(1, device="cuda")
    g = Gds()
    staging = torch.empty(STAGING_BYTES, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    g.buf_register(staging.data_ptr(), STAGING_BYTES)

    rows = []

    def flush_rows():
        if not rows:
            return
        out = os.path.join(HERE, "results.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    t_start = time.perf_counter()
    for geom_name, base_chunk, L in GEOMS:
        for mult in TOKEN_MULT:
            chunk = base_chunk * mult
            fd = os.open(os.path.join(DATA, f"{geom_name}_x{mult}.bin"),
                         os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o644)
            os.posix_fallocate(fd, 0, REPS * chunk)
            fh = g.handle_register(fd)
            pin = torch.empty(chunk, dtype=torch.uint8, pin_memory=True)
            pin_mv = memoryview(pin.numpy())
            for span_name, n_spans in (("1", 1), ("L", L), ("2L", 2 * L)):
                buf = SpanBuf(chunk, n_spans)
                buf.fill_pattern()
                meta = dict(geom=geom_name, chunk=chunk, tok_mult=mult,
                            spans=n_spans, span_name=span_name)
                for transport in ("gds_reg", "gds_unreg", "gds_staging", "posix"):
                    run_cell(g, meta, transport, fh, fd, buf, pin, pin_mv, staging, rows)
                flush_rows()
                print(f"[{time.perf_counter()-t_start:7.1f}s] {geom_name} x{mult} "
                      f"span{span_name} done", flush=True)
                del buf
                torch.cuda.empty_cache()
            g.handle_deregister(fh)
            os.close(fd)
            del pin_mv, pin

    g.buf_deregister(staging.data_ptr())
    g.close()
    flush_rows()
    print(f"wrote {len(rows)} rows -> results.csv")


if __name__ == "__main__":
    main()
