"""러너 안에서 쓰는 이벤트 기록기. 한 줄에 wall_ns와 mono_ns를 같이 둬 nsys, iostat, 블록 IO와 정렬 가능.
- Events(path): emit(kind, **data)를 즉시 파일에 흘려 씀(죽어도 남음). phase(name)는 단계 경계 마커.
- wrap_transport(cls, name, op, ev): expfs 전송 함수를 감싸 begin/end 이벤트와 outstanding 수를 남김."""
import json, os, threading, time

class Events:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "a", buffering=1); self.lk = threading.Lock()
        self.outs = {"r": 0, "w": 0}; self.io = []      # (mono_s0, mono_s1, bytes, op) 메모리 사본. 집계용
    def emit(self, kind, **data):
        row = dict(kind=kind, wall_ns=time.time_ns(), mono_ns=time.monotonic_ns(), tid=threading.get_ident(), **data)
        with self.lk: self.f.write(json.dumps(row) + "\n")
    def phase(self, name, **data):
        self.emit("phase", name=name, **data); return time.monotonic()
    def wrap_transport(self, cls, name, op):
        fn = getattr(cls, name); ev = self
        def inner(self_, path, spans, chunk_bytes):
            nb = sum(s[2] for s in spans)
            with ev.lk: ev.outs[op] += 1; outs = ev.outs[op]
            t0 = time.monotonic(); ev.emit(f"kv_{op}_begin", path=os.path.basename(path), bytes=nb, outstanding=outs)
            try:
                return fn(self_, path, spans, chunk_bytes)
            finally:
                t1 = time.monotonic()
                with ev.lk: ev.outs[op] -= 1; ev.io.append((t0, t1, nb, op))
                ev.emit(f"kv_{op}_end", path=os.path.basename(path), bytes=nb, dur_ms=round((t1 - t0) * 1e3, 3))
        setattr(cls, name, inner)
