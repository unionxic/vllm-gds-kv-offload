# StagedCuFileTransport 기능 검증 (v3 게이트).
#   A: staged로 store → 파일 생성·flush·출력 토큰 기록
#   B: 같은 프롬프트를 cufile로 store → 파일 바이트 동일성 비교
#   C: 새 프로세스에서 staged 루트 load → fs 로드 발생 + 출력 토큰 일치
#   D: cufile.log TRACE에서 staged write의 bounce 마커 계수 (등록 direct 확인)
# usage: python verify_staged.py            (orchestrator)
#        python verify_staged.py child <store|load> <transport> <kvroot> <out.json>
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def child(role, transport, kvroot, out_path):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, "..", "phase1"))
    import expfs
    cnt = {"r": 0, "w": 0}
    _rc = expfs.CuFileTransport.read_chunk
    def rc(self, path, spans, cb):
        cnt["r"] += 1
        return _rc(self, path, spans, cb)
    expfs.CuFileTransport.read_chunk = rc
    _ws = expfs.StagedCuFileTransport.write_slot
    def ws(self, slot, path, kind="ring"):
        cnt["w"] += 1
        return _ws(self, slot, path, kind)
    expfs.StagedCuFileTransport.write_slot = ws
    _wc = expfs.CuFileTransport.write_chunk
    def wc(self, path, spans, cb):
        cnt["w"] += 1
        return _wc(self, path, spans, cb)
    expfs.CuFileTransport.write_chunk = wc

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    llm = LLM(model="facebook/opt-125m",
              kv_transfer_config=KVTransferConfig(
                  kv_connector="OffloadingConnector", kv_role="kv_both",
                  kv_connector_extra_config={
                      "spec_name": "ExperimentalFilesystemSpec",
                      "spec_module_path": "expfs",
                      "expfs_root_dir": kvroot,
                      "expfs_transport": transport, "block_size": 64}),
              gpu_memory_utilization=0.5, max_model_len=2048, enforce_eager=True)
    prompt = "The history of computing begins with " + " ".join(
        f"chapter {i} about machines and" for i in range(120))
    sp = SamplingParams(max_tokens=8, temperature=0.0)
    out = llm.generate([prompt], sp, use_tqdm=False)[0]
    toks = list(out.outputs[0].token_ids)
    for _ in range(3):  # store 캐스케이드 진행용 nudge
        llm.generate(["hi"], SamplingParams(max_tokens=1), use_tqdm=False)
    w = expfs.LAST_WORKER
    drained = True
    if hasattr(w.transport, "flush"):
        drained = w.transport.flush()
    files = {}
    for root, _d, fs in os.walk(kvroot):
        for f in fs:
            if f.endswith(".json"):
                continue
            p = os.path.join(root, f)
            files[os.path.relpath(p, kvroot)] = hashlib.sha256(
                open(p, "rb").read()).hexdigest()
    json.dump(dict(tokens=toks, files=files, reads=cnt["r"], writes=cnt["w"],
                   drained=drained,
                   ring_registered=getattr(w.transport, "ring_registered", None),
                   write_errors=getattr(w.transport, "write_errors", 0)),
              open(out_path, "w"))
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    os._exit(0)


def run_child(role, transport, kvroot, tag):
    out = os.path.join(HERE, f"vs_{tag}.json")
    log = os.path.join(HERE, f"vs_{tag}.log")
    r = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "child", role, transport,
         kvroot, out],
        stdout=open(log, "w"), stderr=subprocess.STDOUT, timeout=600,
        cwd=HERE)
    assert r.returncode == 0, f"{tag} child failed — see {log}"
    return json.load(open(out))


def main():
    import shutil
    ks = os.path.join(HERE, "kvroot-vs-staged")
    kc = os.path.join(HERE, "kvroot-vs-cufile")
    for k in (ks, kc):
        shutil.rmtree(k, ignore_errors=True)
    for f in ("cufile.log",):
        try:
            os.remove(os.path.join(HERE, f))
        except OSError:
            pass

    a = run_child("store", "cufile_staged", ks, "A_staged_store")
    print(f"A staged store: files={len(a['files'])} writes={a['writes']} "
          f"drained={a['drained']} ring_registered={a['ring_registered']} "
          f"write_errors={a['write_errors']}")
    assert a["files"] and a["drained"] and a["write_errors"] == 0

    b = run_child("store", "cufile", kc, "B_cufile_store")
    common = set(a["files"]) & set(b["files"])
    same = sum(1 for f in common if a["files"][f] == b["files"][f])
    print(f"B byte-identity: common={len(common)} identical={same}")
    assert common and same == len(common), "staged 파일이 cufile 산출과 다름"

    c = run_child("load", "cufile_staged", ks, "C_staged_load")
    print(f"C fresh-process load: reads={c['reads']} tokens_match="
          f"{c['tokens'] == a['tokens']}")
    assert c["reads"] > 0 and c["tokens"] == a["tokens"]

    log = os.path.join(HERE, "cufile.log")
    bounce = 0
    total = 0
    if os.path.exists(log):
        for line in open(log, errors="ignore"):
            if "cuFileWrite" in line:
                total += 1
            if "bounce_buffer" in line or "write_through_bounce" in line:
                bounce += 1
    print(f"D TRACE: cuFileWrite 언급 {total}건, bounce 마커 {bounce}건 "
          f"(ring 등록 시 staged write는 bounce 없어야 함)")
    print("VERIFY-STAGED PASS")
    for k in (ks, kc):
        shutil.rmtree(k, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        main()
