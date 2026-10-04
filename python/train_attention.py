#!/usr/bin/env python
"""
RTL 구조를 그대로 따르는 Self-Attention 모델 학습 + FP32 대 Q8.8 비교

모델 (rtl/ 의 데이터 흐름과 같다. 전부 바이어스 없음)
  x (B,4,4) = 센서 x 시간
  Q = x Wq, K = x Wk, V = x Wv                 Wq, Wk, Wv : 4x4  -> weights.mem 48워드
  S = Q K^T / 2                                 (RTL: >>> 1)
  A = softmax(S, 행 방향)                        (RTL: softmax_lut)
  O = A V
  z = flatten(O) . w_fc                         w_fc : 16개 -> fc_weights.mem
  label = z > 0.5                               (RTL argmax.sv 의 임시 threshold 0x0080)
  cause = argmax_j sum_i A[i][j]                (RTL cause_identifier 의 열 합산)

학습: BCE(z - 0.5, y). RTL threshold 가 0.5 라서 같은 기준으로 학습한다.
원인 센서는 기본적으로 지도하지 않는다(--cause-loss 0). Attention 열 합이 '이상 판단에 기여한 센서'를
스스로 가리키는지를 보기 위해서다. --cause-loss > 0 이면 합성 정답(cause)으로 보조 지도한다.

사용법
  python train_attention.py --data dataset_t --eval-data dataset_t_hard --out model_out --mem-dir <저장소>/mem
  python train_attention.py --eval-only model_out/model_fp32.npz --data dataset_t --eval-data dataset_t_hard --out model_out
  python train_attention.py --data dataset_t --eval-data dataset_t_hard --out sweep_out --mem-dir <저장소>/mem --sweep
      (설정 6가지를 한 번에 비교)
  python train_attention.py --data dataset_t --eval-data dataset_t_hard --out arch_out --mem-dir <저장소>/mem --arch-sweep
      (구조 3가지 비교: rtl / square / dist)
  python train_attention.py --data dataset_t --eval-data dataset_t_hard --out cause_out --mem-dir <저장소>/mem --cause-sweep
      (square 구조에서 원인 센서 마진 손실 + 모델 선택 조건 6가지 비교)
  python train_attention.py --from-mem <저장소>/mem --data dataset_t --eval-data dataset_t_hard --out mem_check --mem-dir <저장소>/mem
      (지금 저장소에 들어 있는 가중치로 RTL 정수 연산의 오차만 확인. torch 불필요)

필요 패키지: numpy, scikit-learn, torch (학습에만 필요. --eval-only 는 torch 없이 동작)
"""
import argparse
import json
import os
import time

import numpy as np

from q88_golden import compare_fp32_q88, export_mem, forward_fp32, fwd_fp32_w, loo_compare_q88, print_rows

ATYPES = ["offset", "ramp", "swap_cond"]


def load_split(data_dir, split):
    z = np.load(os.path.join(data_dir, f"dataset_{split}.npz"))
    return {k: z[k] for k in z.files}


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


EFF_BINS = [0.05, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 99]
HW_RANGE_OK = 100.0


def evaluate_fp32(w, d, thr=0.5, title="", quiet=False):
    """FP32 모델을 d(X, y, cause, atype)에 적용해 지표를 출력하고 dict 로 반환."""
    X = d["X"].astype(np.float64)
    y = d["y"]
    z, a, rng = fwd_fp32_w(w, X)
    lab = (z > thr).astype(np.int64)
    anom = y == 1
    cause_pred = a.sum(axis=1).argmax(axis=1)
    res = dict(auc=auc(y, z), acc=float((lab == y).mean()), tpr=float((lab[anom] == 1).mean()),
               fpr=float((lab[~anom] == 1).mean()), cause_acc=float((cause_pred[anom] == d["cause"][anom]).mean()))
    neg = ~anom
    if "eff" in d:
        big = neg | (anom & (d["eff"] >= 1.0))
        res["auc_big"] = auc(y[big], z[big])
    cs = np.sort(a.sum(axis=1), axis=1)
    gap = cs[:, -1] - cs[:, -2]
    res["gap_med"] = float(np.median(gap))
    res["gap_small"] = float((gap < 0.02).mean())
    if not quiet:
        print(f"[{title}] FP32  AUC={res['auc']:.3f}  Acc={res['acc']:.3f}  TPR={res['tpr']:.3f}  "
              f"FPR={res['fpr']:.3f}  원인센서 정확도(이상 윈도우, Attention 열합)={res['cause_acc']:.3f}")
        if "atype" in d:
            parts = []
            for t, nm in enumerate(ATYPES):
                m = neg | (d["atype"] == t)
                parts.append(f"{nm} {auc(y[m], z[m]):.3f}")
            print("      이상 종류별 AUC: " + ", ".join(parts))
        print(f"      Attention 열합 1위-2위 차이: 중앙값 {res['gap_med']:.3f} (열합 평균 1.0), "
              f"차이가 0.02 미만인 윈도우 {res['gap_small']:.1%}  "
              f"(Q8.8 양자화 잡음이 순위를 바꿀 수 있는 크기는 약 0.01~0.02)")
        if "eff" in d:
            parts = []
            for lo, hi in zip(EFF_BINS[:-1], EFF_BINS[1:]):
                m = neg | (anom & (d["eff"] >= lo) & (d["eff"] < hi))
                n_ab = int((anom & (d["eff"] >= lo) & (d["eff"] < hi)).sum())
                parts.append(f"{lo}~{hi if hi < 99 else 'inf'}: {auc(y[m], z[m]):.3f}(n={n_ab})" if n_ab >= 20
                             else f"{lo}~{hi if hi < 99 else 'inf'}: 해당 없음")
            print("      변화 크기(eff)별 AUC: " + ", ".join(parts))
    res["z"] = z
    return res


def load_weights_npz(path):
    z = np.load(path)
    w = {k: z[k].astype(np.float64) for k in ("wq", "wk", "wv", "wfc")}
    for k in ("b", "c"):
        if k in z.files:
            w[k] = z[k].astype(np.float64)
    w["arch"] = np.array(str(z["arch"]) if "arch" in z.files else "rtl")
    return w


def weights_from_mem(mem_dir):
    """weights.mem(48워드: Wq, Wk, Wv 각 4x4, qkv_transform.sv 의 w[i][j] = rom[g*16 + i*4 + j]) 와
    fc_weights.mem(16워드)를 Q8.8 -> float 로 읽는다."""
    from q88_golden import load_mem, wrap16
    m = wrap16(load_mem(os.path.join(mem_dir, "weights.mem"))) / 256.0
    f = wrap16(load_mem(os.path.join(mem_dir, "fc_weights.mem"))) / 256.0
    if len(m) != 48 or len(f) != 16:
        raise SystemExit(f"weights.mem {len(m)}워드(48이어야 함), fc_weights.mem {len(f)}워드(16이어야 함)")
    return dict(wq=m[:16].reshape(4, 4), wk=m[16:32].reshape(4, 4), wv=m[32:].reshape(4, 4), wfc=f)


def weights_from_checkpoint(path):
    """예전 SensorSelfAttentionClassifier 체크포인트. nn.Linear 는 x @ W.T 라서 RTL 배치(w[입력][출력])로 전치한다."""
    import torch
    ck = torch.load(path, map_location="cpu")
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    g = lambda k: sd[k].detach().cpu().numpy().astype(np.float64)
    return dict(wq=g("q_proj.weight").T, wk=g("k_proj.weight").T, wv=g("v_proj.weight").T,
                wfc=g("fc.weight").reshape(16))


def ablation_report(w, d, title="Test"):
    """교수님 피드백 5번: Attention 열합이 고른 센서를 '정상값(Train 평균 = 표준화 0)'으로 바꿨을 때
    이상 점수가 얼마나 줄어드는지를 무작위 센서와 비교한다. (열합이 이상 판단에 기여한 센서를 가리키는지의 검증)"""
    X = d["X"].astype(np.float64)
    anom = d["y"] == 1
    Xa = X[anom]
    z0, a0, _ = fwd_fp32_w(w, Xa)
    chosen = a0.sum(axis=1).argmax(axis=1)
    drops = np.zeros((len(Xa), 4))
    for r in range(4):
        Xr = Xa.copy()
        Xr[:, r, :] = 0.0
        drops[:, r] = z0 - fwd_fp32_w(w, Xr)[0]
    idx = np.arange(len(Xa))
    d_chosen = drops[idx, chosen]
    d_true = drops[idx, d["cause"][anom]]
    d_other = (drops.sum(axis=1) - d_chosen) / 3.0
    rank = (drops > d_chosen[:, None]).sum(axis=1)               # 0 이면 감소량 1위
    print(f"[{title}] 센서 대체(ablation) 검증 - 이상 윈도우 {len(Xa):,}개, 센서 행을 Train 평균(0)으로 대체했을 때 점수 감소량")
    loo_acc = float((drops.argmax(axis=1) == d["cause"][anom]).mean())
    print(f"      열합 1위 센서 {d_chosen.mean():.3f} | 나머지 3개 센서 평균 {d_other.mean():.3f} | 정답(합성) 센서 {d_true.mean():.3f}")
    print(f"      감소량이 가장 큰 센서(leave-one-out)로 원인 센서를 고르면 정확도 {loo_acc:.3f} "
          f"(Attention 열합 방식 {float((chosen == d['cause'][anom]).mean()):.3f}, 무작위 0.25)")
    print(f"      열합 1위 센서가 감소량 1위인 비율 {float((rank == 0).mean()):.3f} (무작위 0.25), "
          f"감소량이 나머지 평균보다 큰 비율 {float((d_chosen > d_other).mean()):.3f} (무작위 약 0.5)")
    print("      해석 주의: 평균(0)으로 바꾸는 것은 '정상값'의 근사일 뿐이고, 열합이 거의 균등하면 선택이 우연에 가까워 위 값들이 무작위와 비슷하게 나온다.")


def youden_threshold(z, y):
    cand = np.unique(np.quantile(z, np.linspace(0, 1, 501)))
    j = [float((z[y == 1] > t).mean() - (z[y == 0] > t).mean()) for t in cand]
    return float(cand[int(np.argmax(j))])


# ---------------------------------------------------------------------------
# PyTorch 학습
# ---------------------------------------------------------------------------
def train_torch(tr, va, args):
    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    class RtlAttention(nn.Module):
        def __init__(self, arch):
            super().__init__()
            self.arch = arch
            self.wq = nn.Parameter(torch.randn(4, 4) * 0.3)
            self.wk = nn.Parameter(torch.randn(4, 4) * 0.3)
            self.wv = nn.Parameter(torch.randn(4, 4) * 0.3)
            self.wfc = nn.Parameter(torch.randn(16) * 0.1)
            if arch in ("square", "res_square", "dist"):
                self.b = nn.Parameter(torch.zeros(()))
            if arch == "dist":
                self.c = nn.Parameter(torch.zeros(16))

        def forward(self, x):                              # x: (B, 4, 4) = 센서 x 시간
            q, k, v = x @ self.wq, x @ self.wk, x @ self.wv
            s = (q @ k.transpose(1, 2)) / 2.0
            a = torch.softmax(s, dim=-1)                   # 행 방향
            o = a @ v
            if self.arch == "res_square":
                o = o + x                                  # 잔차 연결: 센서별 원래 시간값을 더함
            f = o.reshape(x.shape[0], 16)
            extra = ()
            if self.arch == "rtl":
                z = f @ self.wfc                           # 바이어스 없음 (현재 RTL)
            elif self.arch in ("square", "res_square"):
                sq = f ** 2
                z = sq @ self.wfc + self.b
                extra = (sq,)
            else:                                          # dist
                sq = (f - self.c) ** 2
                z = sq @ self.wfc + self.b
                extra = (sq,)
            return z, a, (q, k, v, s, o) + extra

    model = RtlAttention(args.arch)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    Xtr = torch.tensor(tr["X"], dtype=torch.float32)
    ytr = torch.tensor(tr["y"], dtype=torch.float32)
    ctr = torch.tensor(tr["cause"], dtype=torch.long)

    def weights_np():
        w = {k: getattr(model, k).detach().cpu().numpy().astype(np.float64)
             for k in ("wq", "wk", "wv", "wfc")}
        w["arch"] = np.array(args.arch)
        if args.arch in ("square", "res_square", "dist"):
            w["b"] = np.array(float(model.b.detach().cpu()))
        if args.arch == "dist":
            w["c"] = model.c.detach().cpu().numpy().astype(np.float64)
        return w

    best = (-1.0, None, 0)            # 조건(열합 차이)을 만족한 epoch 중 val AUC 최선
    fallback = (-1.0, None, 0, -1.0)  # 조건을 만족한 epoch 이 없을 때: 열합 차이가 가장 큰 epoch
    last_improve = 0
    n = len(Xtr)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            x, y, c = Xtr[idx], ytr[idx], ctr[idx]
            z, a, mids = model(x)
            loss = nn.functional.binary_cross_entropy_with_logits(z - 0.5, y)
            if args.hw_penalty > 0:
                # Q8.8 은 약 ±128 까지만 표현한다. 중간값이 hw_limit 을 넘으면 wrap 되어 하드웨어 결과가 깨지므로 억제한다.
                pen = sum(torch.relu(t.abs() - args.hw_limit).pow(2).mean() for t in (*mids, z))
                loss = loss + args.hw_penalty * pen
            if args.cause_loss > 0:
                m = y > 0.5
                if m.any():
                    cs = a.sum(dim=1)                            # (B,4) 열 합 (각 행의 합이 1 이므로 전체 합 = 4)
                    cm, cc = cs[m], c[m].unsqueeze(1)
                    if args.cause_margin > 0:                    # 힌지: (정답 열합 - 나머지 최대 열합) >= margin
                        true = cm.gather(1, cc).squeeze(1)
                        other = cm.scatter(1, cc, float("-inf")).max(dim=1).values
                        loss = loss + args.cause_loss * torch.relu(args.cause_margin - (true - other)).mean()
                    elif args.cause_temp > 0:                    # softmax(열합 x T)
                        loss = loss + args.cause_loss * (-torch.log_softmax(cm * args.cause_temp, dim=1).gather(1, cc)).mean()
                    else:                                        # 기존: 열합/4 를 확률로 취급
                        loss = loss + args.cause_loss * (-torch.log(cm / 4.0 + 1e-8).gather(1, cc)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
            cnt += len(idx)
        w = weights_np()
        zv, av, _ = fwd_fp32_w(w, va["X"].astype(np.float64))
        v_auc = auc(va["y"], zv)
        anom = va["y"] == 1
        v_cause = float((av.sum(axis=1).argmax(axis=1)[anom] == va["cause"][anom]).mean())
        csv = np.sort(av[anom].sum(axis=1), axis=1)
        v_gap = float(np.median(csv[:, -1] - csv[:, -2]))          # 이상 윈도우의 열합 1위-2위 차이 중앙값
        if getattr(args, "verbose", True):
            print(f"epoch {ep:3d}  train loss {tot / cnt:.4f}  val AUC {v_auc:.3f}  val 원인센서 {v_cause:.3f}  열합차이 {v_gap:.3f}")
        eligible = args.min_gap <= 0 or v_gap >= args.min_gap
        if eligible and v_auc > best[0]:
            best, last_improve = (v_auc, w, ep), ep
        if v_gap > fallback[3]:
            fallback = (v_auc, w, ep, v_gap)
            if best[1] is None:
                last_improve = ep
        if ep - last_improve >= args.patience:
            if getattr(args, "verbose", True):
                print(f"{args.patience} epoch 동안 개선이 없어 중단")
            break
    if best[1] is None:                                            # 열합 차이 조건을 한 번도 못 채움
        print(f"[경고] 열합 차이가 {args.min_gap} 이상인 epoch 이 없었습니다. 열합 차이가 가장 큰 epoch {fallback[2]} "
              f"(차이 {fallback[3]:.3f}, val AUC {fallback[0]:.3f}) 를 사용합니다.")
        best = (fallback[0], fallback[1], fallback[2])
    if getattr(args, "verbose", True):
        print(f"선택된 모델: epoch {best[2]}, val AUC {best[0]:.3f}")
    return best[1], best[0]


def train_best_of(tr, va, args):
    """--restarts 번 서로 다른 시작값으로 학습해 val AUC 가 가장 높은 가중치를 고른다."""
    import copy
    n = max(1, int(args.restarts))
    best = (-1.0, None)
    for r in range(n):
        a = copy.copy(args)
        a.seed = args.seed + r
        w, v = train_torch(tr, va, a)
        if n > 1 and getattr(args, "verbose", True):
            print(f"  [시작값 {r + 1}/{n}] seed={a.seed} val AUC {v:.3f}")
        if v > best[0]:
            best = (v, w)
    return best[1], best[0]


def run_sweep(tr, va, te, args, configs=None):
    """설정 몇 가지를 한 번에 돌려 표로 비교한다. 모델 선택은 Val AUC 로만 하고, Test 는 보고만 한다.
    단, 중간값 최대가 HW_RANGE_OK(=100)를 넘는 설정은 Q8.8(약 ±128)에서 wrap 위험이라 선택에서 제외한다."""
    import copy
    if configs is None:
        configs = [
            ("기본", {}),
            ("벌점 끔", {"hw_penalty": 0.0}),
            ("원인 지도 0.5", {"cause_loss": 0.5}),
            ("lr 1e-2", {"lr": 1e-2, "epochs": 150, "patience": 30}),
            ("시작값 5회", {"restarts": 5}),
            ("시작값 5회 + 원인 지도 0.5", {"restarts": 5, "cause_loss": 0.5}),
        ]
    table = []
    for i, (name, ov) in enumerate(configs):
        a = copy.copy(args)
        for k, v in ov.items():
            setattr(a, k, v)
        a.verbose = False
        t0 = time.time()
        print(f"[sweep {i + 1}/{len(configs)}] {name} ...", flush=True)
        w, v_auc = train_best_of(tr, va, a)
        r_va = evaluate_fp32(w, va, quiet=True)
        r_te = evaluate_fp32(w, te, quiet=True)
        rows, rng = compare_fp32_q88(w, te["X"].astype(np.float64), te["X_q88"].astype(np.int64),
                                     te["y"], te["cause"], mem_dir=args.mem_dir)
        asis, fixed = rows[1], rows[-1]
        table.append(dict(name=name, val_auc=r_va["auc"], test_auc=r_te["auc"], auc_big=r_te.get("auc_big", float("nan")),
                          tpr=r_te["tpr"], fpr=r_te["fpr"], cause=r_te["cause_acc"],
                          gap=r_te["gap_med"], cagree=rows[-1]["cause_agree"], cause8=rows[-1]["cause_acc"],
                          agree_asis=asis["agree"], agree_fixed=fixed["agree"], z_mae=fixed["z_mae"],
                          sec=time.time() - t0, zmax=max(float(v) for v in rng.values())))
        d = os.path.join(args.out, f"sweep_{i + 1}")
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, "model_fp32.npz"), **w)
        export_mem(d, w)
    print("\n=== sweep 결과 (모델 선택 기준은 Val AUC, Test 는 보고용) ===")
    print(f"{'설정':<28}{'ValAUC':>8}{'TestAUC':>9}{'AUC(eff≥1σ)':>12}{'TPR@0.5':>9}{'FPR@0.5':>9}{'원인정확도':>9}"
          f"{'열합차이':>8}{'원인일치(수정4)':>14}{'원인정확도(Q8.8)':>15}{'Q8.8일치(현재RTL)':>17}{'Q8.8일치(수정4)':>15}{'z오차':>8}{'중간값최대':>9}{'초':>5}")
    for r in table:
        print(f"{r['name']:<28}{r['val_auc']:>8.3f}{r['test_auc']:>9.3f}{r['auc_big']:>12.3f}{r['tpr']:>9.3f}{r['fpr']:>9.3f}"
              f"{r['cause']:>9.3f}{r['gap']:>8.3f}{r['cagree']:>14.3f}{r['cause8']:>15.3f}{r['agree_asis']:>17.3f}{r['agree_fixed']:>15.3f}"
              f"{r['z_mae']:>8.3f}{r['zmax']:>9.1f}{r['sec']:>5.0f}")
    ok = [i for i in range(len(table)) if table[i]["zmax"] <= HW_RANGE_OK]
    skipped = [i + 1 for i in range(len(table)) if i not in ok]
    if skipped:
        print(f"\n중간값 최대가 {HW_RANGE_OK:.0f} 를 넘어 선택에서 제외: {skipped} (Q8.8 wrap 위험)")
    pool = ok or list(range(len(table)))
    best = max(pool, key=lambda i: table[i]["val_auc"])
    print(f"Val AUC 기준 최선(하드웨어 범위 안): [{best + 1}] {table[best]['name']} -> {args.out}/sweep_{best + 1}/ 에 가중치와 .mem 저장")
    with open(os.path.join(args.out, "sweep_results.json"), "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_t", help="학습/검증 데이터셋 폴더 (make_dataset.py 결과)")
    ap.add_argument("--eval-data", default=None, help="평가(Test)에 쓸 데이터셋 폴더. 기본은 --data")
    ap.add_argument("--out", default="model_out")
    ap.add_argument("--mem-dir", default=None, help="저장소 mem 폴더(softmax_exp_rom.mem 등). 있으면 현재 ROM 으로도 비교")
    ap.add_argument("--eval-only", default=None, help="저장된 model_fp32.npz 로 평가만 (torch 불필요)")
    ap.add_argument("--from-mem", default=None,
                    help="저장소 mem 폴더의 weights.mem/fc_weights.mem 을 FP32 가중치로 읽어 평가만 (torch 불필요)")
    ap.add_argument("--from-checkpoint", default=None,
                    help="예전 학습 코드(train_attention_model.py)의 best_model.pt 에서 가중치를 읽어 평가만 (torch 필요)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--cause-loss", type=float, default=0.0, help="0 이면 원인 센서를 지도하지 않음")
    ap.add_argument("--cause-temp", type=float, default=0.0,
                    help="0 이면 열합/4 를 확률로 사용(기존). 양수 T 이면 softmax(열합 x T) 로 원인 센서 손실을 계산 "
                         "(열합 1위-2위 차이를 벌려 Q8.8 양자화 잡음에 강하게 만들려는 용도)")
    ap.add_argument("--cause-margin", type=float, default=0.0,
                    help="0 이면 사용 안 함. 양수 M 이면 원인 센서 손실을 힌지로 바꿈: (정답 열합 - 나머지 최대 열합) >= M. "
                         "열합 차이가 Q8.8 해상도(1/256)보다 충분히 커지도록 직접 압박 (예: 0.05~0.2). --cause-loss > 0 필요")
    ap.add_argument("--min-gap", type=float, default=0.0,
                    help="0 이면 사용 안 함. 양수이면 검증셋 이상 윈도우의 열합 1위-2위 차이 중앙값이 이 값 이상인 epoch 만 "
                         "모델 선택 후보로 인정 (예: 0.02). 하나도 없으면 차이가 가장 큰 epoch 로 대체하고 경고")
    ap.add_argument("--cause-sweep", action="store_true",
                    help="square 구조에서 원인 센서 손실 가중치/온도 조합 6가지를 비교 (torch 필요)")
    ap.add_argument("--hw-penalty", type=float, default=0.1,
                    help="Q, K, V, score, attn_out, z 의 절댓값이 --hw-limit 을 넘으면 주는 벌점 가중치 (0 이면 끔)")
    ap.add_argument("--hw-limit", type=float, default=64.0, help="Q8.8 표현 한계(약 128)보다 여유를 둔 상한")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", choices=["rtl", "square", "res_square", "dist"], default="rtl",
                    help="rtl: 현재 RTL 구조. square: FC 앞에 제곱 한 단계 + 바이어스. res_square: (O + X) 잔차 연결 후 제곱 + 바이어스. dist: (O - c)^2 거리 + 바이어스 "
                         "(square/dist 는 fc_layer RTL 변경이 필요)")
    ap.add_argument("--arch-sweep", action="store_true",
                    help="구조 3가지(rtl/square/dist)를 같은 설정으로 학습해 비교표를 출력 (torch 필요)")
    ap.add_argument("--restarts", type=int, default=1, help="서로 다른 시작값으로 N번 학습해 val AUC 최선을 고름")
    ap.add_argument("--sweep", action="store_true", help="설정 6가지를 한 번에 돌려 비교표를 출력 (학습 6~15회, torch 필요)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    eval_dir = args.eval_data or args.data

    va = load_split(args.data, "val")
    te = load_split(eval_dir, "test")
    if args.from_mem:
        w = weights_from_mem(args.from_mem)
        print(f"mem 파일의 가중치 사용(Q8.8 을 float 로 환원): {args.from_mem}")
    elif args.from_checkpoint:
        w = weights_from_checkpoint(args.from_checkpoint)
        print(f"체크포인트의 가중치 사용: {args.from_checkpoint}")
        print("  주의: 예전 코드는 sigmoid(logit) 의 threshold 를 검증셋에서 골랐다. RTL 은 logit > 0.5 라서 판정 기준이 다르다.")
    elif args.eval_only:
        w = load_weights_npz(args.eval_only)
        print(f"저장된 가중치 사용: {args.eval_only}  (arch={w['arch']})")
    else:
        tr = load_split(args.data, "train")
        print(f"학습 {len(tr['y']):,} / 검증 {len(va['y']):,} / 평가 {len(te['y']):,}  "
              f"(cause-loss={args.cause_loss}, hw-penalty={args.hw_penalty}, hw-limit={args.hw_limit})")
        if args.cause_sweep:
            run_sweep(tr, va, te, args, configs=[
                ("square, 원인 지도 없음", {"arch": "square"}),
                ("마진 0.05, 가중치 1, 차이≥0.02", {"arch": "square", "cause_loss": 1.0, "cause_margin": 0.05, "min_gap": 0.02}),
                ("마진 0.1, 가중치 1, 차이≥0.02", {"arch": "square", "cause_loss": 1.0, "cause_margin": 0.1, "min_gap": 0.02}),
                ("마진 0.1, 가중치 3, 차이≥0.02", {"arch": "square", "cause_loss": 3.0, "cause_margin": 0.1, "min_gap": 0.02}),
                ("마진 0.2, 가중치 3, 차이≥0.05", {"arch": "square", "cause_loss": 3.0, "cause_margin": 0.2, "min_gap": 0.05}),
                ("마진 0.1, 가중치 3, 선택조건 없음", {"arch": "square", "cause_loss": 3.0, "cause_margin": 0.1}),
            ])
            return
        if args.arch_sweep:
            run_sweep(tr, va, te, args, configs=[
                ("rtl (현재 구조)", {"arch": "rtl"}),
                ("square (FC 앞 제곱 + 바이어스)", {"arch": "square"}),
                ("dist ((O-c)^2 + 바이어스)", {"arch": "dist"}),
                ("square, lr 1e-2", {"arch": "square", "lr": 1e-2, "epochs": 150, "patience": 30}),
                ("dist, lr 1e-2", {"arch": "dist", "lr": 1e-2, "epochs": 150, "patience": 30}),
            ])
            return
        if args.sweep:
            run_sweep(tr, va, te, args)
            return
        t0 = time.time()
        w, _ = train_best_of(tr, va, args)
        print(f"학습 시간 {time.time() - t0:.0f}초")
        np.savez(os.path.join(args.out, "model_fp32.npz"), **w)

    print("\n=== FP32 평가 (RTL threshold 0.5) ===")
    r_va = evaluate_fp32(w, va, title="Val ")
    r_te = evaluate_fp32(w, te, title="Test")
    ablation_report(w, te)
    thr_v = youden_threshold(r_va["z"], va["y"])
    lab = (r_te["z"] > thr_v)
    print(f"참고: Val 에서 TPR-FPR 이 최대인 threshold = {thr_v:.3f} -> Test TPR {float(lab[te['y'] == 1].mean()):.3f}, "
          f"FPR {float(lab[te['y'] == 0].mean()):.3f}  (RTL argmax.sv 의 THRESHOLD 를 바꿔야 적용됨)")

    print("\n=== FP32 대 Q8.8 (Test, RTL 수정 후보를 단계별로 켠 표) ===")
    rows, rng = compare_fp32_q88(w, te["X"].astype(np.float64), te["X_q88"].astype(np.int64),
                                 te["y"], te["cause"], mem_dir=args.mem_dir)
    print_rows(rows, rng)
    loo = loo_compare_q88(w, te["X"].astype(np.float64), te["X_q88"].astype(np.int64), te["y"], te["cause"],
                          mem_dir=args.mem_dir)
    print(f"\n=== leave-one-out 원인 센서 (센서 행을 0 으로 바꿨을 때 점수가 가장 많이 줄어드는 센서), 이상 윈도우 {loo['n']:,}개 ===")
    print(f"  정확도: FP32 {loo['acc32']:.3f}, Q8.8(RTL 수정 4가지 적용) {loo['acc88']:.3f}, 무작위 0.25 | "
          f"FP32와 Q8.8 선택 일치 {loo['agree']:.3f}")
    print(f"  1위-2위 감소량 차이(중앙값): FP32 {loo['margin32']:.3f}, Q8.8 {loo['margin88']:.3f}  (Q8.8 해상도 0.004 보다 충분히 커야 안정적)")

    export_mem(args.out, w)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(dict(args=vars(args), val=dict((k, v) for k, v in r_va.items() if k != "z"),
                       test=dict((k, v) for k, v in r_te.items() if k != "z"), q88=rows,
                       fp32_max_abs=rng, recommended_threshold=thr_v), f, ensure_ascii=False, indent=2)
    print(f"\n저장: {args.out}/model_fp32.npz, weights.mem, fc_weights.mem, results.json")


if __name__ == "__main__":
    main()