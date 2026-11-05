#!/usr/bin/env python
"""
[실험 전용] attention 뒤의 분류 머리를 키우거나 잔차 연결을 넣으면 AUC 가 오르는가?

이 파일은 RTL, q88_golden.py, train_attention.py 와 무관한 독립 실험이다. 소프트웨어(FP32)만 본다.
결과가 좋아도 하드웨어에 넣으려면 fc_layer 등 RTL 변경과 Q8.8 골든 모델 확장이 따로 필요하다.

비교하는 구조 (모두 같은 데이터, 같은 학습 설정, 같은 시작값 후보)
  square      : attention -> O^2 . w + b                       (현재 후보, 기준)
  res_square  : attention -> (O + X)^2 . w + b                 (잔차 연결: 센서별 원래 값을 머리에 전달, 덧셈 16개 추가)
  mlp8        : attention -> Linear(16,8) -> ReLU -> Linear(8,1)
  res_mlp8    : attention -> (O + X) -> 위와 같은 MLP 머리
  mlp16       : attention -> Linear(16,16) -> ReLU -> Linear(16,1)
  raw_mlp16   : attention 없이 X(16) -> Linear(16,16) -> ReLU -> Linear(16,1)   (attention 이 병목인지 가려내는 대조군)
  raw_square  : attention 없이 X^2 . w + b                                      (res_square 의 이득이 attention 때문인지 가리는 대조군)
  raw_mlp8    : attention 없이 X(16) -> Linear(16,8) -> ReLU -> Linear(8,1)     (mlp8 / res_mlp8 과 파라미터 수를 맞춘 대조군)

읽는 법
  - mlp/res 계열이 square 보다 크게 오르면 '분류 머리'(또는 정보 전달)가 병목이었다는 뜻.
  - 모두 square 근처에 머물고 raw_mlp16 만 높으면 attention 앞단이 정보를 잃고 있다는 뜻.
  - 모두 낮으면 이 데이터에서 작은 모델의 한계로 보고 여기서 멈추는 것이 합리적.

사용법
  python exp_head.py --data dataset_t --eval-data dataset_t_hard --out exp_head_out
  python exp_head.py --data dataset_t --eval-data dataset_t_hard --out exp_head_out --only square,res_square

필요 패키지: numpy, scikit-learn, torch
"""
import argparse
import copy
import json
import os
import time

import numpy as np

ATYPES = ["offset", "ramp", "swap_cond"]

SPECS = {
    "square":     dict(front="attn", res=False, head="square", H=0,  label="attention + 제곱 머리 (기준)"),
    "res_square": dict(front="attn", res=True,  head="square", H=0,  label="attention + 잔차 + 제곱 머리"),
    "mlp8":       dict(front="attn", res=False, head="mlp",    H=8,  label="attention + MLP(은닉 8)"),
    "res_mlp8":   dict(front="attn", res=True,  head="mlp",    H=8,  label="attention + 잔차 + MLP(은닉 8)"),
    "mlp16":      dict(front="attn", res=False, head="mlp",    H=16, label="attention + MLP(은닉 16)"),
    "raw_mlp16":  dict(front="raw",  res=False, head="mlp",    H=16, label="attention 없이 MLP(은닉 16), 대조군"),
    "rtl":        dict(front="attn", res=False, head="linear", H=0,  label="기본 RTL 구조 (attention + 선형 FC, 바이어스 없음)"),
    "uniform_res_square": dict(front="attn", res=True, head="square", H=0, uniform=True,
                               label="균등 attention(1/4 고정) + 잔차 + 제곱 머리 (대조군)"),
    "raw_square": dict(front="raw",  res=False, head="square", H=0,  label="attention 없이 X^2 . w + b, 대조군"),
    "raw_mlp8":   dict(front="raw",  res=False, head="mlp",    H=8,  label="attention 없이 MLP(은닉 8), 대조군"),
}


def load_split(data_dir, split):
    z = np.load(os.path.join(data_dir, f"dataset_{split}.npz"))
    return {k: z[k] for k in z.files}


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def cost(spec):
    """(파라미터 수, 머리의 곱셈-누적 횟수) 추정. attention 앞단은 모두 48개 가중치."""
    p = 48 if spec["front"] == "attn" else 0
    if spec["head"] == "linear":
        return p + 16, 16
    if spec["head"] == "square":
        return p + 17, 16 + 16          # 제곱 16 + 가중치 곱 16
    h = spec["H"]
    return p + 18 * h + 1, 16 * h + h


# ---------------------------------------------------------------------------
# PyTorch 모델/학습
# ---------------------------------------------------------------------------
def make_net(spec):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if spec["front"] == "attn":
                self.wq = nn.Parameter(torch.randn(4, 4) * 0.3)
                self.wk = nn.Parameter(torch.randn(4, 4) * 0.3)
                self.wv = nn.Parameter(torch.randn(4, 4) * 0.3)
            if spec["head"] == "linear":
                self.wfc = nn.Parameter(torch.randn(16) * 0.1)
            elif spec["head"] == "square":
                self.wfc = nn.Parameter(torch.randn(16) * 0.1)
                self.b = nn.Parameter(torch.zeros(()))
            else:
                self.l1 = nn.Linear(16, spec["H"])
                self.l2 = nn.Linear(spec["H"], 1)

        def forward(self, x):                          # x: (B,4,4) 센서 x 시간
            mids = []
            if spec["front"] == "attn":
                q, k, v = x @ self.wq, x @ self.wk, x @ self.wv
                s = (q @ k.transpose(1, 2)) / 2.0
                a = torch.softmax(s, dim=-1)
                if spec.get("uniform"):
                    a = torch.full_like(a, 0.25)           # 대조군: attention 가중치를 균등(1/4)으로 고정 -> 센서 평균 문맥만 남김
                o = a @ v
                if spec["res"]:
                    o = o + x                          # 잔차 연결: 센서 i 의 원래 시간값을 더해 줌
                mids += [q, k, v, s, o]
                f = o.reshape(x.shape[0], 16)
            else:
                f = x.reshape(x.shape[0], 16)
            if spec["head"] == "linear":
                z = f @ self.wfc                           # 바이어스 없음 (현재 RTL)
            elif spec["head"] == "square":
                sq = f ** 2
                z = sq @ self.wfc + self.b
                mids.append(sq)
            else:
                h = torch.relu(self.l1(f))
                z = self.l2(h).squeeze(-1)
                mids.append(h)
            return z, tuple(mids)

    return Net()


def predict(model, X, batch=4096):
    """(점수 z, 중간값 최대 |값|). 학습 중이 아닌 평가용."""
    import torch
    model.eval()
    zs, mx = [], 0.0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32)
            z, mids = model(xb)
            zs.append(z.cpu().numpy())
            mx = max(mx, float(z.abs().max()), *[float(t.abs().max()) for t in mids])
    return np.concatenate(zs).astype(np.float64), mx


def train_one(spec, tr, va, args, seed):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_net(spec)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    Xtr = torch.tensor(tr["X"], dtype=torch.float32)
    ytr = torch.tensor(tr["y"], dtype=torch.float32)
    n = len(Xtr)
    best = (-1.0, None, 0)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            z, mids = model(Xtr[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(z - 0.5, ytr[idx])
            if args.hw_penalty > 0:
                pen = sum(torch.relu(t.abs() - args.hw_limit).pow(2).mean() for t in (*mids, z))
                loss = loss + args.hw_penalty * pen
            opt.zero_grad()
            loss.backward()
            opt.step()
        zv, _ = predict(model, va["X"])
        v = auc(va["y"], zv)
        if v > best[0]:
            best = (v, copy.deepcopy(model.state_dict()), ep)
        elif ep - best[2] >= args.patience:
            break
    model.load_state_dict(best[1])
    return model, best[0], best[2]


# ---------------------------------------------------------------------------
def evaluate(z, d):
    y = d["y"]
    neg = y == 0
    out = dict(auc=auc(y, z))
    for t, nm in enumerate(ATYPES):
        m = neg | (d["atype"] == t)
        out[nm] = auc(y[m], z[m])
    return out


def run(specs, tr, va, te, args, trainer=train_one, predictor=predict):
    rows = []
    for name in specs:
        spec = SPECS[name]
        t0 = time.time()
        best = (-1.0, None, 0, 0)
        for r in range(max(1, args.restarts)):
            model, v_auc, ep = trainer(spec, tr, va, args, args.seed + r)
            print(f"  [{name}] 시작값 {r + 1}/{args.restarts}: val AUC {v_auc:.3f} (epoch {ep})", flush=True)
            if v_auc > best[0]:
                best = (v_auc, model, ep, r)
        v_auc, model, ep, r = best
        zt, mx = predictor(model, te["X"])
        res = evaluate(zt, te)
        params, macs = cost(spec)
        rows.append(dict(name=name, label=spec["label"], params=params, macs=macs, val_auc=v_auc,
                         test=res, maxabs=mx, sec=time.time() - t0, epoch=ep))
        print(f"[{name}] Test AUC {res['auc']:.3f} (offset {res['offset']:.3f}, ramp {res['ramp']:.3f}, "
              f"swap_cond {res['swap_cond']:.3f})  중간값 최대 {mx:.1f}  {time.time() - t0:.0f}초", flush=True)
    return rows


def print_table(rows):
    print("\n=== 구조 비교 (Test = eval-data 의 Test, 모델 선택은 Val AUC 로만) ===")
    print(f"{'구조':<36}{'파라미터':>8}{'머리 MAC':>9}{'Val AUC':>9}{'Test AUC':>9}{'offset':>8}{'ramp':>7}{'swap':>7}{'중간값최대':>10}{'초':>5}")
    for r in rows:
        t = r["test"]
        print(f"{r['label']:<36}{r['params']:>8}{r['macs']:>9}{r['val_auc']:>9.3f}{t['auc']:>9.3f}"
              f"{t['offset']:>8.3f}{t['ramp']:>7.3f}{t['swap_cond']:>7.3f}{r['maxabs']:>10.1f}{r['sec']:>5.0f}")
    base = next((r for r in rows if r["name"] == "square"), None)
    if base:
        print(f"\n기준(square) Test AUC {base['test']['auc']:.3f} 대비:")
        for r in rows:
            if r["name"] != "square":
                print(f"  {r['label']}: {r['test']['auc'] - base['test']['auc']:+.3f}")
    print("\n판단 기준: Test AUC 0.85 이상이면 하드웨어 변경을 검토할 가치가 있음, 0.80~0.85 는 애매, 0.80 미만이면 여기서 멈춤.")
    print("중간값 최대가 100 을 넘는 구조는 Q8.8(약 ±128)에서 wrap 위험이 있다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_t", help="학습/검증 데이터셋 폴더")
    ap.add_argument("--eval-data", default=None, help="평가(Test) 데이터셋 폴더. 기본은 --data")
    ap.add_argument("--out", default="exp_head_out")
    ap.add_argument("--only", default=None, help="쉼표로 구분한 구조 이름 (기본: 전체). 가능: " + ",".join(SPECS))
    ap.add_argument("--restarts", type=int, default=2, help="구조마다 서로 다른 시작값으로 N번 학습해 val AUC 최선 선택")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--hw-penalty", type=float, default=0.1)
    ap.add_argument("--hw-limit", type=float, default=64.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    names = [s.strip() for s in args.only.split(",")] if args.only else ["square", "res_square", "mlp8", "res_mlp8", "mlp16", "raw_mlp16"]
    for nm in names:
        if nm not in SPECS:
            raise SystemExit(f"알 수 없는 구조: {nm}. 가능: {', '.join(SPECS)}")
    tr, va = load_split(args.data, "train"), load_split(args.data, "val")
    te = load_split(args.eval_data or args.data, "test")
    print(f"학습 {len(tr['y']):,} / 검증 {len(va['y']):,} / 평가 {len(te['y']):,}  구조 {len(names)}개 x 시작값 {args.restarts}회")
    rows = run(names, tr, va, te, args)
    print_table(rows)
    with open(os.path.join(args.out, "exp_head_results.json"), "w", encoding="utf-8") as f:
        json.dump(dict(args=vars(args), rows=rows), f, ensure_ascii=False, indent=2)
    print(f"\n저장: {args.out}/exp_head_results.json (이 실험은 가중치/.mem 을 내보내지 않는다)")


if __name__ == "__main__":
    main()