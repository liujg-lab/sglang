"""Round 9 follow-up: is ClearTree's under-growth at high acceptance the simulator's learner?

The simulator's learned model starts the global intercept a at 0 and moves a and g by SGD (step 0.01 per label,
feasibility.ClearTree.learn). At high acceptance almost every label is a hit with p near 1, so the gradient is
tiny and a stays low; per-request effects are shrunk (prior sd 0.6, clipped at +-3). The design instead fits the
GLMM by maximum likelihood in a background process. CT5MLE adds that: every 10 rounds, Newton steps on (a, g)
over the last 100 labels of every request, with the current per-request effects as offsets.
Runs: high and mid workloads, b in {1, 4, 8}, K fixed at 32, cost margin 1.0, W=4, seed 7.
usage: diag9.py <proc 0|1>"""
import json, sys, time
import numpy as np
import feasibility as F
import ext5 as V
import run8 as R8
import run9 as R9


class CT5MLE(V.CT5):
    def learn(self, world, slots, trees, com):
        super().learn(world, slots, trees, com)
        self._nl = getattr(self, "_nl", 0) + 1
        if self._nl % 10:
            return
        X, Y, Dd = [], [], []
        for r, (xs, ys) in self.lab.items():
            xs_, ys_ = xs[-100:], ys[-100:]
            X += xs_; Y += ys_; Dd += [self.dhat.get(r, 0.0)] * len(xs_)
        if len(X) < 50:
            return
        X, Y, Dd = np.asarray(X), np.asarray(Y), np.asarray(Dd)
        a, g = self.ahat_, self.ghat
        for _ in range(5):
            p = F.sig(a + g * X + Dd)
            w = p * (1 - p) + 1e-6
            ga, gg = (Y - p).sum(), ((Y - p) * X).sum()
            haa, hag, hgg = w.sum() + 1e-3, (w * X).sum(), (w * X * X).sum() + 1e-3
            det = haa * hgg - hag ** 2
            a += (hgg * ga - hag * gg) / det
            g += (haa * gg - hag * ga) / det
        self.ahat_, self.ghat = float(np.clip(a, -5, 10)), float(np.clip(g, 0.0, 5.0))


def ct_mle(w, ks=(32,), fd=1.0, W=4):
    for m in (R9.F, R9.E, R9.X, R9.V):
        m.KS = ks
    cm = R9.Y.CostModel3(w, "learned", p_force=0.03 if len(ks) > 1 else 0.0, fd=fd)
    cm.fd_b1 = 1.0
    return CT5MLE(W, "learned", cost=cm, slow=None, n_snap=4, kreuse=10, learn_beta=True, dyn_delta=True, audit_stop=True)


def main():
    proc = int(sys.argv[1])
    jobs = sorted([(k, b) for k in ("high", "mid") for b in R8.BS], key=lambda j: -j[1])
    for n, (kind, b) in enumerate(jobs):
        if n % 2 != proc:
            continue
        t0 = time.time()
        w = R8.make_world(R9.FIT, kind, b, 7)
        p = ct_mle(w)
        res = R9.run(w, p, 4)
        res["ahat"], res["ghat"] = round(p.ahat_, 3), round(p.ghat, 3)
        a = V.audit_summary(p.srec)
        res["audit"] = {k: round(a[k], 4) for k in ("stop_dG_mu", "stop_dA_mu", "grow_dA_mu", "all_dA_mu")}
        print(json.dumps(dict(kind=kind, b=b, res=res, sec=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
