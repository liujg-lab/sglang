"""Round 38: the verify-cost regression at startup, plus a log of replayed width.

BatchAccount22 (defaults: the base class, bit for bit):
  vnn      if the solved coefficient of sum-l or of (K+1)*sum-l is below -1e-8, drop that
           column and solve again. The b slope and the three K intercepts stay. No parameter.
  vprior   ridge on the three slopes (b, sum-l, (K+1)*sum-l), prior 0. Intercepts are not
           penalised. The ridge on a slope is n0 times that feature's weighted mean square
           (diagonal of A, divided by the forgetting-weighted sample count). n0 <= 0 leaves
           the base solver in place unless vnn is also on.
  explog   per b bucket, accumulate the replayed throughput of the current width on explore
           rounds and the measured throughput of the current width on normal rounds.
           Recorded only.

Layer regressions are untouched. A cost-event clear zeros the verify regression's sample
count; the weighted count restarts on the next add.
"""
import numpy as np
import ext9 as Q
import ext18 as H
import ext28 as E


class _VerifyRLS(H._RLS):
    def __init__(self, n0=0.0, vnn=False):
        super().__init__(6)
        self.n0 = float(n0)
        self.vnn = bool(vnn)
        self.n_eff = 0.0

    def add(self, x, t):
        if self.n == 0:
            self.n_eff = 0.0
        super().add(x, t)
        self.n_eff = self.lam * self.n_eff + 1.0

    def _solve(self, idx):
        idx = list(idx)
        A = self.A[np.ix_(idx, idx)]
        y = self.y[idx]
        scale = np.maximum(np.diag(A), 1e-9)
        pen = 1e-6 * scale + 1e-12
        if self.n0 > 0.0 and self.n_eff > 0.0:
            for j, i in enumerate(idx):
                if i >= 3:
                    pen[j] += self.n0 * (self.A[i, i] / self.n_eff)
        return np.linalg.solve(A + np.diag(pen), y)

    @staticmethod
    def _pack(idx, sol):
        full = np.zeros(6)
        for j, i in enumerate(idx):
            full[i] = sol[j]
        return full

    def coef(self):
        if getattr(self, "_w", None) is None:
            idx = [0, 1, 2, 3, 4, 5]
            w = self._pack(idx, self._solve(idx))
            if self.vnn:
                for _ in range(2):
                    drop = [i for i in (4, 5) if i in idx and w[i] < -1e-8]
                    if not drop:
                        break
                    idx = [i for i in idx if i not in drop]
                    w = self._pack(idx, self._solve(idx))
            self._w = w
        return self._w


class BatchAccount22(E.BatchAccount21):
    def __init__(self, vnn=False, vprior=0.0, explog=False, **kw):
        super().__init__(**kw)
        BatchAccount22.current = self
        self.vnn = bool(vnn)
        self.vprior = float(vprior)
        self.explog = bool(explog)
        self.ex = {}
        if self.vnn or self.vprior > 0.0:
            self.rv = _VerifyRLS(n0=self.vprior, vnn=self.vnn)

    def _post14m(self, b, W, tag, rep, com, t_round, m_actual):
        if self.explog and tag in ("explore", "normal"):
            bk = Q.bkey(b)
            s = self.ctx14.get(bk)
            if s is not None and s["cur"] in rep:
                cur = s["cur"]
                if tag == "explore":
                    ctok = float(rep[cur]["tok"])
                    ctm = float(t_round - m_actual + rep[cur]["mt"])
                    slot = 0
                else:
                    ctok = float(sum(com))
                    ctm = float(t_round)
                    slot = 3
                if ctok > 0.0 and ctm > 0.0:
                    st = self.ex.setdefault(bk, [0.0, 0.0, 0, 0.0, 0.0, 0])
                    st[slot] += ctok
                    st[slot + 1] += ctm
                    st[slot + 2] += 1
        return super()._post14m(b, W, tag, rep, com, t_round, m_actual)

    def ex_rows(self):
        rows = []
        for bk, st in sorted(self.ex.items()):
            rows.append(dict(b=int(bk), e_tok=st[0], e_mt=st[1], e_n=int(st[2]),
                             n_tok=st[3], n_time=st[4], n_n=int(st[5])))
        return rows
