"""Numeric check of Proposition 3 (self-message-free scores) and Lemma 3.2 (cross-view gauge) on a toy MSCA-type model.
MSCA form (Section III-A): g = (1/L) sum_{l=1..L} A^l E (bipartite LightGCN), gbar = (1/L) sum_{l=0..L-1} A^l E,
item-item views h^nu_i = (S_nu Z_nu)_i, user view h^nu_u = sum_i R(u,i) h^nu_i, e_x = g_x + kappa sum_nu (1-pi^nu_x) h^nu_x,
message m(u,i) = gbar_i + kappa sum_nu (1-pi^nu_u) h^nu_i, score s(u,i) = e_u^T e_i. Attention weights fixed (random in (0,1)).
"""
import numpy as np

rng = np.random.default_rng(0)
nU, nI, d, L, kappa, nV = 7, 9, 5, 2, 0.4, 3
Rbin = (rng.random((nU, nI)) < 0.35).astype(float)
for u in range(nU):
    if Rbin[u].sum() == 0:
        Rbin[u, rng.integers(nI)] = 1
for i in range(nI):
    if Rbin[:, i].sum() == 0:
        Rbin[rng.integers(nU), i] = 1
pi_raw = rng.random((nU + nI, nV)) + 0.1
PI = pi_raw / pi_raw.sum(1, keepdims=True)              # softmax-like weights in (0,1)
S = [None] * nV
for nu in range(nV):                                     # item-item graphs with self-loops, row-normalized
    M = (rng.random((nI, nI)) < 0.3).astype(float); M = np.maximum(M, M.T); np.fill_diagonal(M, 1)
    S[nu] = M / M.sum(1, keepdims=True)


def build(Rb):
    deg_u, deg_i = Rb.sum(1), Rb.sum(0)
    Rt = Rb / np.sqrt(np.outer(np.maximum(deg_u, 1), np.maximum(deg_i, 1)))
    A = np.zeros((nU + nI, nU + nI)); A[:nU, nU:] = Rt; A[nU:, :nU] = Rt.T
    return Rt, A


def forward(theta, Rb, Q=None):
    E, Z = theta
    Rt, A = build(Rb)
    P = [np.linalg.matrix_power(A, l) for l in range(L + 1)]
    g = sum(P[l] @ E for l in range(1, L + 1)) / L
    gbar = sum(P[l] @ E for l in range(0, L)) / L
    hi = [S[nu] @ Z[nu] for nu in range(nV)]
    if Q is not None:
        hi = [hi[nu] @ Q[nu].T for nu in range(nV)]       # h -> Q h (rows are vectors)
    hu = [Rt @ hi[nu] for nu in range(nV)]
    eu = g[:nU] + kappa * sum((1 - PI[:nU, nu])[:, None] * hu[nu] for nu in range(nV))
    ei = g[nU:] + kappa * sum((1 - PI[nU:, nu])[:, None] * hi[nu] for nu in range(nV))
    return dict(Rt=Rt, g=g, gbar=gbar, hi=hi, hu=hu, eu=eu, ei=ei)


def message(f, u, i):
    return f['gbar'][nU + i] + kappa * sum((1 - PI[u, nu]) * f['hi'][nu][i] for nu in range(nV))


def rand_theta():
    return (rng.normal(size=(nU + nI, d)), [rng.normal(size=(nI, d)) for _ in range(nV)])


# --- Proposition 1: e_u = sum_i R(u,i) m(u,i)
th = rand_theta(); f = forward(th, Rbin)
err = max(np.abs(f['eu'][u] - sum(f['Rt'][u, i] * message(f, u, i) for i in range(nI))).max() for u in range(nU))
print(f'Prop1 decomposition max err {err:.2e}')

# --- Proposition 3: self summand T(theta) = R(u,p) m(u,p)^T e_p is a quadratic form in theta.
u = 0; p = int(np.flatnonzero(Rbin[u])[0])
def T(theta, Rb=Rbin, W=None):
    f = forward(theta, Rb)
    W = np.eye(d) if W is None else W
    return f['Rt'][u, p] * message(f, u, p) @ W @ f['ei'][p]
# quadratic-form check: polarization T(a+b)-T(a)-T(b) is bilinear and symmetric part nonzero for W = I
vals = [T(rand_theta()) for _ in range(5)]
print('Prop3 W=I: self summand on random tables', np.round(vals, 4))
# constructive table: only row p of E nonzero, Z = 0  -> T = R gbar_p^T g_p = R alpha beta |v|^2 > 0
E0 = np.zeros((nU + nI, d)); v = rng.normal(size=d); E0[nU + p] = v
th0 = (E0, [np.zeros((nI, d)) for _ in range(nV)])
f0 = forward(th0, Rbin)
_, A = build(Rbin)
alpha = sum(np.linalg.matrix_power(A, l)[nU + p, nU + p] for l in range(0, L)) / L
beta = sum(np.linalg.matrix_power(A, l)[nU + p, nU + p] for l in range(1, L + 1)) / L
print(f'Prop3 constructive table: T = {T(th0):.6f}, predicted R*alpha*beta*|v|^2 = {f0["Rt"][u,p]*alpha*beta*v@v:.6f}, alpha={alpha:.3f} beta={beta:.3f}')
# same with Z rows p nonzero and E row p: all terms nonneg (h_p = S_pp z) -> still >= R alpha beta |v|^2
Z1 = [np.zeros((nI, d)) for _ in range(nV)]
for nu in range(nV): Z1[nu][p] = v * rng.random()
print(f'Prop3 with view rows: T = {T((E0, Z1)):.6f} >= {f0["Rt"][u,p]*alpha*beta*v@v:.6f}')
# (a) edge removal: R(u,p)=0 -> self summand absent (Lemma 3.1)
Rm = Rbin.copy(); Rm[u, p] = 0
fm = forward(th, Rm)
print('Lemma3.1: R(u,p) after masking =', fm['Rt'][u, p])
# (b) antisymmetric kernel with message = representation: x^T W x = 0 for skew W
Wsk = rng.normal(size=(d, d)); Wsk = Wsk - Wsk.T
x = rng.normal(size=d); print(f'Prop3 (b): x^T W x for skew W = {x @ Wsk @ x:.2e}')
# but with MSCA message != representation, skew W does NOT kill the summand in general
print(f'Prop3 (b) needs the map from e_p to m(u,p) to be fixed: MSCA with skew W gives T = {T(th, W=Wsk):.4f}')

# --- Lemma 3.2: per-view orthogonal gauge on both towers
def rand_orth():
    q, _ = np.linalg.qr(rng.normal(size=(d, d))); return q
Q = [rand_orth() for _ in range(nV)]
fq = forward(th, Rbin, Q)
s0 = f['eu'] @ f['ei'].T; sq = fq['eu'] @ fq['ei'].T
print(f'Lemma3.2: max |s_Q - s| = {np.abs(sq - s0).max():.3f} (changes)')
# block decomposition
def blocks(f):
    out = {}
    eu_g, ei_g = f['g'][:nU], f['g'][nU:]
    out['gg'] = eu_g @ ei_g.T
    for nu in range(nV):
        cu = kappa * (1 - PI[:nU, nu])[:, None] * f['hu'][nu]; ci = kappa * (1 - PI[nU:, nu])[:, None] * f['hi'][nu]
        out[f'g{nu}'] = eu_g @ ci.T; out[f'{nu}g'] = cu @ ei_g.T
        for mu in range(nV):
            cmi = kappa * (1 - PI[nU:, mu])[:, None] * f['hi'][mu]
            out[f'{nu}{mu}'] = cu @ cmi.T
    return out
b0, bq = blocks(f), blocks(fq)
inv = [k for k in b0 if np.abs(b0[k] - bq[k]).max() < 1e-10]
chg = [k for k in b0 if k not in inv]
print('Lemma3.2 invariant blocks:', inv); print('Lemma3.2 changed blocks:', chg)
print('sum of blocks == score:', np.abs(sum(b0.values()) - s0).max() < 1e-10)
# norms preserved
print('norms preserved:', all(np.allclose(np.linalg.norm(f['hi'][nu], axis=1), np.linalg.norm(fq['hi'][nu], axis=1)) for nu in range(nV)))
# nonnegative self part of Prop 1(a): R kappa^2 sum (1-pi_u)(1-pi_p)|h_p|^2, and collaborative self part gbar_p^T g_p
def diag_self(f):
    return f['Rt'][u, p] * kappa**2 * sum((1 - PI[u, nu]) * (1 - PI[nU + p, nu]) * f['hi'][nu][p] @ f['hi'][nu][p] for nu in range(nV))
print(f'Lemma3.2 diagonal self part: {diag_self(f):.6f} vs gauged {diag_self(fq):.6f}')
# first-order: gradient of full-graph BPR loss along a skew generator is NONZERO in general (relational cross-view terms),
# while the diagonal self part has zero derivative.
Gs = [rng.normal(size=(d, d)) for _ in range(nV)]; Gs = [G - G.T for G in Gs]
expm = lambda X: np.linalg.solve(np.eye(X.shape[0]) - X / 2, np.eye(X.shape[0]) + X / 2)  # Cayley map, orthogonal for skew X
def bpr(f):
    tot = 0
    for uu in range(nU):
        for pp in np.flatnonzero(Rbin[uu]):
            for nn in np.flatnonzero(1 - Rbin[uu]):
                tot += np.log1p(np.exp(-(f['eu'][uu] @ f['ei'][pp] - f['eu'][uu] @ f['ei'][nn])))
    return tot
eps = 1e-5
fp = forward(th, Rbin, [expm(eps * G) for G in Gs]); fmn = forward(th, Rbin, [expm(-eps * G) for G in Gs])
print(f'd BPR / d gauge (central diff) = {(bpr(fp) - bpr(fmn)) / (2 * eps):.4e}  (nonzero: full-graph BPR does move the gauge through cross-view products)')
print(f'd diag-self / d gauge = {(diag_self(fp) - diag_self(fmn)) / (2 * eps):.2e}')
# self summand cross-view part: derivative nonzero as well
def self_summand(f):
    return f['Rt'][u, p] * message(f, u, p) @ f['ei'][p]
print(f'd self-summand / d gauge = {(self_summand(fp) - self_summand(fmn)) / (2 * eps):.4e} (the cross-view part of the self summand depends on the gauge)')
# masked view: self summand absent, gauge acts only through other history items
fmq = forward(th, Rm, Q)
print('masked view: R(u,p) =', fmq['Rt'][u, p])
