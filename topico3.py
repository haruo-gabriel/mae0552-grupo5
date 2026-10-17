"""
Topico 3 - Compressao de Dados, AEP e o Criterio de Kelly
Teoria da Informacao - Primeira atividade em grupo

Uso:     python topico3.py      (precisa de topico1.py na mesma pasta)
Requer:  numpy, matplotlib
Saida:   resultados/fig_t3_*.png e resultados/tab_t3_*.tex
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from topico1 import entropia, divergencia_kl, info_mutua, salvar_tabela, PASTA

rng = np.random.default_rng(2026)

def fmt(x, casas=4):
    if np.isneginf(x): return "$-\\infty$"
    if np.isposinf(x): return "$+\\infty$"
    return f"{x:.{casas}f}"

def W(b, p, o):
    """W(b,p) = soma p_i log2(b_i o_i)."""
    b, p, o = (np.asarray(v, float) for v in (b, p, o))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log2(b * o), 0.0)
    return float(np.sum(t))

def log_inc(X, b, o):
    """Incrementos log2(b_X o_X) para uma matriz de resultados X (temporadas x rodadas)."""
    with np.errstate(divide="ignore"):
        return np.log2(np.asarray(b, float) * np.asarray(o, float))[X]

# =====================================================================
# Passo 1: mundo real
# =====================================================================
m = 3
p = np.array([0.5, 0.3, 0.2])                 # casa, empate, visitante
o_justa = np.full(m, 3.0)                     # casa "ingenua": o_i = m
o_casa = np.array([1.8, 3.5, 5.0])            # casa com margem (soma 1/o > 1)
H_p = entropia(p)
print(f"== Passo 1 == p = {p}, H(p) = {H_p:.4f}, log2 m - H(p) = {np.log2(m)-H_p:.4f}")

# =====================================================================
# Passos 2-7: tres estrategias, R temporadas de n partidas
# =====================================================================
o = o_justa
R, n = 1000, 1000
Xs = rng.choice(m, size=(R, n), p=p)          # uma linha por temporada
estr = {"Tudo no favorito": np.array([1.0, 0.0, 0.0]),
        "Uniforme": np.full(m, 1 / m),
        "Kelly ($b=p$)": p.copy()}
traj, linhas6 = {}, []
print(f"\n== Passos 4-7 (o = {o}, {R} temporadas de {n} partidas) ==")
for nome, b in estr.items():
    traj[nome] = np.cumsum(log_inc(Xs, b, o), axis=1)   # log2(S_j / S_0)
    L = traj[nome][:, -1]
    taxa = L / n
    W_teo = W(b, p, o)
    logES = np.log2(np.sum(p * b * o))                   # (1/n) log2 E[S_n]
    quebra = np.mean(L < np.log2(1e-3))
    sorte = np.mean(L > np.log2(1e3))
    print(f"{nome:<18} W teo {fmt(W_teo)} | taxa media {fmt(np.mean(taxa))} | mediana "
          f"{fmt(np.median(taxa))} | quebrou {quebra:.3f} | >1000x {sorte:.3f} | (1/n)log2 E[S] {logES:.4f}")
    linhas6.append([nome, fmt(W_teo), fmt(np.mean(taxa)), fmt(np.median(taxa)),
                    f"{100*quebra:.1f}\\%", f"{100*sorte:.1f}\\%", f"{logES:.4f}"])
salvar_tabela("tab_t3_passo6.tex",
              ["Estratégia", "$W(b,p)$", "Taxa média", "Taxa mediana", "$S_n<10^{-3}S_0$",
               "$S_n>10^{3}S_0$", "$\\frac1n\\log_2 E[S_n]$"], linhas6)

fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for a, (nome, t) in zip(ax, traj.items()):
    t20 = np.where(np.isfinite(t[:20]), t[:20], np.nan)
    a.plot(np.arange(1, n + 1), t20.T, lw=0.8, alpha=0.7)
    a.plot(np.arange(1, n + 1), np.arange(1, n + 1) * W(estr[nome], p, o) if
           np.isfinite(W(estr[nome], p, o)) else np.full(n, np.nan), "k--", lw=2, label="$jW(b,p)$")
    a.set_title(nome); a.set_xlabel("partida $j$"); a.grid(alpha=0.3)
ax[0].set_ylabel("$\\log_2(S_j/S_0)$")
ax[0].text(0.05, 0.1, "quebra na 1ª derrota\n($\\log_2 S = -\\infty$)", transform=ax[0].transAxes)
ax[2].legend()
plt.suptitle("Passo 4: evolução do patrimônio (20 temporadas, cotações justas $o_i=3$)")
plt.tight_layout(); plt.savefig(os.path.join(PASTA, "fig_t3_passo4.png"), dpi=150); plt.close()

# =====================================================================
# Passo 8: concentracao (AEP) da taxa de Kelly e de -1/n log2 p(X^n)
# =====================================================================
W_star = W(p, p, o)
t_k = np.log2(p * o); sd_k = np.sqrt(np.sum(p * t_k ** 2) - W_star ** 2)
t_a = -np.log2(p);    sd_a = np.sqrt(np.sum(p * t_a ** 2) - H_p ** 2)
R8, eps = 4000, 0.05
linhas8 = []
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
print("\n== Passo 8 ==")
for n8 in [10, 100, 1000]:
    X8 = rng.choice(m, size=(R8, n8), p=p)
    taxa = log_inc(X8, p, o).sum(1) / n8
    aep = t_a[X8].mean(axis=1)                         # -1/n log2 p(X_1..X_n)
    tip = np.mean(np.abs(aep - H_p) < eps)
    print(f"n={n8:>5}: taxa Kelly media {taxa.mean():.4f} dp {taxa.std():.4f} (teo {sd_k/np.sqrt(n8):.4f})"
          f" | AEP media {aep.mean():.4f} dp {aep.std():.4f} | tipicas (eps={eps}) {tip:.3f}")
    linhas8.append([str(n8), f"{taxa.mean():.4f}", f"{taxa.std():.4f}", f"{sd_k/np.sqrt(n8):.4f}",
                    f"{aep.mean():.4f}", f"{aep.std():.4f}", f"{100*tip:.1f}\\%"])
    ax[0].hist(taxa, bins=60, density=True, histtype="step", lw=1.5, label=f"n = {n8}")
    ax[1].hist(aep, bins=60, density=True, histtype="step", lw=1.5, label=f"n = {n8}")
ax[0].axvline(W_star, color="red", ls="--", label=f"$W^*$ = {W_star:.4f}")
ax[1].axvline(H_p, color="red", ls="--", label=f"$H(p)$ = {H_p:.4f}")
ax[0].set_xlabel("$\\frac{1}{n}\\log_2(S_n/S_0)$ (Kelly)"); ax[1].set_xlabel("$-\\frac{1}{n}\\log_2 p(X^n)$")
for a in ax: a.legend(fontsize=8); a.grid(alpha=0.3)
plt.suptitle("Passo 8: equipartição assintótica")
plt.tight_layout(); plt.savefig(os.path.join(PASTA, "fig_t3_passo8.png"), dpi=150); plt.close()
salvar_tabela("tab_t3_passo8.tex",
              ["$n$", "Taxa Kelly (média)", "dp", "dp teórico", "$-\\frac1n\\log_2 p(X^n)$ (média)",
               "dp", f"Típicas ($\\varepsilon={eps}$)"], linhas8)

# =====================================================================
# Passos 9-10: informacao lateral
# =====================================================================
Py_x = np.array([[0.6, 0.2, 0.2],
                 [0.2, 0.6, 0.2],
                 [0.2, 0.2, 0.6]])               # P(Y = y | X = x), linhas = x
conj = p[:, None] * Py_x                          # p(x, y), m x k
k = conj.shape[1]
p_x_dado_y = conj / conj.sum(axis=0, keepdims=True)   # colunas normalizadas
I_XY = info_mutua(conj)
idx = rng.choice(m * k, size=(R, n), p=conj.ravel())  # sorteio no vetor achatado
Xc, Yc = np.divmod(idx, k)                            # idx = x*k + y
print(f"\n== Passos 9-10 == I(X;Y) = {I_XY:.4f} bits")
linhas10 = []
for rot, oo in [("Justas $o_i=3$", o_justa), ("Casa $(1{,}8;\\,3{,}5;\\,5)$", o_casa)]:
    W_sem = W(p, p, oo)
    W_com = float(np.sum(conj * np.log2(p_x_dado_y * oo[:, None])))
    taxa_sem = log_inc(Xc, p, oo).sum(1) / n
    taxa_com = np.log2(p_x_dado_y[Xc, Yc] * oo[Xc]).sum(1) / n
    ganho = np.mean(taxa_com - taxa_sem)
    print(f"{rot}: W* {W_sem:.4f} | W com dica {W_com:.4f} | ganho teo {W_com-W_sem:.4f}"
          f" | ganho empirico {ganho:.4f} | I(X;Y) {I_XY:.4f}")
    linhas10.append([rot, f"{W_sem:.4f}", f"{W_com:.4f}", f"{W_com-W_sem:.4f}",
                     f"{ganho:.4f}", f"{I_XY:.4f}"])
salvar_tabela("tab_t3_passo10.tex",
              ["Cotações", "$W^*$ sem dica", "$W$ com dica", "$\\Delta W$ teórico",
               "$\\Delta W$ empírico", "$I(X;Y)$"], linhas10)

# =====================================================================
# Passo 11: apostador com crencas erradas q
# =====================================================================
q = np.array([0.65, 0.20, 0.15])                  # superestima o favorito
D_pq = divergencia_kl(p, q)
print(f"\n== Passo 11 == q = {q}, D(p||q) = {D_pq:.4f}")
linhas11 = []
for rot, oo in [("Justas $o_i=3$", o_justa), ("Casa $(1{,}8;\\,3{,}5;\\,5)$", o_casa)]:
    W_p, W_q = W(p, p, oo), W(q, p, oo)
    perda_emp = np.mean(log_inc(Xs, p, oo).sum(1) / n - log_inc(Xs, q, oo).sum(1) / n)
    print(f"{rot}: W* {W_p:.4f} | W(q,p) {W_q:.4f} | perda teo {W_p-W_q:.4f} | "
          f"perda emp {perda_emp:.4f} | D(p||q) {D_pq:.4f}")
    linhas11.append([rot, f"{W_p:.4f}", f"{W_q:.4f}", f"{W_p-W_q:.4f}", f"{perda_emp:.4f}", f"{D_pq:.4f}"])
salvar_tabela("tab_t3_passo11.tex",
              ["Cotações", "$W^*$", "$W(q,p)$", "Perda teórica", "Perda empírica", "$D(p\\|q)$"], linhas11)

print("\nArquivos gerados em", PASTA + "/")