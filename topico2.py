"""
Topico 2 - Desigualdade de Processamento de Dados e Cadeias de Markov
Teoria da Informacao - Primeira atividade em grupo

Uso:     python topico2.py      (precisa de topico1.py na mesma pasta)
Requer:  numpy, scipy, matplotlib
Saida:   resultados/fig_t2_*.png e resultados/tab_t2_*.tex
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from topico1 import (entropia, info_mutua, info_mutua_condicional, divergencia_kl,
                     tabela_contagens, num, salvar_tabela, PASTA)

rng = np.random.default_rng(2026)

def Hb(p):
    return entropia([p, 1 - p])

def canal_bsc(bits, p):
    """Inverte cada bit com probabilidade p (uniforme < p  ->  XOR com 1)."""
    erros = (rng.random(bits.size) < p).astype(bits.dtype)
    return bits ^ erros

def cadeia(X, p, k):
    """X -> Y1 -> ... -> Yk; guarda TODOS os estagios."""
    estagios = [X]
    for _ in range(k):
        estagios.append(canal_bsc(estagios[-1], p))
    return estagios

def I_teorica_cadeia(p, j):
    """X uniforme; j BSC(p) em serie equivalem a um BSC com p_j = (1-(1-2p)^j)/2."""
    return 1 - Hb((1 - (1 - 2 * p) ** j) / 2)

def I_amostral(estagios):
    return np.array([info_mutua(tabela_contagens(estagios[0], Y)) for Y in estagios])

# =====================================================================
# Passos 1-5: cadeia de BSCs com p = 0,1
# =====================================================================
p, k, N = 0.10, 10, 200_000
X = rng.integers(0, 2, N, dtype=np.int8)
est = cadeia(X, p, k)
I_est = I_amostral(est)
I_teo = np.array([I_teorica_cadeia(p, j) for j in range(k + 1)])
solav = int(np.sum(np.diff(I_est) > 0))

tab3 = np.zeros((2, 2, 2), dtype=np.int64)
np.add.at(tab3, (est[0], est[2], est[1]), 1)
I_X_Y2_dado_Y1 = info_mutua_condicional(tab3)       # deve ser ~0 (Markov)
tab3b = np.zeros((2, 2, 2), dtype=np.int64)
np.add.at(tab3b, (est[0], est[1], est[2]), 1)
I_X_Y1_dado_Y2 = info_mutua_condicional(tab3b)      # > 0 (ordem "errada")

print(f"== Passos 1-5: p = {p}, k = {k}, N = {N} ==")
for j in range(k + 1):
    print(f"I(X;Y{j:<2}) estimada {I_est[j]:.5f} | teorica {I_teo[j]:.5f}")
print(f"aumentos ('solavancos') na sequencia: {solav}")
print(f"I(X;Y2|Y1) = {I_X_Y2_dado_Y1:.6f}  (~0)  |  I(X;Y1|Y2) = {I_X_Y1_dado_Y2:.4f}  (>0)")

salvar_tabela("tab_t2_passo4.tex", ["Estágio $j$", "$\\hat I(X;Y_j)$", "$I(X;Y_j)$ teórico", "Diferença"],
              [[str(j), num(I_est[j]), num(I_teo[j]), f"{I_est[j]-I_teo[j]:+.5f}"]
               for j in range(k + 1)]
              + [["$\\hat I(X;Y_2|Y_1)$", num(I_X_Y2_dado_Y1), "0", ""],
                 ["$\\hat I(X;Y_1|Y_2)$", num(I_X_Y1_dado_Y2), "", ""]])

# Solavancos x tamanho da amostra (p = 0,2)
print("\nSolavancos (p = 0,2; media em 200 repeticoes):")
for Ns in [200, 2000, 20000]:
    n_sol = [np.sum(np.diff(I_amostral(cadeia(rng.integers(0, 2, Ns, dtype=np.int8), 0.2, k))) > 0)
             for _ in range(200)]
    print(f"  N = {Ns:>6}: {np.mean(n_sol):.2f} aumentos por cadeia de {k} elos")

# =====================================================================
# Passos 5-6: varios niveis de ruido
# =====================================================================
ps = [0.01, 0.05, 0.10, 0.20, 0.30]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
ax[0].plot(range(k + 1), I_est, "o", ms=8, label="estimado")
ax[0].plot(range(k + 1), I_teo, "-", label="teórico")
ax[0].set_title(f"Passo 5: $I(X;Y_j)$, $p$ = {p}")
for pp in ps:
    Ie = I_amostral(cadeia(rng.integers(0, 2, N, dtype=np.int8), pp, k))
    l, = ax[1].plot(range(k + 1), Ie, "o", ms=5)
    ax[1].plot(range(k + 1), [I_teorica_cadeia(pp, j) for j in range(k + 1)], "-",
               color=l.get_color(), label=f"$p$ = {pp}")
ax[1].set_title("Passo 6: diferentes intensidades de ruído")
for a in ax:
    a.set_xlabel("posição $j$ na cadeia"); a.set_ylabel("$I(X;Y_j)$ (bits)")
    a.grid(alpha=0.3); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(PASTA, "fig_t2_passo5_6.png"), dpi=150); plt.close()

# =====================================================================
# Passos 7-8: rotulo -> "imagem" -> compressao -> predicao
# =====================================================================
dim, delta, sigma = 16, 0.4, 1.0
s = np.array([1, 1, 1, 1,  1, 1, 1, -1,  1, 1, -1, -1,  1, -1, 1, -1], dtype=float)
mu = {0: np.zeros(dim), 1: delta * s}                  # 0 = gato, 1 = cachorro
A = np.kron(np.eye(4), np.full((1, 4), 0.25))           # 4 x 16: medias de blocos de 4
N7 = 200_000
X7 = rng.integers(0, 2, N7)
img = np.where(X7[:, None] == 1, mu[1], mu[0]) + sigma * rng.standard_normal((N7, dim))
comp = img @ A.T                                        # representacao comprimida
c0, c1 = A @ mu[0], A @ mu[1]
pred = (((comp - c1) ** 2).sum(1) < ((comp - c0) ** 2).sum(1)).astype(np.int64)

def estatistica_suficiente(Z, m0, m1, Sigma):
    """Log-razao de verossimilhanca (gaussianas, mesma covariancia): projecao 1D
    que preserva TODA a informacao de Z sobre X, logo I(X;Z) = I(X;L)."""
    w = np.linalg.solve(Sigma, m1 - m0)
    return (Z - (m0 + m1) / 2) @ w

def I_continua(Xr, L, bins=50):
    """Plug-in com discretizacao de L em caixas de quantis (igualmente povoadas)."""
    bordas = np.quantile(L, np.linspace(0, 1, bins + 1))[1:-1]
    idx = np.searchsorted(bordas, L, side="right")
    return info_mutua(tabela_contagens(Xr, idx, 2, bins))

def I_gauss_binaria(d2):
    """I(X;L), X uniforme, L|X ~ N(+-d2/2, d2): integracao numerica."""
    d = np.sqrt(d2)
    l = np.linspace(-d2 / 2 - 10 * d, d2 / 2 + 10 * d, 40001)
    f1, f0 = norm.pdf(l, d2 / 2, d), norm.pdf(l, -d2 / 2, d)
    f = (f0 + f1) / 2
    m = f > 0
    integ = np.zeros_like(l)
    integ[m] = 0.5 * f1[m] * np.log2(f1[m] / f[m] + 1e-300) + 0.5 * f0[m] * np.log2(f0[m] / f[m] + 1e-300)
    return float(np.sum(integ) * (l[1] - l[0]))

Sig_img, Sig_comp = sigma ** 2 * np.eye(dim), sigma ** 2 * A @ A.T
L_img = estatistica_suficiente(img, mu[0], mu[1], Sig_img)
L_comp = estatistica_suficiente(comp, c0, c1, Sig_comp)
d2_img = float((mu[1] - mu[0]) @ np.linalg.solve(Sig_img, mu[1] - mu[0]))
d2_comp = float((c1 - c0) @ np.linalg.solve(Sig_comp, c1 - c0))
Pe_teo = norm.cdf(-np.sqrt(d2_comp) / 2)
Pe_est = np.mean(pred != X7)

estagios8 = [("Imagem completa", dim, I_continua(X7, L_img), I_gauss_binaria(d2_img)),
             ("Representação comprimida", 4, I_continua(X7, L_comp), I_gauss_binaria(d2_comp)),
             ("Predição final", 1, info_mutua(tabela_contagens(X7, pred)), 1 - Hb(Pe_teo))]
print("\n== Passos 7-8 ==")
print(f"d^2 imagem = {d2_img:.3f}, d^2 comprimida = {d2_comp:.3f}")
print(f"erro de classificacao: medido {Pe_est:.4f} | teorico {Pe_teo:.4f}")
for nome, dmn, Ie, It in estagios8:
    print(f"{nome:<26} I estimada {Ie:.4f} | teorica {It:.4f}")
print(f"(sensibilidade: imagem com 10 caixas -> {I_continua(X7, L_img, 10):.4f})")
salvar_tabela("tab_t2_passo8.tex", ["Estágio", "Dimensão", "$\\hat I(X;\\cdot)$", "$I(X;\\cdot)$ teórico"],
              [[n, str(dd), f"{Ie:.4f}", f"{It:.4f}"] for n, dd, Ie, It in estagios8]
              + [["Erro de classificação", "", f"{Pe_est:.4f}", f"{Pe_teo:.4f}"]])

# =====================================================================
# Passo 9: colapso de modelo (treino recursivo em dados sinteticos)
# =====================================================================
K = 10
p_real = 1.0 / np.arange(1, K + 1); p_real /= p_real.sum()       # tipo Zipf
G, RUNS = 100, 200
tamanhos = [20, 100, 1000]
res9 = {}
for n_am in tamanhos:
    D_rq, D_qr, sup, Hq = (np.zeros((RUNS, G)) for _ in range(4))
    for r in range(RUNS):
        q = p_real.copy()
        for g in range(G):
            amostra = rng.choice(K, size=n_am, p=q)                # geracao g
            q = np.bincount(amostra, minlength=K) / n_am             # frequencia relativa
            D_rq[r, g] = divergencia_kl(p_real, q)                   # inf se perdeu categoria
            D_qr[r, g] = divergencia_kl(q, p_real)
            sup[r, g] = np.count_nonzero(q)
            Hq[r, g] = entropia(q)
    res9[n_am] = (D_rq, D_qr, sup, Hq)

print(f"\n== Passo 9: K = {K}, H(p_real) = {entropia(p_real):.4f} ==")
linhas9 = []
for n_am in tamanhos:
    D_rq, D_qr, sup, Hq = res9[n_am]
    for g in [0, 9, 49, 99]:
        frac_inf = np.mean(np.isinf(D_rq[:, g]))
        print(f"n={n_am:>5} ger {g:>3}: D(q||p) med {np.median(D_qr[:, g]):.4f} | "
              f"D(p||q)=inf em {frac_inf:.2f} | suporte {sup[:, g].mean():.2f} | H {Hq[:, g].mean():.3f}")
        linhas9.append([str(n_am), str(g), num(np.median(D_qr[:, g])), f"{100*frac_inf:.0f}\\%",
                        f"{sup[:, g].mean():.2f}", f"{Hq[:, g].mean():.3f}"])
salvar_tabela("tab_t2_passo9.tex",
              ["$n$/geração", "Geração", "Mediana $D(q_g\\|p)$", "$D(p\\|q_g)=\\infty$",
               "Categorias vivas", "$H(q_g)$ média"], linhas9)

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for n_am in tamanhos:
    D_rq, D_qr, sup, Hq = res9[n_am]
    ax[0].plot(np.median(D_qr, axis=0), label=f"n = {n_am}")
    ax[1].plot(sup.mean(axis=0), label=f"n = {n_am}")
    ax[2].plot(np.mean(np.isinf(D_rq), axis=0), label=f"n = {n_am}")
ax[0].set_ylabel("mediana de $D(q_g\\|p_{real})$ (bits)")
ax[1].set_ylabel("nº médio de categorias com $q_g>0$")
ax[2].set_ylabel("fração com $D(p_{real}\\|q_g)=\\infty$")
for a in ax:
    a.set_xlabel("geração $g$"); a.grid(alpha=0.3); a.legend(fontsize=8)
plt.suptitle("Passo 9: colapso de modelo")
plt.tight_layout(); plt.savefig(os.path.join(PASTA, "fig_t2_passo9.png"), dpi=150); plt.close()

print("\nArquivos gerados em", PASTA + "/")