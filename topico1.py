"""
Topico 1 - Entropia, Informacao Mutua e o Teste de Independencia
Teoria da Informacao - Primeira atividade em grupo

Uso:     python topico1.py
Requer:  numpy, scipy, matplotlib
Saida:   pasta resultados/ com figuras (fig_t1_*.png) e tabelas (tab_t1_*.tex)

As funcoes basicas ficam no topo e sao reaproveitadas nos Topicos 2 e 3
(from topico1 import ...). A simulacao so roda se este arquivo for executado.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2, chi2_contingency

PASTA = "resultados"
os.makedirs(PASTA, exist_ok=True)

# =====================================================================
# Funcoes basicas (reaproveitadas nos outros topicos)
# =====================================================================
def entropia(v):
    """H em bits de um vetor (ou tabela) de probabilidades OU contagens.
    Normaliza internamente; termos com p = 0 sao descartados (0 log 0 = 0)."""
    v = np.asarray(v, dtype=float).ravel()
    v = v / v.sum()
    v = v[v > 0]
    return float(-np.sum(v * np.log2(v)))

def entropia_conjunta(tab):
    """H(A,B): entropia de todas as celulas da tabela conjunta."""
    return entropia(tab)

def info_mutua(tab):
    """I(A;B) = soma p(a,b) log2[ p(a,b) / (p(a)p(b)) ]  (eq. (2) do enunciado)."""
    P = np.asarray(tab, dtype=float)
    P = P / P.sum()
    pa = P.sum(axis=1, keepdims=True)
    pb = P.sum(axis=0, keepdims=True)
    m = P > 0
    return float(np.sum(P[m] * np.log2(P[m] / (pa * pb)[m])))

def info_mutua_condicional(tab3):
    """I(A;B|C) a partir de uma tabela 3D indexada [a, b, c]."""
    T = np.asarray(tab3, dtype=float)
    T = T / T.sum()
    pc = T.sum(axis=(0, 1))
    return float(sum(pc[c] * info_mutua(T[:, :, c]) for c in range(T.shape[2]) if pc[c] > 0))

def entropia_cruzada(p, q):
    """H(p,q) = -soma p log2 q."""
    p = np.asarray(p, float); p = p / p.sum()
    q = np.asarray(q, float); q = q / q.sum()
    m = p > 0
    if np.any(q[m] == 0):
        return np.inf
    return float(-np.sum(p[m] * np.log2(q[m])))

def divergencia_kl(p, q):
    """D(p||q) em bits (infinita se q = 0 onde p > 0)."""
    p = np.asarray(p, float); p = p / p.sum()
    q = np.asarray(q, float); q = q / q.sum()
    m = p > 0
    if np.any(q[m] == 0):
        return np.inf
    return float(np.sum(p[m] * np.log2(p[m] / q[m])))

def tabela_contagens(a, b, na=2, nb=2):
    """Conta 'na mao' quantas vezes cada par (a, b) aparece."""
    tab = np.zeros((na, nb), dtype=np.int64)
    np.add.at(tab, (np.asarray(a), np.asarray(b)), 1)
    return tab

def teste_independencia(tab):
    """Estatistica G = 2 N ln(2) I_hat ~ qui-quadrado com (|A|-1)(|B|-1) g.l. sob H0."""
    tab = np.asarray(tab)
    N = tab.sum()
    I = info_mutua(tab)
    G = 2 * N * np.log(2) * I
    gl = (tab.shape[0] - 1) * (tab.shape[1] - 1)
    return G, gl, chi2.sf(G, gl), I

# ---------------- Utilitarios para o relatorio ----------------
def num(x):
    """Formata numero (>= 0) para LaTeX."""
    if x is None:
        return "--"
    if x == 0:
        return "0"
    if x >= 1e-3:
        return f"{x:.4f}"
    m, e = f"{x:.2e}".split("e")
    return f"${m}\\times 10^{{{int(e)}}}$"

def salvar_tabela(nome, colunas, linhas):
    with open(os.path.join(PASTA, nome), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l" + "r" * (len(colunas) - 1) + "}\n\\toprule\n")
        f.write(" & ".join(colunas) + " \\\\\n\\midrule\n")
        for l in linhas:
            f.write(" & ".join(l) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------- Modelo do Topico 1 ----------------
def modelo_conjunto(pD, fp, fn):
    """Tabela teorica p(d,t): linhas d = 0,1; colunas t = 0,1."""
    return np.array([[(1 - pD) * (1 - fp), (1 - pD) * fp],
                     [pD * fn,             pD * (1 - fn)]])

def simula_DT(N, pD, fp, fn, rng):
    """Sorteia D, depois T condicionalmente a D (falso positivo / falso negativo)."""
    D = (rng.random(N) < pD).astype(np.int64)
    pT1 = np.where(D == 1, 1 - fn, fp)
    T = (rng.random(N) < pT1).astype(np.int64)
    return D, T

def quantidades(tab):
    H_D = entropia(tab.sum(axis=1))
    H_T = entropia(tab.sum(axis=0))
    H_DT = entropia_conjunta(tab)
    return {"H(D)": H_D, "H(T)": H_T, "H(D,T)": H_DT,
            "H(D|T)": H_DT - H_T, "I(D;T)": info_mutua(tab)}

def simula_markov(n_dias, M, P, pi0, rng):
    """M historicos de n_dias; P[i, j] = P(T_hoje = j | T_ontem = i).
    Laco simples nos dias (vetorizado nos pacientes)."""
    T = np.zeros((M, n_dias), dtype=np.int64)
    T[:, 0] = rng.random(M) < pi0[1]
    for i in range(1, n_dias):
        T[:, i] = rng.random(M) < P[T[:, i - 1], 1]
    return T

def pv_tex(pv):
    return "$<10^{-300}$" if pv == 0 else num(pv)

# =====================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(2026)

    # ---------------- Passo 1 ----------------
    pD, FP, FN = 0.10, 0.05, 0.20
    tab_teo = modelo_conjunto(pD, FP, FN)
    teo = quantidades(tab_teo)
    print("== Passo 1: modelo verdadeiro ==")
    print(f"p(D=1) = {pD}, falso positivo = {FP}, falso negativo = {FN}")
    print("p(d,t) =\n", tab_teo)
    for k, v in teo.items():
        print(f"  {k:<7} = {v:.4f} bits")
    H_T_dado_D = teo["H(D,T)"] - teo["H(D)"]
    print(f"  conferencia da simetria: H(T) - H(T|D) = {teo['H(T)'] - H_T_dado_D:.4f}")

    # ---------------- Passos 2-3 ----------------
    N = 5000
    D, T = simula_DT(N, pD, FP, FN, rng)
    tab_emp = tabela_contagens(D, T)
    est = quantidades(tab_emp)
    print(f"\n== Passos 2-3: N = {N} pacientes ==")
    print("contagens (linhas D=0,1; colunas T=0,1):\n", tab_emp)
    for k in teo:
        print(f"{k:<8} teorico {teo[k]:.4f} | estimado {est[k]:.4f} | dif {est[k]-teo[k]:+.4f}")
    salvar_tabela("tab_t1_passo3.tex",
                  ["Quantidade", "Teórico", f"Estimado ($N={N}$)", "Diferença"],
                  [[f"${k}$", f"{teo[k]:.4f}", f"{est[k]:.4f}", f"{est[k]-teo[k]:+.4f}"]
                   for k in teo])

    # ---------------- Passo 4: erro e vies em funcao de N ----------------
    Ns = [20, 50, 100, 500, 2000, 10000]
    REP = 500
    FP0, FN0 = 0.30, 0.70        # cenario independente: P(T=1|D) = 0,3 sempre
    med, dp, rmse, med0 = [], [], [], []
    for n in Ns:
        I_r = np.array([info_mutua(tabela_contagens(*simula_DT(n, pD, FP, FN, rng)))
                        for _ in range(REP)])
        I_r0 = np.array([info_mutua(tabela_contagens(*simula_DT(n, pD, FP0, FN0, rng)))
                         for _ in range(REP)])
        med.append(I_r.mean()); dp.append(I_r.std())
        rmse.append(np.sqrt(np.mean((I_r - teo["I(D;T)"]) ** 2)))
        med0.append(I_r0.mean())
    print(f"\n== Passo 4 ({REP} repeticoes por N) ==")
    linhas4 = []
    for i, n in enumerate(Ns):
        prev0 = 1 / (2 * n * np.log(2))
        print(f"N={n:>6}: media {med[i]:.4f} vies {med[i]-teo['I(D;T)']:+.4f} RMSE {rmse[i]:.4f}"
              f" | indep.: media {med0[i]:.5f} (previsto {prev0:.5f})")
        linhas4.append([str(n), f"{med[i]:.4f}", f"{med[i]-teo['I(D;T)']:+.4f}",
                        f"{rmse[i]:.4f}", num(med0[i]), num(prev0)])
    salvar_tabela("tab_t1_passo4.tex",
                  ["$N$", "Média $\\hat I$", "Viés", "RMSE",
                   "Média $\\hat I$ (indep.)", "$1/(2N\\ln 2)$"], linhas4)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].errorbar(Ns, med, yerr=dp, fmt="o-", capsize=4, label="média ± dp de $\\hat I$")
    ax[0].axhline(teo["I(D;T)"], color="red", ls="--", label="valor teórico")
    ax[0].set_xscale("log"); ax[0].set_xlabel("$N$"); ax[0].set_ylabel("bits")
    ax[0].set_title("Estimativa de $I(D;T)$ (modelo verdadeiro)"); ax[0].legend()
    ax[1].loglog(Ns, med0, "o", ms=8, label="média de $\\hat I$ (D e T independentes)")
    ax[1].loglog(Ns, 1 / (2 * np.array(Ns) * np.log(2)), "-", label="previsão $1/(2N\\ln 2)$")
    ax[1].set_xlabel("$N$"); ax[1].set_ylabel("bits")
    ax[1].set_title("Viés positivo quando $I=0$"); ax[1].legend()
    for a in ax: a.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(PASTA, "fig_t1_passo4.png"), dpi=150); plt.close()

    # ---------------- Passo 5: entropia cruzada e KL ----------------
    p_D = np.array([1 - pD, pD])
    q_D = np.array([0.70, 0.30])          # modelo "de outra regiao"
    H_p, H_pq = entropia(p_D), entropia_cruzada(p_D, q_D)
    D_pq, D_qp = divergencia_kl(p_D, q_D), divergencia_kl(q_D, p_D)
    custo_p = np.mean(-np.log2(p_D[D]))   # comprimentos ideais -log2 p(d) na amostra
    custo_q = np.mean(-np.log2(q_D[D]))
    print("\n== Passo 5 ==")
    print(f"H(p) = {H_p:.4f}, H(p,q) = {H_pq:.4f}, H(p,q)-H(p) = {H_pq-H_p:.4f}")
    print(f"D(p||q) = {D_pq:.4f}, D(q||p) = {D_qp:.4f}  (assimetria)")
    print(f"custo empirico: com p {custo_p:.4f}, com q {custo_q:.4f}, excesso {custo_q-custo_p:.4f}")
    salvar_tabela("tab_t1_passo5.tex", ["Quantidade", "Bits"],
                  [["$H(p)$", f"{H_p:.4f}"], ["$H(p,q)$", f"{H_pq:.4f}"],
                   ["$H(p,q)-H(p)$", f"{H_pq-H_p:.4f}"], ["$D(p\\|q)$", f"{D_pq:.4f}"],
                   ["$D(q\\|p)$", f"{D_qp:.4f}"],
                   [f"Custo médio na amostra com $p$ ($N={N}$)", f"{custo_p:.4f}"],
                   ["Custo médio na amostra com $q$", f"{custo_q:.4f}"],
                   ["Excesso medido", f"{custo_q-custo_p:.4f}"]])

    # ---------------- Passos 6-7: historico com cadeia de Markov ----------------
    Pm = np.array([[0.90, 0.10],
                   [0.20, 0.80]])
    pi = np.array([2 / 3, 1 / 3])                     # estacionaria
    assert np.allclose(pi @ Pm, pi)
    H1_teo = entropia(pi)
    Hc_teo = float(sum(pi[i] * entropia(Pm[i]) for i in range(2)))
    Iviz_teo = H1_teo - Hc_teo

    n_dias, M = 8, 200_000
    Tm = simula_markov(n_dias, M, Pm, pi, rng)

    def H_prefixo(i):
        """Entropia empirica de (T_1, ..., T_i): cada historico vira um inteiro."""
        if i == 0:
            return 0.0
        codigo = Tm[:, :i] @ (1 << np.arange(i))
        return entropia(np.bincount(codigo, minlength=2 ** i))

    Hpref = np.array([H_prefixo(i) for i in range(n_dias + 1)])
    H_seq = Hpref[-1]
    cond_passado = np.diff(Hpref)                       # H(T_i | T_1..T_{i-1})
    cond_ontem = np.array([entropia(np.bincount(Tm[:, 0], minlength=2))] +
                          [entropia(tabela_contagens(Tm[:, i - 1], Tm[:, i]))
                           - entropia(np.bincount(Tm[:, i - 1], minlength=2))
                           for i in range(1, n_dias)])  # H(T_i | T_{i-1})
    H1_est = entropia(np.bincount(Tm[:, 0], minlength=2))
    ingenuo = n_dias * H1_est
    H_seq_teo = H1_teo + (n_dias - 1) * Hc_teo
    print(f"\n== Passos 6-7: {M} historicos de {n_dias} dias ==")
    print(f"H(T_1..T_n) estimada  = {H_seq:.4f} (teorica {H_seq_teo:.4f})")
    print(f"soma regra da cadeia  = {cond_passado.sum():.4f}")
    print(f"soma so com 'ontem'   = {cond_ontem.sum():.4f}")
    print(f"ingenuo n*H(T_1)      = {ingenuo:.4f} (teorico {n_dias*H1_teo:.4f})")
    linhas7 = []
    for i in range(n_dias):
        t = H1_teo if i == 0 else Hc_teo
        linhas7.append([str(i + 1), f"{cond_passado[i]:.4f}", f"{cond_ontem[i]:.4f}", f"{t:.4f}"])
    linhas7.append(["Soma", f"{cond_passado.sum():.4f}", f"{cond_ontem.sum():.4f}", f"{H_seq_teo:.4f}"])
    linhas7.append(["$n\\,H(T_1)$ (ingênuo)", f"{ingenuo:.4f}", "--", f"{n_dias*H1_teo:.4f}"])
    salvar_tabela("tab_t1_passo7.tex",
                  ["Dia $i$", "$\\hat H(T_i|T^{i-1})$", "$\\hat H(T_i|T_{i-1})$", "Teoria"], linhas7)

    plt.figure(figsize=(6.5, 4.3))
    dias = np.arange(1, n_dias + 1)
    plt.plot(dias, cond_passado, "o-", label="$\\hat H(T_i\\,|\\,T_1..T_{i-1})$")
    plt.plot(dias, cond_ontem, "x--", ms=9, label="$\\hat H(T_i\\,|\\,T_{i-1})$")
    plt.axhline(H1_teo, color="gray", ls=":", label=f"$H(T_1)$ = {H1_teo:.3f} (ingênuo)")
    plt.axhline(Hc_teo, color="red", ls="--", label=f"$H(T_i|T_{{i-1}})$ = {Hc_teo:.3f}")
    plt.xlabel("dia $i$"); plt.ylabel("bits"); plt.ylim(0, 1.05)
    plt.title("Passo 7: regra da cadeia, dia a dia"); plt.legend(fontsize=8)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(PASTA, "fig_t1_passo7.png"), dpi=150); plt.close()

    # ---------------- Passo 8: economia (um unico paciente, historico longo) ----------------
    serie = simula_markov(100_000, 1, Pm, pi, rng)[0]
    tab_viz = tabela_contagens(serie[:-1], serie[1:])
    H1_s = entropia(np.bincount(serie, minlength=2))
    Hc_s = entropia(tab_viz) - entropia(tab_viz.sum(axis=1))
    Iviz_s = info_mutua(tab_viz)
    econ = ingenuo - H_seq
    print("\n== Passo 8 ==")
    print(f"economia (n={n_dias}): {econ:.4f} bits; (n-1) I = {(n_dias-1)*Iviz_teo:.4f}")
    print(f"paciente unico (1e5 dias): H(T) {H1_s:.4f}, H(T_i|T_i-1) {Hc_s:.4f}, I {Iviz_s:.4f}")
    salvar_tabela("tab_t1_passo8.tex", ["", "Estimado", "Teórico"],
                  [[f"Custo ingênuo, $n={n_dias}$ dias", f"{ingenuo:.4f}", f"{n_dias*H1_teo:.4f}"],
                   ["Custo condicional $H(T_1,\\dots,T_n)$", f"{H_seq:.4f}", f"{H_seq_teo:.4f}"],
                   ["Economia", f"{econ:.4f}", f"{(n_dias-1)*Iviz_teo:.4f}"],
                   ["$(n-1)\\,I(T_i;T_{i-1})$", f"{(n_dias-1)*Iviz_s:.4f}", f"{(n_dias-1)*Iviz_teo:.4f}"],
                   ["Economia por dia (1 paciente, $10^5$ dias)", f"{H1_s-Hc_s:.4f}", f"{Iviz_teo:.4f}"]])

    # ---------------- Passo 9: teste de independencia ----------------
    serie_curta = simula_markov(2000, 1, Pm, pi, rng)[0]
    tab_curta = tabela_contagens(serie_curta[:-1], serie_curta[1:])
    linhas9 = []
    print("\n== Passo 9 ==")
    for nome, tab in [("$D$ vs $T$", tab_emp), ("$T_i$ vs $T_{i-1}$", tab_curta)]:
        G, gl, pv, I = teste_independencia(tab)
        # conferencia: razao de verossimilhanca, SEM correcao de Yates (padrao em 2x2!)
        G_sc, pv_sc, gl_sc, _ = chi2_contingency(tab, correction=False, lambda_="log-likelihood")
        dec = "rejeita $H_0$" if pv < 0.05 else "não rejeita $H_0$"
        print(f"{nome}: N={tab.sum()} I={I:.4f} G={G:.2f} gl={gl} p={pv:.3g} | scipy G={G_sc:.2f}")
        linhas9.append([nome, str(tab.sum()), f"{I:.4f}", f"{G:.2f}", f"{G_sc:.2f}",
                        str(gl), pv_tex(pv), dec])
    salvar_tabela("tab_t1_passo9.tex",
                  ["Teste", "$N$", "$\\hat I$", "$G$", "$G$ (scipy)", "g.l.", "valor $p$", "Decisão"],
                  linhas9)

    # ---------------- Passo 10: tres cenarios, muitas repeticoes ----------------
    cenarios = [("Forte", 0.05, 0.20), ("Fraca", 0.40, 0.50), ("Independente", 0.30, 0.70)]
    Ns10, REP10, alfa = [100, 500, 2000], 1000, 0.05
    linhas10, taxas = [], {}
    print(f"\n== Passo 10 ({REP10} repeticoes, alfa = {alfa}) ==")
    for nome, fp, fn in cenarios:
        I_c = info_mutua(modelo_conjunto(pD, fp, fn))
        taxas[nome] = []
        for n in Ns10:
            rej, pequenas = 0, 0
            for _ in range(REP10):
                tab = tabela_contagens(*simula_DT(n, pD, fp, fn, rng))
                _, _, pv, _ = teste_independencia(tab)
                rej += pv < alfa
                esperado = np.outer(tab.sum(1), tab.sum(0)) / n
                pequenas += esperado.min() < 5
            taxas[nome].append(rej / REP10)
            print(f"{nome:<13} I={I_c:.5f} N={n:>5}: rejeicoes {rej/REP10:.3f}"
                  f" | tabelas com esperado<5: {pequenas/REP10:.3f}")
            linhas10.append([nome, num(I_c), str(n), f"{100*rej/REP10:.1f}\\%",
                             f"{100*pequenas/REP10:.1f}\\%"])
    salvar_tabela("tab_t1_passo10.tex",
                  ["Cenário", "$I(D;T)$", "$N$", "Rejeições de $H_0$", "Tabelas c/ esperado $<5$"],
                  linhas10)

    plt.figure(figsize=(6.5, 4.3))
    larg = 0.25
    for j, (nome, _, _) in enumerate(cenarios):
        plt.bar(np.arange(len(Ns10)) + (j - 1) * larg, taxas[nome], larg, label=nome)
    plt.axhline(alfa, color="red", ls="--", label=f"$\\alpha$ = {alfa}")
    plt.xticks(range(len(Ns10)), [f"N = {n}" for n in Ns10])
    plt.ylabel("fração de rejeições de $H_0$"); plt.title("Passo 10: desempenho do teste")
    plt.legend(fontsize=8); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(PASTA, "fig_t1_passo10.png"), dpi=150); plt.close()

    print("\nArquivos gerados em", PASTA + "/")