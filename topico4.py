"""
Topico 4 - Capacidade de Canal e Codigos Corretores de Erro
Teoria da Informacao - Primeira atividade em grupo

Uso:     python topico4.py
Requer:  numpy, scipy, matplotlib
Saida:   pasta resultados/ com figuras (.png) e tabelas (.tex) usadas no relatorio
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom

PASTA = "resultados"
os.makedirs(PASTA, exist_ok=True)
rng = np.random.default_rng(2026)          # semente fixa: resultados reprodutiveis

# =====================================================================
# Funcoes basicas
# =====================================================================
def Hb(p):
    """Entropia binaria em bits (com 0 log 0 = 0)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def capacidade(p):
    """Capacidade do canal binario simetrico: C = 1 - Hb(p)."""
    return 1.0 - Hb(p)

def canal_bsc(bits, p):
    """Canal binario simetrico, bit a bit:
    sorteia um uniforme por bit; se < p, o bit e invertido (XOR com 1)."""
    erros = (rng.random(bits.size) < p).astype(np.int8)
    return bits ^ erros

# ---------------- Codigo de repeticao ----------------
def repeticao_transmite(bits, nr, p):
    enviados = np.repeat(bits, nr)                        # cada bit vira nr copias
    recebidos = canal_bsc(enviados, p).reshape(-1, nr)    # uma linha por bit original
    return (recebidos.sum(axis=1) > nr // 2).astype(np.int8)   # votacao majoritaria

def erro_teorico_repeticao(nr, p):
    """Erro se a maioria das nr copias for invertida: P(X >= nr//2 + 1), X ~ Bin(nr, p)."""
    return binom.sf(nr // 2, nr, p)

# ---------------- Hamming (7,4) com matrizes ----------------
# Convencao: posicoes 1..7 do bloco = [p1 p2 d1 p4 d2 d3 d4]
G = np.array([[1, 1, 1, 0, 0, 0, 0],      # d1
              [1, 0, 0, 1, 1, 0, 0],      # d2
              [0, 1, 0, 1, 0, 1, 0],      # d3
              [1, 1, 0, 1, 0, 0, 1]],     # d4
             dtype=np.int8)
H = np.array([[1, 0, 1, 0, 1, 0, 1],      # s1 vigia posicoes 1,3,5,7
              [0, 1, 1, 0, 0, 1, 1],      # s2 vigia posicoes 2,3,6,7
              [0, 0, 0, 1, 1, 1, 1]],     # s4 vigia posicoes 4,5,6,7
             dtype=np.int8)
POS_DADOS = [2, 4, 5, 6]                  # indices (base 0) de d1, d2, d3, d4
assert not ((G.astype(int) @ H.T) % 2).any(), "G e H incompativeis"

def hamming_codifica(blocos4):
    return ((blocos4.astype(int) @ G) % 2).astype(np.int8)

def hamming_decodifica(r):
    s = (r.astype(int) @ H.T) % 2                 # sindrome (s1, s2, s4)
    pos = s @ np.array([1, 2, 4])                 # = posicao do erro (1..7); 0 = sem erro
    r = r.copy()
    linhas = np.nonzero(pos)[0]
    r[linhas, pos[linhas] - 1] ^= 1               # desvira o bit apontado
    return r[:, POS_DADOS], pos

def hamming_transmite(bits, p):
    blocos = bits.reshape(-1, 4)
    recebidos = canal_bsc(hamming_codifica(blocos).ravel(), p).reshape(-1, 7)
    dec, _ = hamming_decodifica(recebidos)
    return dec.ravel()

# ---------------- Utilitarios para o relatorio ----------------
def num(x):
    """Formata numero para LaTeX."""
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

# =====================================================================
# Teste manual do Hamming (antes da simulacao grande)
# =====================================================================
print("== Teste manual do Hamming ==")
d = np.array([[1, 0, 1, 1]], dtype=np.int8)
c = hamming_codifica(d)
e = c.copy(); e[0, 4] ^= 1                         # estraga a posicao 5
rd, pos = hamming_decodifica(e)
print("dados        :", d[0])
print("bloco enviado:", c[0])
print("bloco errado :", e[0], "(posicao 5 invertida)")
print("sindrome     : posicao", pos[0])
print("recuperado   :", rd[0], "->", "OK" if np.array_equal(rd, d) else "FALHOU")

teste_ok = True
for m in range(16):                                # todas as 16 mensagens
    d = np.array([[(m >> k) & 1 for k in range(4)]], dtype=np.int8)
    c = hamming_codifica(d)
    teste_ok = teste_ok and np.array_equal(hamming_decodifica(c)[0], d)
    for j in range(7):                             # erro em cada uma das 7 posicoes
        e = c.copy(); e[0, j] ^= 1
        rd, pos = hamming_decodifica(e)
        teste_ok = teste_ok and np.array_equal(rd, d) and pos[0] == j + 1
print("Teste exaustivo (16 mensagens x 7 posicoes):", "OK" if teste_ok else "FALHOU")

# =====================================================================
# Passos 1 a 7  (p = 0,1)
# =====================================================================
p = 0.10
C = capacidade(p)
print(f"\nPassos 1-2: p = {p:.2f} -> C = 1 - Hb(p) = {C:.4f} bits por uso do canal")

N = 400_000                                        # >> 100/p; multiplo de 4
foto = rng.integers(0, 2, N, dtype=np.int8)        # Passo 3

res7 = []                                          # (nome, R, erro medido, erro teorico)
res7.append(("Sem código", 1.0, np.mean(canal_bsc(foto, p) != foto), p))       # Passo 4
for nr in (3, 5):                                                                  # Passo 5
    dec = repeticao_transmite(foto, nr, p)
    res7.append((f"Repetição {nr}x", 1 / nr, np.mean(dec != foto),
                 erro_teorico_repeticao(nr, p)))
dec_h = hamming_transmite(foto, p)                                                 # Passo 6
erro_bit_h = np.mean(dec_h != foto)
erro_bloco_h = np.mean((dec_h.reshape(-1, 4) != foto.reshape(-1, 4)).any(axis=1))
teo_bloco_h = binom.sf(1, 7, p)                    # >= 2 erros no bloco de 7
res7.append(("Hamming (7,4)", 4 / 7, erro_bit_h, None))

print(f"\nPasso 7 (p = {p}, {N} bits)")
print(f"{'Metodo':<16}{'R':>7}{'Erro medido':>13}{'Erro teorico':>14}{'R<C?':>6}")
for nome, R, em, et in res7:
    print(f"{nome:<16}{R:>7.3f}{em:>13.5f}{('-' if et is None else f'{et:.5f}'):>14}"
          f"{('sim' if R < C else 'nao'):>6}")
print(f"Hamming, erro de bloco: medido {erro_bloco_h:.5f} | teorico {teo_bloco_h:.5f}")

salvar_tabela("tab_passo7.tex",
              ["Método", "$R$", "Erro de bit (medido)", "Erro de bit (teoria)", "$R<C$?"],
              [[n, f"{R:.3f}", num(em), num(et), "sim" if R < C else "não"]
               for n, R, em, et in res7]
              + [["Hamming, erro de \\emph{bloco}", f"{4/7:.3f}", num(erro_bloco_h),
                  num(teo_bloco_h), "não" if 4 / 7 > C else "sim"]])

# ---------- Figura ilustrativa: uma "foto" de Marte ----------
L = 100
yy, xx = np.mgrid[0:L, 0:L]
img = (((xx - 50) ** 2 + (yy - 50) ** 2) < 35 ** 2).astype(np.int8)       # planeta
img[((xx - 62) ** 2 + (yy - 40) ** 2) < 8 ** 2] = 0                       # cratera
bits_img = img.ravel()                                                    # 10 000 bits
versoes = [("Original", bits_img),
           ("Sem código", canal_bsc(bits_img, p)),
           ("Repetição 3x", repeticao_transmite(bits_img, 3, p)),
           ("Hamming (7,4)", hamming_transmite(bits_img, p))]
fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
for a, (t, b) in zip(ax, versoes):
    a.imshow(b.reshape(L, L), cmap="gray", interpolation="nearest")
    a.set_title(f"{t}\nerro = {100*np.mean(b != bits_img):.1f}%")
    a.axis("off")
plt.suptitle(f"Foto de 10 000 bits enviada por um canal com p = {p}")
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "fig_foto.png"), dpi=150)
plt.close()

# =====================================================================
# Passo 8: varios valores de p
# =====================================================================
ps = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
N8 = 1_000_000
curvas = {"Sem código": [], "Repetição 3x": [], "Hamming (7,4)": []}
for pp in ps:
    x = rng.integers(0, 2, N8, dtype=np.int8)
    curvas["Sem código"].append(np.mean(canal_bsc(x, pp) != x))
    curvas["Repetição 3x"].append(np.mean(repeticao_transmite(x, 3, pp) != x))
    curvas["Hamming (7,4)"].append(np.mean(hamming_transmite(x, pp) != x))

print(f"\nPasso 8 ({N8} bits por ponto)")
print(f"{'p':>6}{'C':>8}{'Sem':>10}{'Rep3':>10}{'Hamming':>10}")
for i, pp in enumerate(ps):
    print(f"{pp:>6.2f}{capacidade(pp):>8.3f}{curvas['Sem código'][i]:>10.5f}"
          f"{curvas['Repetição 3x'][i]:>10.5f}{curvas['Hamming (7,4)'][i]:>10.5f}")

plt.figure(figsize=(6.5, 4.5))
for nome, v in curvas.items():
    v = np.array(v); ok = v > 0
    plt.semilogy(np.array(ps)[ok], v[ok], "o-", label=nome)
plt.xlabel("probabilidade de inversão $p$")
plt.ylabel("taxa de erro de bit (escala log)")
plt.title("Passo 8: erro de bit em função do ruído")
plt.grid(True, which="both", alpha=0.3); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "fig_passo8.png"), dpi=150)
plt.close()

# =====================================================================
# Passo 9: repeticao cada vez mais longa (p = 0,1)
# =====================================================================
nrs = [1, 3, 5, 7, 9, 11, 13]
N9 = 1_000_000
res9 = []                                          # (nr, R, erro medido, teoria, n_erros)
for nr in nrs:
    x = rng.integers(0, 2, N9, dtype=np.int8)
    dec = repeticao_transmite(x, nr, p)
    n_err = int(np.sum(dec != x))
    res9.append((nr, 1 / nr, n_err / N9, erro_teorico_repeticao(nr, p), n_err))

print(f"\nPasso 9 (p = {p}, C = {C:.3f}, {N9} bits por nr)")
print(f"{'nr':>4}{'R':>8}{'Erro medido':>13}{'Teoria':>12}{'n erros':>9}")
for nr, R, em, et, ne in res9:
    aviso = "  (poucos eventos!)" if ne < 30 else ""
    print(f"{nr:>4}{R:>8.3f}{em:>13.6f}{et:>12.6f}{ne:>9}{aviso}")

salvar_tabela("tab_passo9.tex",
              ["$n_r$", "$R=1/n_r$", "Erro (medido)", "Erro (binomial)", "Nº de erros"],
              [[str(nr), f"{R:.3f}", num(em), num(et), str(ne)]
               for nr, R, em, et, ne in res9])

plt.figure(figsize=(6.5, 4.5))
R9 = np.array([r[1] for r in res9]); e9 = np.array([r[2] for r in res9])
t9 = np.array([r[3] for r in res9]); ok = e9 > 0
plt.semilogy(R9[ok], e9[ok], "o", ms=8, label="repetição (simulado)")
plt.semilogy(R9, t9, "-", label="repetição (binomial)")
plt.semilogy([4 / 7], [erro_bit_h], "s", ms=9, label="Hamming (7,4) (simulado)")
for nr, R, _, et, _ in res9:
    plt.annotate(f"$n_r$={nr}", (R, et), textcoords="offset points", xytext=(4, 6), fontsize=8)
plt.axvline(C, color="red", ls="--", label=f"capacidade $C$ = {C:.3f}")
plt.axhline(p, color="gray", ls=":", label=f"sem código (erro = $p$ = {p})")
plt.xlabel("taxa $R$ (bits úteis por bit enviado)")
plt.ylabel("taxa de erro de bit (escala log)")
plt.title(f"Passo 9: erro × taxa ($p$ = {p})")
plt.grid(True, which="both", alpha=0.3); plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "fig_passo9.png"), dpi=150)
plt.close()

# =====================================================================
# Passo 10: tentando transmitir acima da capacidade (p = 0,2)
# =====================================================================
p10 = 0.20
C10 = capacidade(p10)
N10 = 400_000                                       # multiplo de 4 e de 100
TAM_MSG = 100                                       # "mensagem" de 100 bits
esquemas = [("Sem código",    1.0,    lambda x: canal_bsc(x, p10)),
            ("Hamming (7,4)", 4 / 7,  lambda x: hamming_transmite(x, p10)),
            ("Repetição 3x",  1 / 3,  lambda x: repeticao_transmite(x, 3, p10)),
            ("Repetição 5x",  1 / 5,  lambda x: repeticao_transmite(x, 5, p10)),
            ("Repetição 9x",  1 / 9,  lambda x: repeticao_transmite(x, 9, p10)),
            ("Repetição 15x", 1 / 15, lambda x: repeticao_transmite(x, 15, p10))]
res10 = []
for nome, R, f in esquemas:
    x = rng.integers(0, 2, N10, dtype=np.int8)
    errado = f(x) != x
    msg_ok = np.mean(~errado.reshape(-1, TAM_MSG).any(axis=1))
    res10.append((nome, R, np.mean(errado), msg_ok))

print(f"\nPasso 10 (p = {p10}, C = {C10:.3f})")
print(f"{'Metodo':<16}{'R':>7}{'R>C?':>6}{'Erro de bit':>13}{'Msg 100 bits ok':>17}")
for nome, R, eb, mo in res10:
    print(f"{nome:<16}{R:>7.3f}{('sim' if R > C10 else 'nao'):>6}{eb:>13.5f}{mo:>17.4f}")

salvar_tabela("tab_passo10.tex",
              ["Método", "$R$", "$R>C$?", "Erro de bit", "Mensagens de 100 bits sem erro"],
              [[n, f"{R:.3f}", "sim" if R > C10 else "não", num(eb), f"{100*mo:.1f}\\%"]
               for n, R, eb, mo in res10])

plt.figure(figsize=(6.5, 4.5))
for nome, R, eb, _ in res10:
    plt.semilogy([R], [eb], "o", ms=8)
    plt.annotate(nome, (R, eb), textcoords="offset points", xytext=(5, 5), fontsize=8)
plt.axvspan(C10, 1.05, color="red", alpha=0.12, label="$R>C$: erro não pode ir a zero")
plt.axvline(C10, color="red", ls="--", label=f"$C$ = {C10:.3f}")
plt.axhline(p10, color="gray", ls=":", label=f"sem código ($p$ = {p10})")
plt.xlim(0, 1.05)
plt.xlabel("taxa $R$"); plt.ylabel("taxa de erro de bit (escala log)")
plt.title(f"Passo 10: abaixo e acima da capacidade ($p$ = {p10})")
plt.grid(True, which="both", alpha=0.3); plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "fig_passo10.png"), dpi=150)
plt.close()

print("\nArquivos gerados em", PASTA + "/:", sorted(os.listdir(PASTA)))