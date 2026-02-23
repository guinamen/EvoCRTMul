"""
EvoCRTMul v2.0 — Gerador de Fronteira de Pareto para Bases RNS
===============================================================
Execução:
    python pareto_rns.py --bits 8
    python pareto_rns.py --bits 8 --prime-bound 31 --max-channels 5
    python pareto_rns.py --bits 8 --output resultados.json

Saída padrão: pareto_front_N<bits>.json
"""

import argparse
import json
import math
import itertools
from dataclasses import dataclass, asdict
from sympy import isprime, factorint


# ─────────────────────────────────────────────────────────────────────────────
# 1. ENUMERAÇÃO DE MÓDULOS CANDIDATOS
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_candidate_moduli(prime_bound: int) -> list[int]:
    """
    Retorna todos os módulos válidos dentro do Prime Bound:
    primos e potências de primos (incluindo 2^n) até prime_bound.

    Exemplos para prime_bound=31:
        [2, 3, 4, 5, 7, 8, 9, 16, 17, 19, 23, 25, 27, 29, 31]
    """
    candidates = set()
    for n in range(2, prime_bound + 1):
        factors = factorint(n)
        # Aceita primos e potências puras de um único primo
        if len(factors) == 1:
            candidates.add(n)
    return sorted(candidates)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FILTROS DUROS (ETAPA 1)
# ─────────────────────────────────────────────────────────────────────────────

def are_pairwise_coprime(moduli: tuple[int, ...]) -> bool:
    """Verifica coprimidade total entre todos os pares."""
    from math import gcd
    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            if gcd(moduli[i], moduli[j]) != 1:
                return False
    return True


def product_covers_range(moduli: tuple[int, ...], N: int) -> bool:
    """M = PROD(mi) >= 2^(2N) — condição de não-overflow."""
    M = 1
    for m in moduli:
        M *= m
    return M >= (1 << (2 * N))


def mandatory_power_of_two(moduli: tuple[int, ...], prime_bound: int) -> bool:
    """
    Regra do Hardware Livre: o conjunto DEVE incluir a maior
    potência de 2 disponível dentro do prime_bound.
    """
    largest_pow2 = 1
    p = 2
    while p <= prime_bound:
        largest_pow2 = p
        p *= 2
    return largest_pow2 in moduli


def passes_hard_filters(moduli: tuple[int, ...], N: int, prime_bound: int) -> bool:
    return (
        are_pairwise_coprime(moduli)
        and product_covers_range(moduli, N)
        and mandatory_power_of_two(moduli, prime_bound)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLASSIFICAÇÃO DE MÓDULOS (TAXONOMIA DE CUSTO)
# ─────────────────────────────────────────────────────────────────────────────

def modulus_class(m: int) -> str:
    """
    Classe A — 2^n          : custo O(1), truncamento de fios
    Classe B — 2^n ± 1      : custo O(n), soma com carry parcial
    Classe C — primo arbitrário: custo O(n²), divisor lógico completo
    """
    # Classe A: potência de 2
    if m & (m - 1) == 0:
        return "A"
    # Classe B: 2^n - 1 ou 2^n + 1
    n = m.bit_length()
    if m == (1 << n) - 1 or m == (1 << (n - 1)) + 1:
        return "B"
    return "C"


def carry_cascade_factor(m: int) -> int:
    """
    P_cascata: multiplicador de cascatas de carry para estimativa de delay.
        Classe A -> 0  (nenhuma porta)
        Classe B -> 1  (soma parcial)
        Classe C -> 2  (redução modular complexa)
    """
    cls = modulus_class(m)
    return {"A": 0, "B": 1, "C": 2}[cls]


# ─────────────────────────────────────────────────────────────────────────────
# 4. CÁLCULO DAS QUATRO DIMENSÕES (ETAPA 2)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SetMetrics:
    moduli:        tuple
    M:             int      # produto total
    N:             int      # largura dos operandos

    # Dimensão 1 — Eficiência de Canal
    d1_waste:      int      # ceil(log2(M)) - 2N  →  minimizar

    # Dimensão 2 — Complexidade de Roteamento
    d2_channels:   int      # k = len(moduli)  →  minimizar

    # Dimensão 3 — Desbalanceamento de Pipeline
    d3_delta_delay: float   # max(W*P) - min(W*P)  →  minimizar

    # Dimensão 4 — Custo do Conversor Reverso
    d4_converter:  float    # ClasseMax_score + log2(M)  →  minimizar

    # Metadados auxiliares
    classes:       tuple    # classe (A/B/C) de cada módulo
    evolvability:  int      # 2^(2 * W_max_nao_especial) — estimativa CGP


def compute_metrics(moduli: tuple[int, ...], N: int) -> SetMetrics:
    M = 1
    for m in moduli:
        M *= m

    log2_M = math.log2(M)

    # ── Dimensão 1 ──────────────────────────────────────────────────────────
    d1_waste = math.ceil(log2_M) - 2 * N

    # ── Dimensão 2 ──────────────────────────────────────────────────────────
    d2_channels = len(moduli)

    # ── Dimensão 3 ──────────────────────────────────────────────────────────
    # delay_estimado(mi) = W(mi) * P_cascata(mi)
    delays = [m.bit_length() * carry_cascade_factor(m) for m in moduli]
    d3_delta_delay = float(max(delays) - min(delays))

    # ── Dimensão 4 ──────────────────────────────────────────────────────────
    # Mapeia classes para score numérico: A=0, B=1, C=2
    class_score = {"A": 0, "B": 1, "C": 2}
    classes = tuple(modulus_class(m) for m in moduli)
    max_class_score = max(class_score[c] for c in classes)
    # Combinação: parte discreta (classe) normalizada + parte contínua (log2M)
    d4_converter = float(max_class_score) + log2_M

    # ── Evolvabilidade ───────────────────────────────────────────────────────
    # Tabela-verdade do maior módulo não-trivial (não Classe A)
    non_trivial = [m for m in moduli if modulus_class(m) != "A"]
    if non_trivial:
        w_max = max(m.bit_length() for m in non_trivial)
    else:
        w_max = max(m.bit_length() for m in moduli)
    evolvability = 1 << (2 * w_max)  # 2^(2 * W_max)

    return SetMetrics(
        moduli=moduli,
        M=M,
        N=N,
        d1_waste=d1_waste,
        d2_channels=d2_channels,
        d3_delta_delay=d3_delta_delay,
        d4_converter=d4_converter,
        classes=classes,
        evolvability=evolvability,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. DOMINÂNCIA DE PARETO (ETAPA 3)
# ─────────────────────────────────────────────────────────────────────────────

def dominates(a: SetMetrics, b: SetMetrics) -> bool:
    """
    Retorna True se 'a' domina 'b':
    'a' é melhor ou igual em TODAS as dimensões e
    estritamente melhor em PELO MENOS UMA.
    Todas as dimensões são de minimização.
    """
    dims_a = (a.d1_waste, a.d2_channels, a.d3_delta_delay, a.d4_converter)
    dims_b = (b.d1_waste, b.d2_channels, b.d3_delta_delay, b.d4_converter)

    better_or_equal = all(da <= db for da, db in zip(dims_a, dims_b))
    strictly_better = any(da <  db for da, db in zip(dims_a, dims_b))

    return better_or_equal and strictly_better


def pareto_front(population: list[SetMetrics]) -> list[SetMetrics]:
    """
    Retorna apenas os conjuntos não-dominados (fronteira de Pareto).
    Complexidade: O(n² * d), onde d=4 é o número de dimensões.
    """
    front = []
    for candidate in population:
        dominated = False
        for other in population:
            if other is not candidate and dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


# ─────────────────────────────────────────────────────────────────────────────
# 6. PRUNING POR EVOLVABILIDADE (ETAPA 4)
# ─────────────────────────────────────────────────────────────────────────────

def prune_by_evolvability(
    front: list[SetMetrics],
    max_truth_table: int = 2 ** 20,   # padrão: tabela-verdade de até ~1M linhas
) -> tuple[list[SetMetrics], list[SetMetrics]]:
    """
    Separa os conjuntos da fronteira de Pareto em:
    - evolvable   : tabela-verdade do maior módulo não-trivial <= max_truth_table
    - intractable : acima do limiar, descartados para o motor CGP

    O limiar padrão (2^20) é conservador e deve ser calibrado empiricamente
    conforme o hardware de execução.
    """
    evolvable   = [s for s in front if s.evolvability <= max_truth_table]
    intractable = [s for s in front if s.evolvability >  max_truth_table]
    return evolvable, intractable


# ─────────────────────────────────────────────────────────────────────────────
# 7. SERIALIZAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def metrics_to_dict(s: SetMetrics) -> dict:
    return {
        "moduli":         list(s.moduli),
        "M":              s.M,
        "N":              s.N,
        "dimensions": {
            "d1_waste":       s.d1_waste,
            "d2_channels":    s.d2_channels,
            "d3_delta_delay": s.d3_delta_delay,
            "d4_converter":   round(s.d4_converter, 4),
        },
        "module_classes": dict(zip(s.moduli, s.classes)),
        "evolvability_tt_size": s.evolvability,
    }


def export_results(
    N:           int,
    pareto:      list[SetMetrics],
    intractable: list[SetMetrics],
    total_candidates: int,
    total_filtered:   int,
    output_path: str,
) -> None:
    payload = {
        "config": {
            "N_bits": N,
            "minimum_M": 1 << (2 * N),
            "dimension_descriptions": {
                "d1_waste":       "ceil(log2(M)) - 2N  [minimizar — overhead de representação]",
                "d2_channels":    "k = número de canais RNS  [minimizar — complexidade de roteamento]",
                "d3_delta_delay": "max(W*P) - min(W*P)  [minimizar — desbalanceamento de pipeline]",
                "d4_converter":   "ClasseMax_score + log2(M)  [minimizar — custo do conversor reverso]",
            },
        },
        "statistics": {
            "total_enumerated":          total_candidates,
            "after_hard_filters":        total_filtered,
            "pareto_front_size":         len(pareto),
            "intractable_pruned":        len(intractable),
            "ready_for_evolution":       len(pareto),
        },
        "pareto_front": [metrics_to_dict(s) for s in pareto],
        "intractable_sets": [metrics_to_dict(s) for s in intractable],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# 8. RELATÓRIO DE CONSOLE
# ─────────────────────────────────────────────────────────────────────────────

def print_report(
    N:           int,
    pareto:      list[SetMetrics],
    intractable: list[SetMetrics],
    total_candidates: int,
    total_filtered:   int,
) -> None:
    sep = "─" * 72

    print(f"\n{'═'*72}")
    print(f"  EvoCRTMul v2.0 — Fronteira de Pareto  |  N = {N} bits")
    print(f"{'═'*72}")
    print(f"  Módulos enumerados (Prime Bound) : {total_candidates:>6}")
    print(f"  Conjuntos após filtros duros     : {total_filtered:>6}")
    print(f"  Conjuntos na fronteira de Pareto : {len(pareto):>6}")
    print(f"  Descartados (evolvabilidade)     : {len(intractable):>6}")
    print(sep)

    if not pareto:
        print("  [!] Nenhum conjunto sobreviveu aos filtros.")
        return

    # Cabeçalho
    print(f"  {'Módulos':<28} {'D1':>4} {'D2':>4} {'D3':>6} {'D4':>8}  {'Classes'}")
    print(sep)

    # Ordena por D1 para leitura
    sorted_pareto = sorted(pareto, key=lambda s: (s.d1_waste, s.d2_channels))

    for s in sorted_pareto:
        moduli_str = "{" + ", ".join(str(m) for m in s.moduli) + "}"
        classes_str = " ".join(
            f"{m}:{c}" for m, c in zip(s.moduli, s.classes)
        )
        print(
            f"  {moduli_str:<28}"
            f" {s.d1_waste:>4}"
            f" {s.d2_channels:>4}"
            f" {s.d3_delta_delay:>6.1f}"
            f" {s.d4_converter:>8.2f}"
            f"  {classes_str}"
        )

    print(sep)
    print("  Legenda dimensões:")
    print("    D1 = Desperdício de representação (bits extras acima de 2N)")
    print("    D2 = Número de canais k")
    print("    D3 = Desbalanceamento de pipeline (delay máx - delay mín)")
    print("    D4 = Custo do conversor reverso (ClasseMax + log2 M)")
    print()

    if intractable:
        print(f"  [!] {len(intractable)} conjunto(s) na fronteira de Pareto foram marcados como")
        print( "      intratáveis para o motor CGP (tabela-verdade > limiar).")
        print( "      Estão incluídos em 'intractable_sets' no JSON exportado.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 9. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    N:              int,
    prime_bound:    int,
    max_channels:   int,
    max_tt:         int,
    output_path:    str,
) -> None:

    # ── Etapa 0: Enumeração ──────────────────────────────────────────────────
    moduli_pool = enumerate_candidate_moduli(prime_bound)
    print(f"\n[0] Pool de módulos candidatos (prime_bound={prime_bound}):")
    print(f"    {moduli_pool}")

    # ── Etapa 1: Filtros Duros ───────────────────────────────────────────────
    print(f"\n[1] Aplicando filtros duros para N={N}...")
    all_sets = []
    for k in range(2, max_channels + 1):
        for combo in itertools.combinations(moduli_pool, k):
            if passes_hard_filters(combo, N, prime_bound):
                all_sets.append(combo)

    total_enumerated = sum(
        math.comb(len(moduli_pool), k) for k in range(2, max_channels + 1)
    )
    print(f"    {total_enumerated} combinações verificadas → {len(all_sets)} passaram nos filtros duros.")

    if not all_sets:
        print("\n[!] Nenhum conjunto válido encontrado. "
              "Tente aumentar --prime-bound ou --max-channels.")
        return

    # ── Etapa 2: Cálculo das Dimensões ───────────────────────────────────────
    print(f"\n[2] Calculando métricas para {len(all_sets)} conjuntos...")
    population = [compute_metrics(s, N) for s in all_sets]

    # ── Etapa 3: Dominância de Pareto ────────────────────────────────────────
    print(f"\n[3] Calculando fronteira de Pareto...")
    front = pareto_front(population)
    print(f"    {len(front)} conjuntos não-dominados identificados.")

    # ── Etapa 4: Pruning por Evolvabilidade ──────────────────────────────────
    print(f"\n[4] Pruning por evolvabilidade (limiar de tabela-verdade = 2^{int(math.log2(max_tt))})...")
    evolvable, intractable = prune_by_evolvability(front, max_truth_table=max_tt)
    print(f"    {len(evolvable)} prontos para evolução, {len(intractable)} descartados.")

    # ── Relatório e Exportação ───────────────────────────────────────────────
    print_report(N, evolvable, intractable, total_enumerated, len(all_sets))
    export_results(N, evolvable, intractable, total_enumerated, len(all_sets), output_path)
    print(f"[✓] Resultados exportados para: {output_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 10. CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EvoCRTMul v2.0 — Gerador de Fronteira de Pareto para Bases RNS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bits", "-N", type=int, required=True,
        help="Largura N dos operandos de entrada (ex: 8 para um multiplicador 8x8)."
    )
    parser.add_argument(
        "--prime-bound", type=int, default=None,
        help="Valor máximo permitido para módulos candidatos. "
             "Padrão: 2^(N/2 + 1) - 1  (ex: N=8 → 31)."
    )
    parser.add_argument(
        "--max-channels", type=int, default=5,
        help="Número máximo de canais k em um conjunto (D2 <= max_channels)."
    )
    parser.add_argument(
        "--max-tt", type=int, default=20,
        help="Limiar de evolvabilidade em log2 (ex: 20 significa 2^20 linhas)."
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Caminho do arquivo JSON de saída. "
             "Padrão: pareto_front_N<bits>.json"
    )

    args = parser.parse_args()

    N           = args.bits
    prime_bound = args.prime_bound or ((1 << (N // 2 + 1)) - 1)
    max_channels = args.max_channels
    max_tt      = 1 << args.max_tt
    output_path = args.output or f"pareto_front_N{N}.json"

    run_pipeline(
        N=N,
        prime_bound=prime_bound,
        max_channels=max_channels,
        max_tt=max_tt,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
