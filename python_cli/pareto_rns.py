"""
EvoCRTMul v2.0 — Gerador de Fronteira de Pareto para Bases RNS (Multi-Processo)
================================================================================
Usa ProcessPoolExecutor com um worker por núcleo lógico de CPU para paralelizar
as etapas CPU-bound do pipeline:

  • Etapa 1 (Filtros Duros)  : combinações particionadas por k entre os workers.
  • Etapa 2 (Métricas)       : cálculo das 4 dimensões em chunks paralelos.
  • Etapa 3 (Pareto)         : redução incremental paralela com merge final.

NOTA sobre o GIL:
    threading.Thread não traz ganho para código CPU-bound em CPython devido ao
    Global Interpreter Lock (GIL). A solução correta é multiprocessing, que
    spawna processos independentes com heap separada, contornando o GIL
    completamente. ProcessPoolExecutor é a API de alto nível recomendada.

Execução:
    python pareto_rns_parallel.py --bits 8
    python pareto_rns_parallel.py --bits 16 --prime-bound 255 --workers 8
    python pareto_rns_parallel.py --bits 8 --prime-bound 63 --output resultado.json
"""

import argparse
import json
import math
import os
import time
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from math import gcd
from sympy import factorint


# ─────────────────────────────────────────────────────────────────────────────
# 1. ENUMERAÇÃO DE MÓDULOS CANDIDATOS
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_candidate_moduli(prime_bound: int) -> list[int]:
    """
    Retorna primos e potências puras de um único primo até prime_bound.
    Executado no processo principal — é rápido e não justifica paralelização.
    """
    candidates = set()
    for n in range(2, prime_bound + 1):
        if len(factorint(n)) == 1:
            candidates.add(n)
    return sorted(candidates)


def largest_power_of_two(prime_bound: int) -> int:
    p, result = 2, 1
    while p <= prime_bound:
        result = p
        p *= 2
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. FUNÇÕES PURAS — EXECUTADAS NOS WORKER PROCESSES
#    Devem ser top-level (não closures) para serem picklable pelo multiprocessing.
# ─────────────────────────────────────────────────────────────────────────────

def _are_pairwise_coprime(moduli: tuple) -> bool:
    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            if gcd(moduli[i], moduli[j]) != 1:
                return False
    return True


def _modulus_class(m: int) -> str:
    if m & (m - 1) == 0:
        return "A"
    n = m.bit_length()
    if m == (1 << n) - 1 or m == (1 << (n - 1)) + 1:
        return "B"
    return "C"


def _carry_cascade_factor(m: int) -> int:
    return {"A": 0, "B": 1, "C": 2}[_modulus_class(m)]


# ── Worker: Filtro duro para um valor de k fixo ──────────────────────────────

def _filter_worker(
    moduli_pool: list[int],
    k:           int,
    N:           int,
    required_pow2: int,
    min_M:       int,
) -> list[tuple]:
    """
    Processa todas as C(pool, k) combinações para um dado k.
    Retorna apenas as que passam nos três filtros duros.
    Roda inteiramente em um processo filho — sem locks, sem estado compartilhado.
    """
    valid = []
    for combo in itertools.combinations(moduli_pool, k):
        # Filtro 1: deve conter a maior potência de 2
        if required_pow2 not in combo:
            continue
        # Filtro 2: produto mínimo
        M = 1
        for m in combo:
            M *= m
        if M < min_M:
            continue
        # Filtro 3: coprimidade total
        if not _are_pairwise_coprime(combo):
            continue
        valid.append(combo)
    return valid


# ── Worker: Cálculo de métricas para um chunk de conjuntos ───────────────────

def _metrics_worker(chunk: list[tuple], N: int) -> list[dict]:
    """
    Calcula as 4 dimensões de Pareto para cada conjunto no chunk.
    Retorna lista de dicts (picklable) em vez de dataclasses para evitar
    overhead de serialização de objetos complexos entre processos.
    """
    results = []
    class_score = {"A": 0, "B": 1, "C": 2}

    for moduli in chunk:
        M = 1
        for m in moduli:
            M *= m
        log2_M = math.log2(M)

        d1  = math.ceil(log2_M) - 2 * N
        d2  = len(moduli)
        delays = [m.bit_length() * _carry_cascade_factor(m) for m in moduli]
        d3  = float(max(delays) - min(delays))
        classes = tuple(_modulus_class(m) for m in moduli)
        d4  = float(max(class_score[c] for c in classes)) + log2_M

        non_trivial = [m for m in moduli if _modulus_class(m) != "A"]
        w_max = max(m.bit_length() for m in (non_trivial or moduli))
        evolvability = 1 << (2 * w_max)

        results.append({
            "moduli":          tuple(moduli),
            "M":               M,
            "N":               N,
            "d1_waste":        d1,
            "d2_channels":     d2,
            "d3_delta_delay":  d3,
            "d4_converter":    d4,
            "classes":         classes,
            "evolvability":    evolvability,
        })
    return results


# ── Worker: Pareto parcial sobre um chunk ────────────────────────────────────

def _pareto_partial_worker(chunk: list[dict]) -> list[dict]:
    """
    Reduz um chunk à sua fronteira de Pareto local.
    Isso diminui drasticamente o volume de dados antes do merge final,
    pois um conjunto dominado localmente nunca pode estar na fronteira global.
    """
    front = []
    for candidate in chunk:
        ca = (candidate["d1_waste"], candidate["d2_channels"],
              candidate["d3_delta_delay"], candidate["d4_converter"])
        dominated = False
        for other in chunk:
            if other is candidate:
                continue
            oa = (other["d1_waste"], other["d2_channels"],
                  other["d3_delta_delay"], other["d4_converter"])
            if all(o <= c for o, c in zip(oa, ca)) and any(o < c for o, c in zip(oa, ca)):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATACLASS — usada apenas no processo principal após desserialização
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SetMetrics:
    moduli:         tuple
    M:              int
    N:              int
    d1_waste:       int
    d2_channels:    int
    d3_delta_delay: float
    d4_converter:   float
    classes:        tuple
    evolvability:   int

    @staticmethod
    def from_dict(d: dict) -> "SetMetrics":
        return SetMetrics(**d)

    def dims(self) -> tuple:
        return (self.d1_waste, self.d2_channels, self.d3_delta_delay, self.d4_converter)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DOMINÂNCIA DE PARETO — merge final no processo principal
# ─────────────────────────────────────────────────────────────────────────────

def _dominates(a: SetMetrics, b: SetMetrics) -> bool:
    da, db = a.dims(), b.dims()
    return all(x <= y for x, y in zip(da, db)) and any(x < y for x, y in zip(da, db))


def pareto_front_merge(candidates: list[SetMetrics]) -> list[SetMetrics]:
    """
    Merge final: recebe as fronteiras locais de cada worker e elimina
    os conjuntos que são dominados globalmente.
    """
    front = []
    for candidate in candidates:
        dominated = any(
            _dominates(other, candidate)
            for other in candidates
            if other is not candidate
        )
        if not dominated:
            front.append(candidate)
    return front


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRUNING POR EVOLVABILIDADE
# ─────────────────────────────────────────────────────────────────────────────

def prune_by_evolvability(
    front: list[SetMetrics],
    max_truth_table: int,
) -> tuple[list[SetMetrics], list[SetMetrics]]:
    evolvable   = [s for s in front if s.evolvability <= max_truth_table]
    intractable = [s for s in front if s.evolvability >  max_truth_table]
    return evolvable, intractable


# ─────────────────────────────────────────────────────────────────────────────
# 6. SERIALIZAÇÃO E RELATÓRIO
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
        "module_classes":       dict(zip(s.moduli, s.classes)),
        "evolvability_tt_size": s.evolvability,
    }


def export_results(
    N, pareto, intractable, total_candidates,
    total_filtered, n_workers, elapsed, output_path,
) -> None:
    payload = {
        "config": {
            "N_bits":    N,
            "minimum_M": 1 << (2 * N),
            "workers_used": n_workers,
            "elapsed_seconds": round(elapsed, 3),
            "dimension_descriptions": {
                "d1_waste":       "ceil(log2(M)) - 2N  [minimizar — overhead de representação]",
                "d2_channels":    "k = número de canais RNS  [minimizar — roteamento]",
                "d3_delta_delay": "max(W*P) - min(W*P)  [minimizar — desbalanceamento de pipeline]",
                "d4_converter":   "ClasseMax_score + log2(M)  [minimizar — custo do conversor reverso]",
            },
        },
        "statistics": {
            "total_enumerated":    total_candidates,
            "after_hard_filters":  total_filtered,
            "pareto_front_size":   len(pareto),
            "intractable_pruned":  len(intractable),
            "ready_for_evolution": len(pareto),
        },
        "pareto_front":    [metrics_to_dict(s) for s in pareto],
        "intractable_sets":[metrics_to_dict(s) for s in intractable],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def print_report(N, pareto, intractable, total_candidates, total_filtered,
                 n_workers, elapsed) -> None:
    sep = "─" * 72
    print(f"\n{'═'*72}")
    print(f"  EvoCRTMul v2.0 — Fronteira de Pareto  |  N = {N} bits")
    print(f"{'═'*72}")
    print(f"  Workers (processos)              : {n_workers:>6}")
    print(f"  Tempo total de execução          : {elapsed:>9.3f}s")
    print(f"  Módulos enumerados (Prime Bound) : {total_candidates:>6}")
    print(f"  Conjuntos após filtros duros     : {total_filtered:>6}")
    print(f"  Conjuntos na fronteira de Pareto : {len(pareto):>6}")
    print(f"  Descartados (evolvabilidade)     : {len(intractable):>6}")
    print(sep)

    if not pareto:
        print("  [!] Nenhum conjunto sobreviveu aos filtros.")
        return

    print(f"  {'Módulos':<28} {'D1':>4} {'D2':>4} {'D3':>6} {'D4':>8}  Classes")
    print(sep)

    for s in sorted(pareto, key=lambda s: (s.d1_waste, s.d2_channels)):
        moduli_str  = "{" + ", ".join(str(m) for m in s.moduli) + "}"
        classes_str = " ".join(f"{m}:{c}" for m, c in zip(s.moduli, s.classes))
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

    if intractable:
        print(f"\n  [!] {len(intractable)} conjunto(s) na fronteira de Pareto marcados como")
        print( "      intratáveis para o motor CGP. Ver 'intractable_sets' no JSON.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 7. PIPELINE PARALELO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(N, prime_bound, max_channels, max_tt, output_path, n_workers) -> None:

    t_start = time.perf_counter()

    # ── Etapa 0: Enumeração ──────────────────────────────────────────────────
    moduli_pool  = enumerate_candidate_moduli(prime_bound)
    required_p2  = largest_power_of_two(prime_bound)
    min_M        = 1 << (2 * N)
    total_combos = sum(math.comb(len(moduli_pool), k) for k in range(2, max_channels + 1))

    print(f"\n[0] Pool de módulos (prime_bound={prime_bound}): {moduli_pool}")
    print(f"    Potência de 2 obrigatória: {required_p2}")
    print(f"    Workers disponíveis: {n_workers}")

    # ── Etapa 1: Filtros Duros (paralelo por k) ──────────────────────────────
    # Estratégia de partição: cada worker recebe um valor de k distinto.
    # C(pool, k) cresce com k, então a distribuição por k é naturalmente
    # desbalanceada — mas é a divisão mais simples e evita coordenação.
    # Para casos com muitos k e poucos workers, múltiplos k são agrupados.
    print(f"\n[1] Filtragem paralela (partição por k, {n_workers} workers)...")
    t1 = time.perf_counter()

    k_values = list(range(2, max_channels + 1))
    all_sets: list[tuple] = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_filter_worker, moduli_pool, k, N, required_p2, min_M): k
            for k in k_values
        }
        for future in as_completed(futures):
            k = futures[future]
            result = future.result()
            all_sets.extend(result)
            print(f"    k={k}: {math.comb(len(moduli_pool), k)} combinações → {len(result)} válidas")

    print(f"    Total: {total_combos} combinações → {len(all_sets)} passaram [{time.perf_counter()-t1:.3f}s]")

    if not all_sets:
        print("\n[!] Nenhum conjunto válido. Tente aumentar --prime-bound ou --max-channels.")
        return

    # ── Etapa 2: Cálculo de Métricas (paralelo em chunks) ───────────────────
    # Divide os conjuntos válidos em n_workers chunks de tamanho igual.
    print(f"\n[2] Calculando métricas em {n_workers} chunks paralelos...")
    t2 = time.perf_counter()

    chunk_size = max(1, math.ceil(len(all_sets) / n_workers))
    chunks = [all_sets[i:i + chunk_size] for i in range(0, len(all_sets), chunk_size)]

    raw_metrics: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_metrics_worker, chunk, N) for chunk in chunks]
        for future in as_completed(futures):
            raw_metrics.extend(future.result())

    print(f"    {len(raw_metrics)} métricas calculadas [{time.perf_counter()-t2:.3f}s]")

    # ── Etapa 3: Pareto em dois passos ───────────────────────────────────────
    # Passo 3a — Redução local: cada worker encontra o Pareto do seu chunk.
    #            Isso elimina a maioria dos dominados sem comunicação.
    # Passo 3b — Merge global: o processo principal faz o Pareto final
    #            sobre a união das fronteiras locais (conjunto muito menor).
    print(f"\n[3] Fronteira de Pareto em dois passos...")
    t3 = time.perf_counter()

    # Re-particiona raw_metrics em chunks para a redução local
    chunk_size_p = max(1, math.ceil(len(raw_metrics) / n_workers))
    pareto_chunks = [
        raw_metrics[i:i + chunk_size_p]
        for i in range(0, len(raw_metrics), chunk_size_p)
    ]

    # Passo 3a: fronteiras locais em paralelo
    local_fronts: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_pareto_partial_worker, chunk) for chunk in pareto_chunks]
        for future in as_completed(futures):
            local_fronts.extend(future.result())

    print(f"    Passo 3a — fronteiras locais: {len(raw_metrics)} → {len(local_fronts)} candidatos")

    # Passo 3b: merge global no processo principal
    local_metrics = [SetMetrics.from_dict(d) for d in local_fronts]
    global_front  = pareto_front_merge(local_metrics)
    print(f"    Passo 3b — merge global    : {len(local_fronts)} → {len(global_front)} não-dominados "
          f"[{time.perf_counter()-t3:.3f}s]")

    # ── Etapa 4: Pruning por Evolvabilidade ──────────────────────────────────
    print(f"\n[4] Pruning por evolvabilidade (limiar = 2^{int(math.log2(max_tt))})...")
    evolvable, intractable = prune_by_evolvability(global_front, max_truth_table=max_tt)
    print(f"    {len(evolvable)} prontos para evolução, {len(intractable)} descartados.")

    elapsed = time.perf_counter() - t_start

    # ── Relatório e Exportação ───────────────────────────────────────────────
    print_report(N, evolvable, intractable, total_combos, len(all_sets), n_workers, elapsed)
    export_results(N, evolvable, intractable, total_combos, len(all_sets),
                   n_workers, elapsed, output_path)
    print(f"[✓] Resultados exportados para: {output_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EvoCRTMul v2.0 — Fronteira de Pareto RNS (Multi-Processo)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bits", "-N", type=int, required=True,
        help="Largura N dos operandos (ex: 8 para multiplicador 8x8).")
    parser.add_argument("--prime-bound", type=int, default=None,
        help="Valor máximo dos módulos candidatos. Padrão: 2^(N/2+1)-1.")
    parser.add_argument("--max-channels", type=int, default=5,
        help="Número máximo de canais k em um conjunto.")
    parser.add_argument("--max-tt", type=int, default=20,
        help="Limiar de evolvabilidade em log2 (padrão: 2^20).")
    parser.add_argument("--workers", type=int, default=None,
        help="Número de processos worker. Padrão: os.cpu_count().")
    parser.add_argument("--output", "-o", type=str, default=None,
        help="Arquivo JSON de saída. Padrão: pareto_front_N<bits>.json")

    args = parser.parse_args()

    N            = args.bits
    prime_bound  = args.prime_bound or ((1 << (N // 2 + 1)) - 1)
    max_channels = args.max_channels
    max_tt       = 1 << args.max_tt
    n_workers    = args.workers or os.cpu_count() or 1
    output_path  = args.output or f"pareto_front_N{N}.json"

    run_pipeline(
        N=N,
        prime_bound=prime_bound,
        max_channels=max_channels,
        max_tt=max_tt,
        output_path=output_path,
        n_workers=n_workers,
    )


# CRÍTICO: guard obrigatório para multiprocessing no Windows e macOS (spawn).
# No Linux (fork) seria opcional, mas a boa prática exige sempre.
if __name__ == "__main__":
    main()
