"""
EvoCRTMul v2.0 — Fronteira de Pareto RNS (Máxima Paralelização)
================================================================
Estratégia de decomposição por âncora dupla
-------------------------------------------
O gargalo da versão anterior era a Etapa 1: apenas `max_channels - 1`
futures no total (um por valor de k), deixando cores ociosos.

Solução: para cada (k, anchor_i, anchor_j), geramos uma task independente
que itera apenas sobre os C(pool-2, k-2) combos com os dois primeiros
elementos fixados. Para N=16 isso produz ~2346 tasks por k — suficiente
para saturar qualquer número de cores sem coordenação ou locks.

Todas as etapas CPU-bound são paralelas:

  Etapa 1 — Filtragem   : C(pool, 2) × (max_k - 1) tasks (âncora dupla)
  Etapa 2 — Métricas    : chunks de tamanho fixo, uma task por chunk
  Etapa 3a— Pareto local: chunks independentes, redução sem comunicação
  Etapa 3b— Pareto merge: serial no principal, mas sobre conjunto mínimo

Execução:
    python pareto_rns_maxpar.py --bits 8
    python pareto_rns_maxpar.py --bits 16 --prime-bound 255
    python pareto_rns_maxpar.py --bits 16 --prime-bound 255 --workers 16
    python pareto_rns_maxpar.py --bits 8  --prime-bound 63 --output out.json
"""

import argparse
import json
import math
import os
import sqlite3
import time
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from math import gcd
from sympy import factorint


# ─────────────────────────────────────────────────────────────────────────────
# 1. ENUMERAÇÃO — processo principal, rápido, não paralelizado
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_candidate_moduli(prime_bound: int) -> list[int]:
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
# 2. FUNÇÕES PURE — top-level para serem picklable pelos workers
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


def _carry_cascade(m: int) -> int:
    return {"A": 0, "B": 1, "C": 2}[_modulus_class(m)]


# ─────────────────────────────────────────────────────────────────────────────
# 3. WORKER DE FILTRAGEM — âncora dupla
#
# Recebe um par fixo (a0, a1) e itera sobre C(tail_pool, k-2) combos,
# completando cada conjunto com os k-2 elementos restantes do tail_pool.
# Isso divide o espaço de busca em C(pool, 2) partições disjuntas e
# completas — sem sobreposição, sem lacunas.
# ─────────────────────────────────────────────────────────────────────────────

def _filter_anchor_worker(
    a0:          int,
    a1:          int,
    tail_pool:   list[int],   # pool sem a0 e a1 (e sem elementos < a1)
    k_values:    list[int],   # lista de k a processar nesta task
    N:           int,
    required_p2: int,
    min_M:       int,
) -> list[tuple]:
    """
    Filtra todos os conjuntos de tamanho k que começam com (a0, a1).
    Uma única task cobre múltiplos valores de k para amortizar o overhead
    de spawn quando o pool é pequeno.
    """
    # Pré-calcula produto e coprimidade da âncora
    anchor_gcd_ok = gcd(a0, a1) == 1
    anchor_product = a0 * a1
    anchor_has_p2 = (a0 == required_p2 or a1 == required_p2)

    valid = []

    for k in k_values:
        remaining = k - 2
        if remaining < 0:
            continue
        if remaining == 0:
            # O conjunto é exatamente a âncora
            moduli = (a0, a1)
            M = anchor_product
            if (anchor_gcd_ok
                    and M >= min_M
                    and (anchor_has_p2 or required_p2 in moduli)):
                valid.append(moduli)
            continue

        if not anchor_gcd_ok:
            # Âncora já falhou em coprimidade — nenhum superconjunto passa
            continue

        for rest in itertools.combinations(tail_pool, remaining):
            # Filtro rápido 1: potência de 2 deve estar presente
            if not anchor_has_p2 and required_p2 not in rest:
                continue

            # Filtro rápido 2: produto mínimo
            M = anchor_product
            for m in rest:
                M *= m
            if M < min_M:
                continue

            # Filtro 3: coprimidade entre rest e âncora, e dentro de rest
            moduli = (a0, a1) + rest
            if _are_pairwise_coprime(moduli):
                valid.append(moduli)

    return valid


# ─────────────────────────────────────────────────────────────────────────────
# 4. WORKER DE MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def _metrics_worker(chunk: list[tuple], N: int) -> list[dict]:
    """Calcula as 4 dimensões de Pareto para cada conjunto no chunk."""
    class_score = {"A": 0, "B": 1, "C": 2}
    results = []

    for moduli in chunk:
        M = 1
        for m in moduli:
            M *= m
        log2_M = math.log2(M)

        d1 = math.ceil(log2_M) - 2 * N
        d2 = len(moduli)

        delays  = [m.bit_length() * _carry_cascade(m) for m in moduli]
        d3      = float(max(delays) - min(delays))

        classes = tuple(_modulus_class(m) for m in moduli)
        d4      = float(max(class_score[c] for c in classes)) + log2_M

        non_trivial = [m for m in moduli if _modulus_class(m) != "A"]
        w_max       = max(m.bit_length() for m in (non_trivial or list(moduli)))
        evolvability = 1 << (2 * w_max)

        results.append({
            "moduli":         tuple(moduli),
            "M":              M,
            "N":              N,
            "d1_waste":       d1,
            "d2_channels":    d2,
            "d3_delta_delay": d3,
            "d4_converter":   d4,
            "classes":        classes,
            "evolvability":   evolvability,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. WORKER DE PARETO LOCAL
# ─────────────────────────────────────────────────────────────────────────────

def _pareto_local_worker(chunk: list[dict]) -> list[dict]:
    """
    Reduz o chunk à sua fronteira de Pareto local.
    Um conjunto dominado localmente nunca pode estar na fronteira global,
    eliminando a maioria dos candidatos antes do merge final.
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
# 6. DATACLASS E MERGE FINAL — processo principal
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
        return (self.d1_waste, self.d2_channels,
                self.d3_delta_delay, self.d4_converter)


def _dominates(a: SetMetrics, b: SetMetrics) -> bool:
    da, db = a.dims(), b.dims()
    return (all(x <= y for x, y in zip(da, db))
            and any(x <  y for x, y in zip(da, db)))


def pareto_merge(candidates: list[SetMetrics]) -> list[SetMetrics]:
    """Merge global: elimina dominados na união das fronteiras locais."""
    return [
        c for c in candidates
        if not any(_dominates(o, c) for o in candidates if o is not c)
    ]


def prune_by_evolvability(
    front: list[SetMetrics], max_tt: int
) -> tuple[list[SetMetrics], list[SetMetrics]]:
    return (
        [s for s in front if s.evolvability <= max_tt],
        [s for s in front if s.evolvability >  max_tt],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. SERIALIZAÇÃO E RELATÓRIO
# ─────────────────────────────────────────────────────────────────────────────

def _to_dict(s: SetMetrics) -> dict:
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


def export_results(N, pareto, intractable, stats, output_path) -> None:
    payload = {
        "config": {
            "N_bits":    N,
            "minimum_M": 1 << (2 * N),
            "workers_used":    stats["workers"],
            "elapsed_seconds": round(stats["elapsed"], 3),
            "dimension_descriptions": {
                "d1_waste":       "ceil(log2(M)) - 2N  [minimizar]",
                "d2_channels":    "k = número de canais RNS  [minimizar]",
                "d3_delta_delay": "max(W*P) - min(W*P)  [minimizar]",
                "d4_converter":   "ClasseMax_score + log2(M)  [minimizar]",
            },
        },
        "statistics": {
            "total_enumerated":    stats["total_combos"],
            "after_hard_filters":  stats["after_filter"],
            "pareto_front_size":   len(pareto),
            "intractable_pruned":  len(intractable),
            "ready_for_evolution": len(pareto),
            "timings":             stats["timings"],
        },
        "pareto_front":     [_to_dict(s) for s in pareto],
        "intractable_sets": [_to_dict(s) for s in intractable],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def print_report(N, pareto, intractable, stats) -> None:
    sep = "─" * 76
    print(f"\n{'═'*76}")
    print(f"  EvoCRTMul v2.0 — Fronteira de Pareto  |  N = {N} bits  (máxima paralelização)")
    print(f"{'═'*76}")
    print(f"  Workers (processos)              : {stats['workers']:>6}")
    print(f"  Tasks despachadas                : {stats['total_tasks']:>6}")
    print(f"  Tempo total                      : {stats['elapsed']:>9.3f}s")
    for label, t in stats["timings"].items():
        print(f"    {label:<32}: {t:.3f}s")
    print(sep)
    print(f"  Combinações verificadas          : {stats['total_combos']:>12,}")
    print(f"  Após filtros duros               : {stats['after_filter']:>12,}")
    print(f"  Fronteira de Pareto              : {len(pareto):>12,}")
    print(f"  Descartados (evolvabilidade)     : {len(intractable):>12,}")
    print(sep)

    if not pareto:
        print("  [!] Nenhum conjunto sobreviveu.")
        return

    print(f"  {'Módulos':<30} {'D1':>4} {'D2':>4} {'D3':>6} {'D4':>8}  Classes")
    print(sep)
    for s in sorted(pareto, key=lambda s: (s.d1_waste, s.d2_channels, s.d3_delta_delay)):
        ms = "{" + ", ".join(str(m) for m in s.moduli) + "}"
        cs = " ".join(f"{m}:{c}" for m, c in zip(s.moduli, s.classes))
        print(f"  {ms:<30} {s.d1_waste:>4} {s.d2_channels:>4}"
              f" {s.d3_delta_delay:>6.1f} {s.d4_converter:>8.2f}  {cs}")
    print(sep)
    print("  D1=desperdício  D2=canais  D3=desbalanceamento  D4=custo conversor")
    if intractable:
        print(f"\n  [!] {len(intractable)} conjunto(s) intratáveis para CGP → ver 'intractable_sets' no JSON.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 8. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(N, prime_bound, max_channels, max_tt, output_path, n_workers, db_path=None) -> None:

    t_wall = time.perf_counter()
    timings = {}

    # ── Etapa 0: Enumeração ──────────────────────────────────────────────────
    moduli_pool  = enumerate_candidate_moduli(prime_bound)
    required_p2  = largest_power_of_two(prime_bound)
    min_M        = 1 << (2 * N)
    pool_size    = len(moduli_pool)
    k_values     = list(range(2, max_channels + 1))

    total_combos = sum(math.comb(pool_size, k) for k in k_values)
    # Número de tasks: C(pool,2) pares de âncora × (max_channels-1) valores de k
    # Na prática agrupamos todos os k numa mesma task por par de âncora
    n_anchor_tasks = math.comb(pool_size, 2)

    print(f"\n[0] Pool de módulos (prime_bound={prime_bound}): {pool_size} módulos")
    print(f"    Potência de 2 obrigatória : {required_p2}")
    print(f"    Combinações a verificar   : {total_combos:,}")
    print(f"    Tasks de filtragem        : {n_anchor_tasks:,}  ({n_anchor_tasks} pares âncora × todos k)")
    print(f"    Workers                   : {n_workers}")

    # ── Etapa 1: Filtragem por âncora dupla ──────────────────────────────────
    # Cada task recebe (a0, a1) fixos e itera C(tail, k-2) para cada k.
    # As tasks são geradas em ordem lexicográfica dos pares de âncora.
    print(f"\n[1] Filtragem paralela por âncora dupla...")
    t1 = time.perf_counter()

    all_sets: list[tuple] = []
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for idx_a0, a0 in enumerate(moduli_pool):
            # tail_pool: elementos estritamente maiores que a0 (garante ordem
            # lexicográfica e evita duplicatas entre tasks)
            tail_after_a0 = moduli_pool[idx_a0 + 1:]
            for idx_a1, a1 in enumerate(tail_after_a0):
                # tail para os k-2 restantes: elementos > a1
                tail = tail_after_a0[idx_a1 + 1:]
                f = pool.submit(
                    _filter_anchor_worker,
                    a0, a1, tail, k_values, N, required_p2, min_M,
                )
                futures[f] = (a0, a1)

        total_tasks = len(futures)
        for future in as_completed(futures):
            result = future.result()
            all_sets.extend(result)
            completed += 1
            # Progresso a cada 10% das tasks
            if completed % max(1, total_tasks // 10) == 0:
                pct = 100 * completed / total_tasks
                print(f"    {pct:5.1f}%  ({completed}/{total_tasks} tasks, "
                      f"{len(all_sets):,} válidos até agora)")

    timings["1_filtragem"] = time.perf_counter() - t1
    print(f"    Total: {total_combos:,} combinações → {len(all_sets):,} válidas "
          f"[{timings['1_filtragem']:.3f}s]")

    if not all_sets:
        print("\n[!] Nenhum conjunto válido. Tente aumentar --prime-bound ou --max-channels.")
        return

    # ── Etapa 2: Cálculo de Métricas ─────────────────────────────────────────
    # Chunk size calibrado para ~4× mais tasks que workers: bom balanço
    # entre overhead de spawn e granularidade de carga.
    print(f"\n[2] Calculando métricas ({len(all_sets):,} conjuntos)...")
    t2 = time.perf_counter()

    chunk_size  = max(1, math.ceil(len(all_sets) / (n_workers * 4)))
    chunks      = [all_sets[i:i + chunk_size] for i in range(0, len(all_sets), chunk_size)]
    raw_metrics: list[dict] = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures_m = [pool.submit(_metrics_worker, chunk, N) for chunk in chunks]
        for future in as_completed(futures_m):
            raw_metrics.extend(future.result())

    timings["2_metricas"] = time.perf_counter() - t2
    print(f"    {len(raw_metrics):,} métricas em {len(chunks)} chunks [{timings['2_metricas']:.3f}s]")

    # ── Etapa 3: Pareto em dois passos ───────────────────────────────────────
    # Passo 3a: redução local em chunks — elimina dominados dentro de cada chunk
    # Passo 3b: merge global no principal — sobre o conjunto residual mínimo
    print(f"\n[3] Fronteira de Pareto em dois passos...")
    t3 = time.perf_counter()

    chunk_size_p  = max(1, math.ceil(len(raw_metrics) / (n_workers * 4)))
    pareto_chunks = [raw_metrics[i:i + chunk_size_p]
                     for i in range(0, len(raw_metrics), chunk_size_p)]
    local_fronts: list[dict] = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures_p = [pool.submit(_pareto_local_worker, chunk) for chunk in pareto_chunks]
        for future in as_completed(futures_p):
            local_fronts.extend(future.result())

    timings["3a_pareto_local"] = time.perf_counter() - t3
    print(f"    Passo 3a: {len(raw_metrics):,} → {len(local_fronts)} candidatos locais "
          f"[{timings['3a_pareto_local']:.3f}s]")

    t3b = time.perf_counter()
    global_front = pareto_merge([SetMetrics.from_dict(d) for d in local_fronts])
    timings["3b_pareto_merge"] = time.perf_counter() - t3b
    print(f"    Passo 3b: {len(local_fronts)} → {len(global_front)} não-dominados "
          f"[{timings['3b_pareto_merge']:.3f}s]")

    # ── Etapa 4: Pruning ─────────────────────────────────────────────────────
    print(f"\n[4] Pruning por evolvabilidade (limiar = 2^{int(math.log2(max_tt))})...")
    evolvable, intractable = prune_by_evolvability(global_front, max_tt)
    print(f"    {len(evolvable)} prontos para evolução, {len(intractable)} descartados.")

    elapsed = time.perf_counter() - t_wall
    timings["total"] = elapsed

    stats = {
        "workers":      n_workers,
        "total_tasks":  total_tasks,
        "elapsed":      elapsed,
        "timings":      timings,
        "total_combos": total_combos,
        "after_filter": len(all_sets),
    }

    print_report(N, evolvable, intractable, stats)
    export_results(N, evolvable, intractable, stats, output_path)
    print(f"[✓] Resultados exportados para: {output_path}")

    if db_path:
        dump_to_sqlite(
            db_path=db_path,
            N=N,
            prime_bound=prime_bound,
            max_channels=max_channels,
            max_tt_log2=int(math.log2(max_tt)),
            pareto=evolvable,
            intractable=intractable,
            stats=stats,
        )



# ─────────────────────────────────────────────────────────────────────────────
# 9. PERSISTÊNCIA SQLite
# ─────────────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    N_bits        INTEGER NOT NULL,
    prime_bound   INTEGER NOT NULL,
    max_channels  INTEGER NOT NULL,
    max_tt_log2   INTEGER NOT NULL,
    workers       INTEGER NOT NULL,
    total_combos  INTEGER NOT NULL,
    after_filter  INTEGER NOT NULL,
    pareto_size   INTEGER NOT NULL,
    intractable   INTEGER NOT NULL,
    elapsed_s     REAL    NOT NULL,
    ran_at        TEXT    NOT NULL
);

-- Conjuntos não-dominados (e intratáveis) da fronteira de Pareto.
-- set_hash = SHA-1 dos módulos ordenados: permite detectar o mesmo conjunto
-- em execuções diferentes sem varrer a coluna moduli_json.
CREATE TABLE IF NOT EXISTS pareto_sets (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER NOT NULL REFERENCES runs(run_id),
    set_hash       TEXT    NOT NULL,
    N_bits         INTEGER NOT NULL,
    moduli_json    TEXT    NOT NULL,
    M              INTEGER NOT NULL,
    d1_waste       INTEGER NOT NULL,
    d2_channels    INTEGER NOT NULL,
    d3_delta_delay REAL    NOT NULL,
    d4_converter   REAL    NOT NULL,
    evolvability   INTEGER NOT NULL,
    is_intractable INTEGER NOT NULL DEFAULT 0,
    UNIQUE(set_hash, run_id)
);

-- Módulos individuais normalizados: permite consultas do tipo
-- "quais conjuntos contêm o módulo 31?" sem deserializar JSON.
CREATE TABLE IF NOT EXISTS set_modules (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    set_id         INTEGER NOT NULL REFERENCES pareto_sets(id),
    run_id         INTEGER NOT NULL REFERENCES runs(run_id),
    modulus        INTEGER NOT NULL,
    modulus_class  TEXT    NOT NULL,
    bit_width      INTEGER NOT NULL,
    carry_cascade  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pareto_run    ON pareto_sets(run_id);
CREATE INDEX IF NOT EXISTS idx_pareto_hash   ON pareto_sets(set_hash);
CREATE INDEX IF NOT EXISTS idx_pareto_N      ON pareto_sets(N_bits);
CREATE INDEX IF NOT EXISTS idx_modules_set   ON set_modules(set_id);
CREATE INDEX IF NOT EXISTS idx_modules_mod   ON set_modules(modulus);
"""


def dump_to_sqlite(
    db_path:      str,
    N:            int,
    prime_bound:  int,
    max_channels: int,
    max_tt_log2:  int,
    pareto:       list,
    intractable:  list,
    stats:        dict,
) -> int:
    """
    Persiste os resultados de uma execução do pipeline em um banco SQLite.

    Cria o arquivo e o schema automaticamente se o arquivo não existir.
    Execuções repetidas acumulam histórico — nenhum dado anterior é
    sobrescrito. Cada execução recebe um run_id único e incremental.

    Parâmetros
    ----------
    db_path      : caminho do arquivo .db  (criado se ausente)
    N            : largura dos operandos em bits
    prime_bound  : limite máximo do pool de módulos
    max_channels : k máximo permitido
    max_tt_log2  : log2 do limiar de evolvabilidade usado no pruning
    pareto       : conjuntos da fronteira de Pareto prontos para evolução
    intractable  : conjuntos descartados por evolvabilidade
    stats        : dicionário de metadados retornado pelo pipeline

    Retorna
    -------
    run_id : identificador inteiro da execução recém-inserida

    Schema
    ------
    runs        — metadados de cada execução (1 linha por chamada)
    pareto_sets — conjuntos não-dominados e intratáveis (N linhas)
    set_modules — módulos individuais normalizados (N×k linhas)

    Exemplos de consultas úteis
    ---------------------------
    -- Todos os conjuntos Pareto para N=8:
    SELECT moduli_json, d1_waste, d2_channels, d3_delta_delay, d4_converter
    FROM pareto_sets WHERE N_bits=8 AND is_intractable=0;

    -- Quais conjuntos contêm o módulo 31?
    SELECT ps.moduli_json FROM set_modules sm
    JOIN pareto_sets ps ON sm.set_id = ps.id
    WHERE sm.modulus=31 AND ps.is_intractable=0;

    -- Histórico de execuções:
    SELECT run_id, N_bits, prime_bound, pareto_size, elapsed_s, ran_at
    FROM runs ORDER BY ran_at DESC;
    """
    import hashlib
    from datetime import datetime, timezone

    is_new = not os.path.exists(db_path)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")   # seguro para writes concorrentes futuros
    con.execute("PRAGMA foreign_keys=ON")

    try:
        con.executescript(_DDL)

        # ── Inserir metadados da execução ────────────────────────────────────
        cur = con.execute(
            """
            INSERT INTO runs
                (N_bits, prime_bound, max_channels, max_tt_log2, workers,
                 total_combos, after_filter, pareto_size, intractable,
                 elapsed_s, ran_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                N, prime_bound, max_channels, max_tt_log2,
                stats["workers"],
                stats["total_combos"],
                stats["after_filter"],
                len(pareto),
                len(intractable),
                round(stats["elapsed"], 6),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        run_id = cur.lastrowid

        # ── Inserir conjuntos: fronteira de Pareto + intratáveis ─────────────
        all_sets = [(s, 0) for s in pareto] + [(s, 1) for s in intractable]

        for s, flag in all_sets:
            moduli_sorted = tuple(sorted(s.moduli))
            set_hash = hashlib.sha1(
                ",".join(str(m) for m in moduli_sorted).encode()
            ).hexdigest()

            cur2 = con.execute(
                """
                INSERT OR IGNORE INTO pareto_sets
                    (run_id, set_hash, N_bits, moduli_json, M,
                     d1_waste, d2_channels, d3_delta_delay, d4_converter,
                     evolvability, is_intractable)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, set_hash, s.N,
                    json.dumps(list(moduli_sorted)),
                    s.M,
                    s.d1_waste, s.d2_channels,
                    s.d3_delta_delay,
                    round(s.d4_converter, 6),
                    s.evolvability,
                    flag,
                ),
            )
            set_id = cur2.lastrowid

            # ── Módulos individuais normalizados ─────────────────────────────
            con.executemany(
                """
                INSERT INTO set_modules
                    (set_id, run_id, modulus, modulus_class, bit_width, carry_cascade)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (set_id, run_id, m, cls, m.bit_length(), _carry_cascade(m))
                    for m, cls in zip(s.moduli, s.classes)
                ],
            )

        con.commit()

    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    action = "criado" if is_new else "atualizado"
    print(f"\n[DB] Banco {action}: {db_path!r}")
    print(f"     run_id={run_id}  |  "
          f"{len(pareto)} conjuntos Pareto + {len(intractable)} intratáveis persistidos")
    return run_id


# ─────────────────────────────────────────────────────────────────────────────
# 10. CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EvoCRTMul v2.0 — Fronteira de Pareto RNS (máxima paralelização)",
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
    parser.add_argument("--db", type=str, default=None,
        help="Arquivo SQLite para persistência. Criado automaticamente se ausente.")

    args        = parser.parse_args()
    N           = args.bits
    prime_bound = args.prime_bound or ((1 << (N // 2 + 1)) - 1)
    max_tt      = 1 << args.max_tt
    n_workers   = args.workers or os.cpu_count() or 1
    output_path = args.output or f"pareto_front_N{N}.json"

    run_pipeline(
        N=N,
        prime_bound=prime_bound,
        max_channels=args.max_channels,
        max_tt=max_tt,
        output_path=output_path,
        n_workers=n_workers,
        db_path=args.db,
    )


# Guard obrigatório — spawn no Windows/macOS re-importa o módulo
if __name__ == "__main__":
    main()
