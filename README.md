# Evolvable Hardware via Cartesian Genetic Programming
## Evolução Automática de Circuitos Aritméticos com Geração de VHDL/Verilog

---

## Fundamentação Científica

Este projeto implementa **Evolvable Hardware (EHW)** usando **Cartesian Genetic Programming (CGP)**, 
baseado nas seguintes referências canônicas da área:

| Referência | Contribuição |
|---|---|
| Miller & Thomson (2000). *Cartesian Genetic Programming*. EuroGP 2000, LNCS 1802. | Definição formal do CGP e codificação em grafo cartesiano |
| Beyer & Schwefel (2002). *Evolution strategies – a comprehensive introduction*. Natural Computing 1(1):3–52. | Estratégia (1+λ)-ES usada como motor evolutivo |
| Shang et al. (2020). *Evolvable Hardware Design of Digital Circuits Based on Adaptive Genetic Algorithm*. Springer. | Mutação adaptativa para evitar estagnação |
| Kalkreuth (2024). *CGP++: A Modern C++ Implementation of CGP*. GECCO 2024, Melbourne. | Estado da arte: recombinação e variantes modernas |
| Kalkreuth et al. (2024). *Using M-CGP for Automatic Design of Digital Sequential Circuits*. Applied Sciences 14(23). | Aplicação a circuitos sequenciais digitais |

---

## Arquitetura do Sistema

```
evolvable_hw/
├── src/
│   ├── cgp_core.py     # Representação cromossômica CGP + decodificação de grafo
│   ├── fitness.py      # Funções de fitness: multiplicador, somador, divisor
│   ├── evolution.py    # Motor (1+λ)-ES com mutação adaptativa
│   ├── export.py       # Geração de VHDL e Verilog sintetizáveis
│   └── main.py         # CLI e orquestrador
└── output/             # Arquivos HDL gerados
```

---

## Representação CGP

### Codificação Cromossômica

Um cromossomo CGP é uma grade `n_cols × n_rows` de nós (portas lógicas).
Cada nó é codificado como:

```
gene = [tipo_porta, conexão_A, conexão_B]
```

Onde `conexão_A` e `conexão_B` são índices para o **barramento de valores**:
- Índices `0..n_inputs-1`: entradas primárias
- Índices `n_inputs..n_inputs+n_nodes-1`: saídas de nós anteriores

O parâmetro **levels_back** (L) controla o alcance das conexões:
- Cada nó na coluna `c` só pode conectar a nós nas colunas `max(0, c-L)..c-1` ou entradas primárias.

### Conjunto de Portas

| Porta | Aridade | Operação |
|-------|---------|----------|
| AND   | 2 | A ∧ B |
| OR    | 2 | A ∨ B |
| XOR   | 2 | A ⊕ B |
| NAND  | 2 | ¬(A ∧ B) |
| NOR   | 2 | ¬(A ∨ B) |
| XNOR  | 2 | ¬(A ⊕ B) |
| NOT   | 1 | ¬A |
| BUF   | 1 | A (wire) |

### Nós Ativos

Um nó é **ativo** se e somente se sua saída influencia (direta ou transitivamente) 
alguma saída primária. A análise é feita por travessia reversa a partir dos genes de saída.
Apenas nós ativos são incluídos no VHDL/Verilog gerado.

---

## Motor Evolutivo: (1+λ)-ES

```
Inicializa pai P aleatoriamente
Avalia fitness(P)

loop por max_gens gerações:
    Gera λ filhos: C_i = mutate(P)  para i=1..λ
    Avalia fitness(C_i)
    Seleciona melhor filho C* = argmax fitness(C_i)
    
    if fitness(C*) >= fitness(P):
        P ← C*          # deriva neutra permitida (=)
    
    if estagnação > limiar:
        aumenta taxa de mutação (mutação adaptativa)
    
    if fitness(P) == 1.0:
        break

return P
```

A **deriva neutra** (aceitar filhos com fitness igual) é fundamental no CGP:
permite explorar o espaço de busca sem piorar o fitness, escapando de regiões planas.

### Mutação Adaptativa

Inspirada em Shang et al. (2020): quando a evolução estagna por `stagnation_limit` gerações,
a taxa de mutação é multiplicada por 1.5 (até máximo de 30%). Quando há melhora, 
decai gradualmente de volta à taxa base.

---

## Circuitos Alvo

### Multiplicador N×N bits
- **Entradas**: 2·N bits (operandos A e B)
- **Saídas**: 2·N bits (produto P)
- **Padrões de teste**: 4^N (todos os pares possíveis)
- **Complexidade**: O(N²) portas na solução ótima manual

### Somador N bits
- **Entradas**: 2·N bits (operandos A e B)
- **Saídas**: N+1 bits (soma + carry)
- **Base para comparação**: ripple-carry adder

### Divisor N bits
- **Entradas**: 2·N bits (dividendo A, divisor B)
- **Saídas**: 2·N bits (quociente Q, resto R)
- **Convenção**: divisão por zero → Q=0, R=0

---

## Função de Fitness

```
fitness = bits_corretos / (total_padrões × n_saídas)
```

Varia de 0.0 (pior) a 1.0 (circuito perfeito em todos os padrões).

---

## Uso

### Instalação
```bash
# Sem dependências externas — Python 3.10+ puro
python --version  # >= 3.10 necessário (match/case usado no core)
```

### Exemplos

```bash
# Multiplicador 2×2 bits (mais rápido, bom para testes)
python src/main.py --circuit multiplier --bits 2 --cols 40 --gens 50000

# Somador 4 bits
python src/main.py --circuit adder --bits 4 --cols 64 --gens 100000

# Divisor 2 bits
python src/main.py --circuit divider --bits 2 --cols 48 --gens 80000

# Fixar semente para reprodutibilidade
python src/main.py --circuit multiplier --bits 2 --seed 42

# Apenas Verilog (sem VHDL)
python src/main.py --no-vhdl

# Desabilitar mutação adaptativa
python src/main.py --no-adaptive --mut 0.03
```

### Parâmetros CLI

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--circuit` | multiplier | Tipo: multiplier, adder, divider |
| `--bits` | 2 | Largura de bits |
| `--cols` | 40 | Colunas do grid CGP |
| `--rows` | 1 | Linhas do grid (padrão CGP = 1) |
| `--back` | =cols | Levels-back L |
| `--lambda` | 4 | Número de filhos por geração |
| `--gens` | 30000 | Máximo de gerações |
| `--mut` | 0.05 | Taxa de mutação base |
| `--no-adaptive` | — | Desabilitar mutação adaptativa |
| `--seed` | None | Semente aleatória |
| `--outdir` | output/ | Diretório de saída HDL |

---

## Exemplo de Saída VHDL

```vhdl
-- Evolved Arithmetic Circuit — 2×2 Unsigned Multiplier
-- Method : Cartesian Genetic Programming (CGP)
-- Fitness: 1.000000
-- Active nodes: 14 / 40

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity multiplier_2bit is
  port (
    inp_0 : in  STD_LOGIC;  -- A[1]
    inp_1 : in  STD_LOGIC;  -- A[0]
    inp_2 : in  STD_LOGIC;  -- B[1]
    inp_3 : in  STD_LOGIC;  -- B[0]
    out_0 : out STD_LOGIC;  -- P[3]
    out_1 : out STD_LOGIC;  -- P[2]
    out_2 : out STD_LOGIC;  -- P[1]
    out_3 : out STD_LOGIC   -- P[0]
  );
end multiplier_2bit;
```

O VHDL gerado é sintetizável em ferramentas como Xilinx Vivado, Intel Quartus e GHDL.

---

## Complexidade e Escalabilidade

| Circuito | Bits | Entradas | Saídas | Padrões | Grid Recomendado |
|----------|------|----------|--------|---------|-----------------|
| Multiplicador | 2 | 4 | 4 | 16 | 40 cols |
| Multiplicador | 3 | 6 | 6 | 64 | 80 cols |
| Somador | 4 | 8 | 5 | 256 | 64 cols |
| Divisor | 2 | 4 | 4 | 16 | 48 cols |

---

## Referências

1. Miller, J.F. & Thomson, P. (2000). Cartesian genetic programming. *EuroGP 2000*, LNCS 1802, 121–132. Springer.
2. Beyer, H.-G. & Schwefel, H.-P. (2002). Evolution strategies – a comprehensive introduction. *Natural Computing*, 1(1), 3–52.
3. Shang, Q. et al. (2020). Evolvable Hardware Design of Digital Circuits Based on Adaptive Genetic Algorithm. *Springer AISC*.
4. Kalkreuth, R. (2024). CGP++: A Modern C++ Implementation of Cartesian Genetic Programming. *GECCO 2024*, Melbourne.
5. Kalkreuth, R. et al. (2024). Using M-CGP for Automatic Design of Digital Sequential Circuits. *Applied Sciences*, 14(23), 11153.
6. Miller, J.F. (2019). Cartesian genetic programming: its status and future. *Genetic Programming and Evolvable Machines*, 21, 129–168.

