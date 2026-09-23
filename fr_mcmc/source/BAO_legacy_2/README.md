# BAO legacy 2 (`BAO_legacy_2`)

Compilación de **15** medidas de BAO que combina la parte de bajo redshift de
`../BAO_legacy_1/` con **DESI DR1**, eligiendo los puntos de forma que **no se solapen**.

Antes esta carpeta se llamaba `BAO_full/`; se renombró el 2026-09-22, sin tocar los archivos
(que siguen llamándose `BAO_full_1.csv` y `BAO_full_2.csv`, igual que los argumentos de
`utils/data.read_data_BAO_full`).

## Composición, verificada archivo contra archivo

| origen | medidas | z |
|---|---|---|
| `../BAO_legacy_1/` | 6 | 0.15, 0.38, 0.44, 0.51 |
| `../DESI/` DR1 (`DESI_data_*.txt`) | 7 | 0.706, 0.930, 1.317, 1.491 |
| ninguno de los dos | 2 | 2.33 (par Ly-alpha) |

**No hay solapamiento**: la parte de legacy 1 llega hasta z = 0.51 y la de DESI DR1 arranca en
z = 0.706, sin ningún z ni observable repetido. El corte es deliberado — para quedarse con
BOSS DR12 en z = 0.51 hubo que dejar afuera el LRG1 de DESI DR1, que está en z = 0.510.

**Ojo: no es "legacy 1 más DESI DR1".** Descarta **14 de las 20** medidas de legacy 1 (todas las
de 0.6 ≤ z ≤ 2.4: WiggleZ en 0.6 y 0.73, BOSS DR12 en 0.61, eBOSS LRG en 0.698, DES Y1 en 0.81,
eBOSS QSO en 1.48 y 1.52, y los dos pares Ly-alpha en 2.33 y 2.4), justamente porque se solapan
con los trazadores de DESI. Y tampoco toma **5 de las 12** medidas de DESI DR1: el BGS de
z = 0.295, el par LRG1 de z = 0.510 y el par Ly-alpha de z = 2.330.

> **El par Ly-alpha de z = 2.33 no está identificado.** Tiene D_M/r_d = 38.80 ± 0.75 y
> D_H/r_d = 8.72 ± 0.14, que no coinciden con DESI DR1 (39.71 ± 0.94 y 8.52 ± 0.17), ni con
> DESI DR2 (38.99 y 8.63), ni con el Ly-alpha de legacy 1 (37.77 ± 2.13 y 9.07 ± 0.31). Esos
> valores no aparecen en ningún otro archivo del repo: salen sólo de `BAO_full_original.txt`,
> que empieza con el comentario `#BAO CLAUDIA`. **Conviene rastrear de dónde vienen.**

## Archivos

- `BAO_full_1.csv` — 7 medidas sueltas, sin correlacionar. Columnas `z`, `Dist`, `Stat_error`,
  `Sist_error`, `Type`, `index`; `index` es el de `BAO.Hs_to_Ds` (2 = D_M, 3 = D_V, 4 = H·r_d).
- `BAO_full_2.csv` — 4 pares (D_M/r_d, D_H/r_d) con su coeficiente de correlación `rho`.
- `BAO_full_original.txt` — la tabla original de la que salieron los dos CSV, con los nombres de
  trazadores de DESI (`#LRG2`, `#LRG3 + ELG1`, `#ELG2`, `#QSO`, `#Lyman-alpha`).
- `BAO_full_1.txt`, `BAO_full_2.txt`, `plot_data.py`, `plot_data.ipynb` — auxiliares, no los usa
  el pipeline.

Se leen con `utils/data.read_data_BAO_full`, que devuelve los dos conjuntos por separado.

## `BAO_full_1.csv`

| z | observable | valor | error est. | error sist. | origen |
|---|---|---|---|---|---|
| 0.38 | Dm/r_d | 10.272 | 0.135 | 0.074 | BOSS DR12 — de legacy 1 |
| 0.51 | Dm/r_d | 13.378 | 0.156 | 0.095 | BOSS DR12 — de legacy 1 |
| 0.15 | Dv/r_d | 4.473 | 0.159 | 0 | SDSS DR7 (MGS) — de legacy 1 |
| 0.44 | Dv/r_d | 11.548 | 0.559 | 0 | WiggleZ — de legacy 1 |
| 1.491 | Dv/r_d | 26.07 | 0.67 | 0 | **DESI DR1** (QSO) |
| 0.38 | H·r_d | 12044.07 | 251.226 | 133.002 | BOSS DR12 — de legacy 1 |
| 0.51 | H·r_d | 13374.09 | 251.226 | 147.78 | BOSS DR12 — de legacy 1 |

## `BAO_full_2.csv`

| z_eff | D_M/r_d | D_H/r_d | rho | origen |
|---|---|---|---|---|
| 0.706 | 16.85 ± 0.32 | 20.08 ± 0.6 | -0.42 | **DESI DR1** (LRG2) |
| 0.93 | 21.71 ± 0.28 | 17.88 ± 0.35 | -0.389 | **DESI DR1** (LRG3+ELG1) |
| 1.317 | 27.79 ± 0.69 | 13.82 ± 0.42 | -0.444 | **DESI DR1** (ELG2) |
| 2.33 | 38.8 ± 0.75 | 8.72 ± 0.14 | -0.48 | Ly-alpha — **ni legacy 1 ni DESI** |

## No combinar con DESI DR2

`../DESI/` son los datos de **DESI DR2**, que vuelve a medir los mismos trazadores que DR1.
Como 7 de las 15 medidas de acá **son** DR1, `BAO_legacy_2` y `DESI` no son independientes y no
deben sumarse en un mismo ajuste. Lo mismo vale para `../BAO_legacy_1/`, del que éste toma 6
medidas.

## Referencias

Ver `../BAO_legacy_1/README.md` para los surveys de la parte de bajo z, y
[DESI Collaboration (2024), arXiv:2404.03002](https://arxiv.org/abs/2404.03002) para DR1.
