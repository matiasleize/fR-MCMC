# Cosmic chronometers, versión legacy (`CC_legacy`)

`chronometers_data.txt`: las **30** medidas de H(z) de cronómetros cósmicos usadas en
Leizerovich, Kraiselburd, Landau & Scóccola, *Testing f(R) gravity models with quasar x-ray and
UV fluxes*, [Phys. Rev. D 105, 103526 (2022)](https://doi.org/10.1103/PhysRevD.105.103526)
([arXiv:2112.01492](https://arxiv.org/abs/2112.01492)). Mismo formato que `../CC/`: tres columnas
separadas por tabulaciones, sin encabezado, `z`, `H(z)` [km/s/Mpc], `sigma_H(z)` [km/s/Mpc].

**Sin covarianza.** Este dataset es diagonal por construcción: son los errores tal como los
publicaron los papers originales, sin el presupuesto de sistemáticas de Moresco et al. (2020).
Existe para poder reproducir los resultados de ese paper, no para análisis nuevos.

## Diferencia con `../CC/`

`../CC/chronometers_data.txt` **era** exactamente esta tabla hasta el 2026-09-22, cuando se
amplió a 33 puntos con la compilación de Moresco (2023). Este archivo es una copia de aquella
versión, fila por fila y en el mismo orden.

| | `CC_legacy` | `CC` |
|---|---|---|
| puntos | 30 | 33 |
| covarianza disponible | no | sí, para 15 de ellos (`../CC_cov/`) |
| fuente | tabla del paper de 2022 | Tabla 1 de [Moresco (2023)](https://arxiv.org/abs/2307.09501) |
| agregados en `CC` | — | Ratsimbazafy+17 (z=0.47), Jimenez+23 (z=0.75), Tomasetti+23 (z=1.26) |

## Cómo usarlo

No hace falta código nuevo: `utils/data.read_data_chronometers` lee este archivo igual que el de
`../CC/`, y el chi cuadrado diagonal de `chi_square.py` (`dataset_CC`) lo trata como corresponde.

```python
os.chdir(os.path.join(path_data, 'CC_legacy'))
ds_CC_legacy = read_data_chronometers('chronometers_data.txt')
```

No pasarlo por `utils/CC_cov.read_data_CC_cov`: esa función reemplaza los 15 puntos de Moresco
por los de `../CC_cov/HzTable_MM_BC03.dat` y les agrega las sistemáticas, que es justo lo que
este dataset no quiere tener.

**No combinarlo con `CC` ni con `CC_cov`**: son los mismos datos y se contarían dos veces.

## La tabla

| z | H(z) | sigma_H(z) | referencia |
|---|---|---|---|
| 0.07 | 69 | 19.6 | Zhang+14 |
| 0.09 | 69 | 12 | Simon+05 |
| 0.12 | 68.6 | 26.2 | Zhang+14 |
| 0.17 | 83 | 8 | Simon+05 |
| 0.1791 | 75 | 4 | Moresco+12 |
| 0.1993 | 75 | 5 | Moresco+12 |
| 0.2 | 72.9 | 29.6 | Zhang+14 |
| 0.27 | 77 | 14 | Simon+05 |
| 0.28 | 88.8 | 36.6 | Zhang+14 |
| 0.3519 | 83 | 14 | Moresco+12 |
| 0.3802 | 83 | 13.5 | Moresco+16 |
| 0.4 | 95 | 17 | Simon+05 |
| 0.4004 | 77 | 10.2 | Moresco+16 |
| 0.4247 | 87.1 | 11.2 | Moresco+16 |
| 0.4497 | 92.8 | 12.9 | Moresco+16 |
| 0.4783 | 80.9 | 9 | Moresco+16 |
| 0.48 | 97 | 62 | Stern+10 |
| 0.5929 | 104 | 13 | Moresco+12 |
| 0.6797 | 92 | 8 | Moresco+12 |
| 0.7812 | 105 | 12 | Moresco+12 |
| 0.8754 | 125 | 17 | Moresco+12 |
| 0.88 | 90 | 40 | Stern+10 |
| 0.9 | 117 | 23 | Simon+05 |
| 1.037 | 154 | 20 | Moresco+12 |
| 1.3 | 168 | 17 | Simon+05 |
| 1.363 | 160 | 33.6 | Moresco+15 |
| 1.43 | 177 | 18 | Simon+05 |
| 1.53 | 140 | 14 | Simon+05 |
| 1.75 | 202 | 40 | Simon+05 |
| 1.965 | 186.5 | 50.4 | Moresco+15 |

## Referencias

- **Simon+05** (9 puntos) — [Simon, Verde & Jimenez (2005), PRD 71, 123001](https://arxiv.org/abs/astro-ph/0412269)
- **Stern+10** (2 puntos) — [Stern et al. (2010), JCAP 02, 008](https://arxiv.org/abs/0907.3149)
- **Moresco+12** (8 puntos) — [Moresco et al. (2012), JCAP 08, 006](https://arxiv.org/abs/1201.3609)
- **Zhang+14** (4 puntos) — [Zhang et al. (2014), RAA 14, 1221](https://arxiv.org/abs/1207.4541)
- **Moresco+15** (2 puntos) — [Moresco (2015), MNRAS 450, L16](https://arxiv.org/abs/1503.01116)
- **Moresco+16** (5 puntos) — [Moresco et al. (2016), JCAP 05, 014](https://arxiv.org/abs/1601.01701)
