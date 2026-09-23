# Cosmic chronometers (CC)

`chronometers_data.txt`: **33** medidas de H(z) por el método de los cronómetros cósmicos,
en tres columnas separadas por tabulaciones y sin encabezado: `z`, `H(z)` [km/s/Mpc],
`sigma_H(z)` [km/s/Mpc]. Sin líneas de comentario, para que `np.loadtxt` la lea directo.

Fuente: **Tabla 1 de Moresco (2023)**, *Addressing the Hubble tension with cosmic chronometers*,
[arXiv:2307.09501](https://arxiv.org/abs/2307.09501). Los errores tabulados son **sólo la parte
diagonal** de la covarianza; el propio paper avisa que "for a proper analysis also the full
covariance should be considered" (ver `../CC_cov/`).

## Quién lee este archivo

| función | dataset | tratamiento |
|---|---|---|
| `utils/data.read_data_chronometers` | `dataset_CC` | los 33, covarianza diagonal |
| `utils/CC_cov.read_data_CC_cov` | `dataset_CC_cov` | los 33, con la covarianza de Moresco et al. (2020) en los 15 de Moresco BC03 |

Son **dos tratamientos de los mismos datos**, no dos muestras: hay que usar una o la otra,
nunca las dos sumadas.

> **Cuidado: `dataset_CC` sobre esta tabla no corresponde a ninguna configuración publicada.**
>
> | datos + tratamiento | referencia |
> |---|---|
> | `../CC_legacy/` (30) + `dataset_CC` | Leizerovich et al. (2022), PRD 105, 103526 |
> | `CC` (33) + `dataset_CC_cov` | Moresco (2023) Tabla 1 + Moresco et al. (2020) |
> | `CC` (33) + `dataset_CC` | **ninguna** |
>
> La tercera fila queda en el medio por dos razones: el paper del que salen los 33 puntos avisa
> que sus errores son sólo la diagonal y que "for a proper analysis also the full covariance
> should be considered", y además la elección de Jimenez+23 entre las tres medidas de la misma
> muestra es nuestra, así que ese set de 33 no existe publicado como tal.
>
> Sirve como **diagnóstico** —para aislar el efecto de la covarianza sobre los mismos 33
> puntos— pero para producción conviene `dataset_CC_cov` acá, o `dataset_CC` sobre
> `../CC_legacy/` si lo que se quiere es reproducir el paper de 2022.
>
> Los 10 configs con `USE_CC: True` y la entrada `CC` de `run_test_ext.py` apuntan a esta
> carpeta, así que hoy caen en la tercera fila.

## La tabla

`sigma_eff` es la raíz de la diagonal de la covarianza que arma `read_data_CC_cov`; para los 15
de Moresco no coincide con `sigma_H(z)` porque esa función reemplaza el valor y el error por los
de `../CC_cov/HzTable_MM_BC03.dat` (error estadístico + metalicidad) y les suma los términos
de IMF y SPS.

| z | H(z) | sigma_H(z) | método | referencia | ¿covarianza? | sigma_eff |
|---|---|---|---|---|---|---|
| 0.07 | 69 | 19.6 | F | Zhang+14 | — | 19.60 |
| 0.09 | 69 | 12 | F | Simon+05 | — | 12.00 |
| 0.12 | 68.6 | 26.2 | F | Zhang+14 | — | 26.20 |
| 0.17 | 83 | 8 | F | Simon+05 | — | 8.00 |
| 0.1791 | 75 | 4 | D | Moresco+12 | sí | 5.57 |
| 0.1993 | 75 | 5 | D | Moresco+12 | sí | 6.37 |
| 0.2 | 72.9 | 29.6 | F | Zhang+14 | — | 29.60 |
| 0.27 | 77 | 14 | F | Simon+05 | — | 14.00 |
| 0.28 | 88.8 | 36.6 | F | Zhang+14 | — | 36.60 |
| 0.3519 | 83 | 14 | D | Moresco+12 | sí | 14.65 |
| 0.3802 | 83 | 13.5 | D | Moresco+16 | sí | 14.29 |
| 0.4 | 95 | 17 | F | Simon+05 | — | 17.00 |
| 0.4004 | 77 | 10.2 | D | Moresco+16 | sí | 11.12 |
| 0.4247 | 87.1 | 11.2 | D | Moresco+16 | sí | 12.47 |
| 0.4497 | 92.8 | 12.9 | D | Moresco+16 | sí | 14.07 |
| 0.47 | 89 | 49.6 | F | Ratsimbazafy+17 | — | 49.60 |
| 0.4783 | 80.9 | 9 | D | Moresco+16 | sí | 10.23 |
| 0.48 | 97 | 62 | F | Stern+10 | — | 62.00 |
| 0.5929 | 104 | 13 | D | Moresco+12 | sí | 14.04 |
| 0.6797 | 92 | 8 | D | Moresco+12 | sí | 9.51 |
| 0.75 | 105 | 10.76 | ML | Jimenez+23 | — | 10.76 |
| 0.7812 | 105 | 12 | D | Moresco+12 | sí | 13.30 |
| 0.8754 | 125 | 17 | D | Moresco+12 | sí | 17.07 |
| 0.88 | 90 | 40 | F | Stern+10 | — | 40.00 |
| 0.9 | 117 | 23 | F | Simon+05 | — | 23.00 |
| 1.037 | 154 | 20 | D | Moresco+12 | sí | 20.36 |
| 1.26 | 135 | 65 | F | Tomasetti+23 | — | 65.00 |
| 1.3 | 168 | 17 | F | Simon+05 | — | 17.00 |
| 1.363 | 160 | 33.6 | D | Moresco+15 | sí | 32.85 |
| 1.43 | 177 | 18 | F | Simon+05 | — | 18.00 |
| 1.53 | 140 | 14 | F | Simon+05 | — | 14.00 |
| 1.75 | 202 | 40 | F | Simon+05 | — | 40.00 |
| 1.965 | 186.5 | 50.4 | D | Moresco+15 | sí | 49.77 |

Métodos, según la nomenclatura del paper: **F** full-spectrum fitting, **D** D4000,
**L** índices de Lick, **ML** machine learning.

## Referencias

- **Simon+05** — [Simon, Verde & Jimenez (2005), PRD 71, 123001](https://arxiv.org/abs/astro-ph/0412269)
- **Stern+10** — [Stern et al. (2010), JCAP 02, 008](https://arxiv.org/abs/0907.3149)
- **Moresco+12** — [Moresco et al. (2012), JCAP 08, 006](https://arxiv.org/abs/1201.3609)
- **Zhang+14** — [Zhang et al. (2014), RAA 14, 1221](https://arxiv.org/abs/1207.4541)
- **Moresco+15** — [Moresco (2015), MNRAS 450, L16](https://arxiv.org/abs/1503.01116)
- **Moresco+16** — [Moresco et al. (2016), JCAP 05, 014](https://arxiv.org/abs/1601.01701)
- **Ratsimbazafy+17** — [Ratsimbazafy et al. (2017), MNRAS 467, 3239](https://arxiv.org/abs/1702.00418)
- **Jimenez+23** — [Jimenez, Moresco, Verde & Wandelt (2023)](https://arxiv.org/abs/2306.11425)
- **Tomasetti+23** — [Tomasetti et al. (2023)](https://arxiv.org/abs/2305.16387)

Al usar `CC_cov`, el [repositorio de Moresco](https://gitlab.com/mmoresco/CCcovariance) pide citar
además Moresco et al. 2012, 2015 y 2016, que son las medidas que cubre la covarianza.

## Historial y las dos filas que faltan a propósito

Hasta 2026-09-22 el archivo tenía **30** filas. Se agregaron tres de la Tabla 1 que faltaban:

| z | H(z) | sigma | referencia | por qué |
|---|---|---|---|---|
| 0.47 | 89 | 49.6 | Ratsimbazafy+17 | medida independiente, entra directo |
| 1.26 | 135 | 65 | Tomasetti+23 | medida independiente, entra directo |
| 0.75 | 105 | 10.76 | Jimenez+23 | elegida entre tres de la misma muestra (ver abajo) |

**Las dos que quedan fuera, y por qué.** La Tabla 1 tiene 35 filas, pero tres de ellas salen de
la **misma muestra de galaxias** y el paper marca con un asterisco que *"these data have been
obtained from the same sample, and should not be used together in an analysis"*:

| z | H(z) | sigma | método | referencia | |
|---|---|---|---|---|---|
| 0.75 | 98.8 | 33.6 | L | Borghi+22 | excluida |
| 0.75 | 105 | 10.76 | ML | Jimenez+23 | **la que usamos** |
| 0.8 | 113.1 | 25.22 | F | Jiao+23 | excluida |

Se eligió Jimenez+23 por ser la de menor error. Es una elección basada en la precisión del
método, no en el valor medido, pero conviene tenerla presente: es la única de las tres obtenida
con machine learning, y cambiar de elección cambia los resultados.

Por eso el máximo utilizable son **33** puntos y no 35.

Los tres puntos nuevos **no** entran en la covarianza de Moresco et al. (2020), que está definida
sólo para sus propias medidas: quedan diagonales, igual que los de Simon+05, Stern+10 y Zhang+14.

> Después de este cambio hay que **rehacer todas las corridas con CC o CCcov**, que se hicieron
> con 30 puntos. El efecto es chico —ajustando sólo CC con covarianza, H0 pasa de 68.9 +- 4.1 a
> 68.8 +- 4.1 y Omega_m de 0.324 +- 0.063 a 0.323 +- 0.063— porque dos de las tres nuevas tienen
> errores muy grandes.
>
> `~/Documents/PhD/code/Cobaya/data/cc_data/cc_data.txt` es **otra copia** de esta tabla y quedó
> con las 30 filas viejas.

La versión de 30 filas se conservó en **`../CC_legacy/`**, que es la tabla del paper de 2022
([arXiv:2112.01492](https://arxiv.org/abs/2112.01492)) y sirve para reproducir sus resultados.
Es el mismo dato que este archivo tenía antes del cambio, así que **no se puede combinar** con
`CC` ni con `CC_cov`.
