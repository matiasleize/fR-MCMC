# BAO legacy 1 (`BAO_legacy_1`)

Las **20** medidas de BAO pre-DESI usadas en Leizerovich, Kraiselburd, Landau & Scóccola (2022),
[Phys. Rev. D 105, 103526](https://doi.org/10.1103/PhysRevD.105.103526)
([arXiv:2112.01492](https://arxiv.org/abs/2112.01492)). **Ningún punto viene de DESI.**

Antes esta carpeta se llamaba `BAO/`; se renombró el 2026-09-22, sin tocar los archivos.

## Archivos

Cinco tablas, una por observable, con encabezado y columnas
`z`, `Dist`, `Stat_error`, `Sist_error`, `wb_fid`, `type`:

| archivo | observable | índice en `BAO.Hs_to_Ds` | puntos |
|---|---|---|---|
| `BAO_data_da.txt` | D_A/r_d | 0 | 1 |
| `BAO_data_dh.txt` | D_H/r_d | 1 | 4 |
| `BAO_data_dm.txt` | D_M/r_d | 2 | 7 |
| `BAO_data_dv.txt` | D_V/r_d | 3 | 5 |
| `BAO_data_H.txt` | H·r_d [km/s] | 4 | 3 |

Se leen con `utils/data.read_data_BAO`, que devuelve `z`, el valor y
`Stat_error**2 + Sist_error**2`. **El orden importa**: `chi_square.py` asume la lista
`[da, dh, dm, dv, H]` para que el índice coincida con `Hs_to_Ds`.

> **Antes de usar este dataset, leer `rd-roto`**: la rama `dataset_BAO` de `chi_square.py`
> pisa el r_d de CLASS con `r_drag(omega_m, H_0, bao_param)`, donde `bao_param` es r_d en Mpc
> pero entra en el lugar de omega_b. Con `bao_param = 147` da r_d ≈ 5 Mpc en vez de ≈ 146.
> Hoy sólo `config_LCDM_4p_PPS+CC+BAO.yml` tiene `USE_BAO: True`.

La columna `wb_fid` está en los archivos pero `read_data_BAO` no la lee.

## La tabla

| z | observable | valor | error est. | error sist. | survey |
|---|---|---|---|---|---|
| 0.81 | Da | 10.75 | 0.43 | 0 | DES Y1 |
| 0.698 | Dh | 19.77 | 0.47 | 0 | eBOSS LRG |
| 1.48 | Dh | 13.23 | 0.47 | 0 | eBOSS QSO (aniso) |
| 2.33 | Dh | 9.07 | 0.31 | 0 | Lya auto |
| 2.4 | Dh | 8.94 | 0.22 | 0 | Lya x QSO |
| 0.38 | Dm | 10.272 | 0.135 | 0.074 | BOSS DR12 |
| 0.51 | Dm | 13.378 | 0.156 | 0.095 | BOSS DR12 |
| 0.61 | Dm | 15.449 | 0.189 | 0.108 | BOSS DR12 |
| 0.698 | Dm | 17.65 | 0.3 | 0 | eBOSS LRG |
| 1.48 | Dm | 30.21 | 0.79 | 0 | eBOSS QSO (aniso) |
| 2.33 | Dm | 37.77 | 2.13 | 0 | Lya auto |
| 2.4 | Dm | 36.6 | 1.2 | 0 | Lya x QSO |
| 0.15 | Dv | 4.473 | 0.159 | 0 | SDSS DR7 (MGS) |
| 0.44 | Dv | 11.548 | 0.559 | 0 | WiggleZ |
| 0.6 | Dv | 14.946 | 0.68 | 0 | WiggleZ |
| 0.73 | Dv | 16.931 | 0.579 | 0 | WiggleZ |
| 1.52 | Dv | 26.005 | 0.995 | 0 | eBOSS QSO (Dv) |
| 0.38 | H·r_d | 12044.07 | 251.226 | 133.002 | BOSS DR12 |
| 0.51 | H·r_d | 13374.09 | 251.226 | 147.78 | BOSS DR12 |
| 0.61 | H·r_d | 14378.994 | 266.004 | 162.558 | BOSS DR12 |

## Referencias

- **SDSS DR7 (MGS)** — [Ross et al. (2015), MNRAS 449, 835](https://doi.org/10.1093/mnras/stv154)
- **WiggleZ** — [Kazin et al. (2014), MNRAS 441, 3524](https://doi.org/10.1093/mnras/stu778)
- **BOSS DR12** — [Alam et al. (2017), MNRAS 470, 2617](https://doi.org/10.1093/mnras/stx721)
- **DES Y1** — [Abbott et al. (2018), MNRAS 483, 4866](https://doi.org/10.1093/mnras/sty3351)
- **eBOSS LRG** — [Bautista et al. (2020), MNRAS 500, 736](https://doi.org/10.1093/mnras/staa2800)
- **eBOSS QSO (Dv)** — [Ata et al. (2017), MNRAS 473, 4773](https://doi.org/10.1093/mnras/stx2630)
- **eBOSS QSO (aniso)** — [Neveux et al. (2020), MNRAS 499, 210](https://arxiv.org/abs/2007.08999)
- **Lya auto** — [Bautista et al. (2017), A&A 603, A12](https://doi.org/10.1051/0004-6361/201730533)
- **Lya x QSO** — [du Mas des Bourboux et al. (2017), A&A 608, A130](https://doi.org/10.1051/0004-6361/201731731)

## Relación con los otros datasets de BAO

`../BAO_legacy_2/` toma **6** de estas 20 medidas (las de z ≤ 0.51) y reemplaza el resto por
DESI DR1. `../DESI/` son los datos de DESI DR2. Los tres se solapan entre sí: usar uno solo.
