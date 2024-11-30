Here is the protocol of the free and fixed parameters:

```
LCDM:
index     free parameters             fixed parameters     

4         Mabs, rd, Omega_m, H_0      _

31        Mabs, Omega_m, H_0          rd
32        rd, Omega_m, H_0            Mabs
21        Omega_m, H_0                Mabs, rd


HS-ST-EXP:
index     free parameters             fixed parameters     

5         Mabs, rd, Omega_m, b, H_0   _

41        Mabs, rd, b, H_0            Omega_m
42        Mabs, Omega_m, b, H_0       rd
43        rd, Omega_m, b, H_0         Mabs

31        Mabs, b, H_0                rd, Omega_m
32        rd, b, H_0                  Mabs, Omega_m
33        Omega_m, b, H_0             Mabs, rd
```