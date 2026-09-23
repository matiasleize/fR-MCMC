CC ?= cc
CFLAGS ?= -O3 -std=c99 -Wall -Wextra

libfr_ode.so: fr_ode.c fr_ode.h
	$(CC) $(CFLAGS) -shared -fPIC -o $@ fr_ode.c -lm

clean:
	rm -f libfr_ode.so

.PHONY: clean
