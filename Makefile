CC = cc
CFLAGS = -O2 -std=c11 -Wall
LDFLAGS = -lm

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
	LDFLAGS += -framework Accelerate
	CFLAGS += -DUSE_BLAS -DACCELERATE
else
	LDFLAGS += -lopenblas
	CFLAGS += -DUSE_BLAS
endif

nanollama: nanollama.c notorch.c notorch.h
	$(CC) $(CFLAGS) -o $@ nanollama.c notorch.c $(LDFLAGS)

clean:
	rm -f nanollama nanollama_ckpt.bin nanollama_final.bin

.PHONY: clean
