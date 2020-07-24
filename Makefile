CC=gcc
CFLAGS=-std=c99 -g3 -Wall -Wextra
LDFLAGS=-lm
TARGET=neural_test
SRC=neural.c rand_gen.c main.c
OBJ=$(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ): %.o : %.c
	$(CC) $(CFLAGS) -c -o $@ $<

.PHONY: clean

clean:
	$(RM) $(TARGET) *.o
