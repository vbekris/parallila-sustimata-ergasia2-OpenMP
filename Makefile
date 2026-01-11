CC = gcc
# Flags:
# -O3: Μέγιστη βελτιστοποίηση (απαραίτητο για να δούμε πραγματικό speedup)
# -fopenmp: Ενεργοποίηση OpenMP
# -Wall: Εμφάνιση όλων των warnings (καλή πρακτική)
CFLAGS = -O3 -fopenmp -Wall
TARGET = poly_mult

all: $(TARGET)

$(TARGET): poly_mult.c
	$(CC) $(CFLAGS) -o $(TARGET) poly_mult.c

clean:
	rm -f $(TARGET)