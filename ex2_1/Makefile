CC = gcc
# Flags:
# -O3: Μέγιστη βελτιστοποίηση (απαραίτητο για να δούμε πραγματικό speedup)
# -fopenmp: Ενεργοποίηση OpenMP
# -Wall: Εμφάνιση όλων των warnings (καλή πρακτική)
CFLAGS = -O3 -fopenmp -Wall
TARGET = ex2_1

all: $(TARGET)

$(TARGET): ex2_1.c
	$(CC) $(CFLAGS) -o $(TARGET) ex2_1.c

clean:
	rm -f $(TARGET)