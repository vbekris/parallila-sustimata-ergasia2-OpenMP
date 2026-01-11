CC = gcc
# Flags:
# -O3: Μέγιστη βελτιστοποίηση (ΚΡΙΣΙΜΟ για να δούμε τη διαφορά CSR vs Dense)
# -fopenmp: Ενεργοποίηση OpenMP
# -Wall: Για να βλέπουμε τυχόν warnings
CFLAGS = -O3 -fopenmp -Wall

# Το όνομα του εκτελέσιμου
TARGET = ex2_2

# Το αρχείο πηγαίου κώδικα
SRC = ex2_2.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)