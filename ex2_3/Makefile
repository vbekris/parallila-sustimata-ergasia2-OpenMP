CC = gcc
# Flags:
# -O3: Μέγιστη βελτιστοποίηση (Κρίσιμο για Merge Sort)
# -fopenmp: Ενεργοποίηση OpenMP
# -Wall: Προειδοποιήσεις για πιθανά λάθη
CFLAGS = -O3 -fopenmp -Wall

# Το όνομα του εκτελέσιμου
TARGET = ex2_3

# Το αρχείο πηγαίου κώδικα
SRC = ex2_3.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)