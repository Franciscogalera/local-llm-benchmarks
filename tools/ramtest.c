/* Speichertest ohne Sonderrechte.
 *
 * Belegt moeglichst viel Arbeitsspeicher, schreibt bekannte Muster hinein und
 * liest sie Wort fuer Wort zurueck. Jede Abweichung wird mit Adresse, Sollwert
 * und Istwert gemeldet.
 *
 * Geprueft werden vier Muster: alles Null und alles Eins decken haengende
 * Bits auf, das Schachbrett 0x55/0xAA erzeugt maximale Flankenwechsel auf den
 * Datenleitungen, und ein adressabhaengiges Muster findet Riegel, die Daten
 * an der falschen Stelle ablegen — dabei liest man sonst zufaellig das
 * Richtige zurueck.
 *
 *   ramtest <gigabyte> [durchgaenge]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/mman.h>

#define BLOCK (1024UL * 1024UL * 1024UL)   /* 1 GiB je Block */

static double jetzt(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

/* Sollwert fuer eine Position: konstante Muster oder adressabhaengig */
static inline uint64_t soll(uint64_t muster, uint64_t *p) {
    return muster == 4 ? (uint64_t)(uintptr_t)p : muster;
}

int main(int argc, char **argv) {
    unsigned long gb = argc > 1 ? strtoul(argv[1], NULL, 10) : 8;
    int runden = argc > 2 ? atoi(argv[2]) : 1;

    uint64_t muster[] = {0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL,
                         0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL, 4};
    const char *name[] = {"alles Null", "alles Eins", "Schachbrett 0x55",
                          "Schachbrett 0xAA", "Adressmuster"};

    void **bloecke = calloc(gb, sizeof(void *));
    unsigned long belegt = 0;

    printf("  belege %lu GiB ...\n", gb);
    fflush(stdout);
    for (unsigned long i = 0; i < gb; i++) {
        void *p = mmap(NULL, BLOCK, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        if (p == MAP_FAILED) break;
        bloecke[i] = p;
        belegt++;
    }
    printf("  belegt: %lu GiB\n\n", belegt);
    if (!belegt) { fprintf(stderr, "  kein Speicher bekommen\n"); return 2; }

    unsigned long long fehler = 0;

    for (int r = 1; r <= runden; r++) {
        if (runden > 1) printf("  Durchgang %d von %d\n", r, runden);

        for (unsigned m = 0; m < sizeof(muster) / sizeof(*muster); m++) {
            double t0 = jetzt();

            for (unsigned long b = 0; b < belegt; b++) {
                uint64_t *p = bloecke[b];
                for (unsigned long i = 0; i < BLOCK / 8; i++)
                    p[i] = soll(muster[m], &p[i]);
            }
            double t1 = jetzt();

            unsigned long long schlecht = 0;
            for (unsigned long b = 0; b < belegt; b++) {
                uint64_t *p = bloecke[b];
                for (unsigned long i = 0; i < BLOCK / 8; i++) {
                    uint64_t s = soll(muster[m], &p[i]);
                    if (p[i] != s) {
                        if (schlecht < 8)
                            printf("    FEHLER bei %p: erwartet %016llx, "
                                   "gelesen %016llx\n", (void *)&p[i],
                                   (unsigned long long)s,
                                   (unsigned long long)p[i]);
                        schlecht++;
                    }
                }
            }
            double t2 = jetzt();
            fehler += schlecht;

            printf("    %-18s schreiben %5.1f GB/s   lesen+pruefen %5.1f GB/s"
                   "   Fehler: %llu\n", name[m],
                   belegt / (t1 - t0), belegt / (t2 - t1), schlecht);
            fflush(stdout);
        }
    }

    printf("\n  %llu Fehler insgesamt auf %lu GiB\n", fehler, belegt);
    return fehler ? 1 : 0;
}
