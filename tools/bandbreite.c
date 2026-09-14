/* Speicherbandbreite nach Art von STREAM.
 *
 * Ein einzelner Kern kommt nie an die Grenze des Speichersystems heran — er
 * kann gar nicht genug Anfragen offen halten. Deshalb laufen hier alle Kerne
 * gleichzeitig ueber drei grosse Felder.
 *
 * Gemessen werden vier Zugriffsmuster, weil sie unterschiedlich viel Verkehr
 * erzeugen: Copy und Scale lesen ein Feld und schreiben eins, Add und Triad
 * lesen zwei und schreiben eins. Triad ist der ueblich zitierte Wert.
 *
 * Die Felder werden von demselben Kern beschrieben, der sie spaeter liest
 * (erste Beruehrung parallel). Auf einem Threadripper mit acht Kanaelen
 * entscheidet das darueber, ob der Speicher lokal am richtigen Knoten liegt.
 *
 *   bandbreite [gigabyte_je_feld] [wiederholungen]
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

static double jetzt(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    double gb = argc > 1 ? atof(argv[1]) : 8.0;
    int wdh = argc > 2 ? atoi(argv[2]) : 10;

    size_t n = (size_t)(gb * 1024.0 * 1024.0 * 1024.0 / sizeof(double));
    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *c = malloc(n * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "  zu wenig Speicher\n"); return 2; }

    printf("  %d Kerne, je Feld %.1f GiB, %d Wiederholungen\n\n",
           omp_get_max_threads(), gb, wdh);

    /* Erste Beruehrung parallel — legt den Speicher am richtigen Knoten an */
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }

    const char *name[] = {"Copy   c = a      ", "Scale  b = 3*c    ",
                          "Add    c = a + b  ", "Triad  a = b + 3*c"};
    /* Bytes je Element: gelesen + geschrieben */
    const int bytes[] = {2, 2, 3, 3};
    double best[4] = {0, 0, 0, 0};

    for (int w = 0; w < wdh; w++) {
        for (int k = 0; k < 4; k++) {
            double t0 = jetzt();
            switch (k) {
            case 0:
                #pragma omp parallel for
                for (size_t i = 0; i < n; i++) c[i] = a[i];
                break;
            case 1:
                #pragma omp parallel for
                for (size_t i = 0; i < n; i++) b[i] = 3.0 * c[i];
                break;
            case 2:
                #pragma omp parallel for
                for (size_t i = 0; i < n; i++) c[i] = a[i] + b[i];
                break;
            case 3:
                #pragma omp parallel for
                for (size_t i = 0; i < n; i++) a[i] = b[i] + 3.0 * c[i];
                break;
            }
            double s = jetzt() - t0;
            double gbs = bytes[k] * n * sizeof(double) / s / 1e9;
            if (gbs > best[k]) best[k] = gbs;
        }
    }

    for (int k = 0; k < 4; k++)
        printf("  %s  %7.1f GB/s\n", name[k], best[k]);
    printf("\n  Triad ist der Wert, den man ueblicherweise vergleicht.\n");
    return 0;
}
