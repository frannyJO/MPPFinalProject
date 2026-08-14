#include "panel_data.h"
#include <stdlib.h>
#include <math.h>

void generate_plant_conditions(int num_panels,
                                unsigned int seed,
                                double *irradiance,
                                double *temperature) {
    unsigned int s = (seed == 0) ? 42u : seed;
    srand(s);

    const double G_base = 900.0;
    const double G_amp  = 150.0;
    const double T_base = 298.15;

    for (int i = 0; i < num_panels; i++) {
        double phase = (2.0 * M_PI * i) / (num_panels > 1 ? num_panels : 1);
        double gradient = G_amp * sin(phase * 3.0);
        double noise = ((double)rand() / RAND_MAX - 0.5) * 20.0;

        double G = G_base + gradient + noise;
        if (G < 50.0) G = 50.0;

        double T = T_base + 0.03 * (G - G_base) +
                   (((double)rand() / RAND_MAX) - 0.5) * 1.0;

        irradiance[i]  = G;
        temperature[i] = T;
    }
}

