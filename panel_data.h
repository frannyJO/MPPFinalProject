#ifndef __PANEL_DATAH__
#define __PANEL_DATAH__

#ifdef __cplusplus
extern "C" {
#endif

void generate_plant_conditions(int num_panels,
                                unsigned int seed,
                                double *irradiance,
                                double *temperature);

#ifdef __cplusplus
}
#endif

#endif
