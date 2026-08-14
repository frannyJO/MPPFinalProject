
#ifndef __FILEH__
#define __FILEH__

#include <sys/time.h>
#include "pv_model.h"

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

#ifdef __cplusplus
extern "C" {
#endif
void verify(const double *irradiance, const double *temperature,
            const PanelParams *params, int num_panels, int newton_iters,
            const double *gpu_current, const double *gpu_power,
            double gpu_total_power = -1.0);
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);
#ifdef __cplusplus
}
#endif

#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

#endif

