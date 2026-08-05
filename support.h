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
