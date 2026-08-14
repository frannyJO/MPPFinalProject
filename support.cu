#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "support.h"
#include "pv_model.h"

void verify(const double *irradiance, const double *temperature,
            const PanelParams *params, int num_panels, int newton_iters,
            const double *gpu_current, const double *gpu_power,
            double gpu_total_power) {

  const double relativeTolerance = 1e-6;
  const bool check_total_power = (gpu_total_power >= 0.0);

  double *cpu_current = (double *) malloc(sizeof(double) * num_panels);
  double *cpu_power   = (double *) malloc(sizeof(double) * num_panels);

  solve_panel_currents(irradiance, temperature, params, num_panels,
                        newton_iters, cpu_current, cpu_power);

  double cpu_total_power = 0.0;

  for (int i = 0; i < num_panels; ++i) {
    double relativeError = (cpu_current[i] - gpu_current[i]) / cpu_current[i];
    if (relativeError > relativeTolerance
      || relativeError < -relativeTolerance) {
      if (check_total_power) {
        printf("TEST FAILED (panel %d current mismatch)\n\n", i);
      } else {
        printf("TEST FAILED\n\n");
      }
      free(cpu_current);
      free(cpu_power);
      exit(0);
    }
    if (check_total_power) {
      cpu_total_power += cpu_power[i];
    }
  }

  if (check_total_power) {
    double totalRelativeError =
        (cpu_total_power - gpu_total_power) / cpu_total_power;
    if (totalRelativeError > relativeTolerance
      || totalRelativeError < -relativeTolerance) {
      printf("TEST FAILED (total power mismatch: CPU %.6f W vs GPU %.6f W)\n\n",
             cpu_total_power, gpu_total_power);
      free(cpu_current);
      free(cpu_power);
      exit(0);
    }
  }

  free(cpu_current);
  free(cpu_power);

  if (check_total_power) {
    printf("TEST PASSED (total power: %.6f W)\n\n", gpu_total_power);
  } else {
    printf("TEST PASSED\n\n");
  }
}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

