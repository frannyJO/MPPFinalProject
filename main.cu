#include <stdio.h>
#include <stdlib.h>

#if defined(KERNEL_COMPUTE)
#include "kernel_compute.cu"
#elif defined(KERNEL_COALESCED)
#include "kernel_coalesced.cu"
#elif defined(KERNEL_SHARED)
#include "kernel_shared.cu"
#else
#include "kernel_naive.cu"
#endif

#include "panel_data.h"
#include "support.h"

int main (int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    double *irradiance_h, *temperature_h, *current_h, *power_h;
#if defined(KERNEL_COALESCED) || defined(KERNEL_COMPUTE)
    double2 *gt_h, *ip_h;
    double2 *gt_d, *ip_d;
#else
    double *irradiance_d, *temperature_d, *current_d, *power_d;
#endif
#if defined(KERNEL_SHARED) || defined(KERNEL_COALESCED) || defined(KERNEL_COMPUTE)
    double total_power_h = 0.0;
    double *total_power_d;
#endif
    size_t panel_sz;
    int num_panels, newton_iters;
    PanelParams params;

    if (argc == 1) {
        num_panels = 50000;
        newton_iters = 8;
    } else if (argc == 2) {
        num_panels = atoi(argv[1]);
        newton_iters = 8;
    } else if (argc == 3) {
        num_panels = atoi(argv[1]);
        newton_iters = atoi(argv[2]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: %s                   # 50000 panels, 8 Newton iterations"
           "\n    Usage: %s <panels>           # <panels> panels, 8 Newton iterations"
           "\n    Usage: %s <panels> <iters>   # <panels> panels, <iters> Newton iterations"
           "\n", argv[0], argv[0], argv[0]);
        exit(0);
    }

    panel_sz = (size_t) num_panels;
    params = pv_default_params();

    irradiance_h  = (double*) malloc( sizeof(double)*panel_sz );
    temperature_h = (double*) malloc( sizeof(double)*panel_sz );
    current_h     = (double*) malloc( sizeof(double)*panel_sz );
    power_h       = (double*) malloc( sizeof(double)*panel_sz );
#if defined(KERNEL_COALESCED) || defined(KERNEL_COMPUTE)
    gt_h = (double2*) malloc( sizeof(double2)*panel_sz );
    ip_h = (double2*) malloc( sizeof(double2)*panel_sz );
#endif

    generate_plant_conditions(num_panels, 42u, irradiance_h, temperature_h);

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    panels: %d\n    newton_iters: %d\n", num_panels, newton_iters);

#if defined(KERNEL_COALESCED) || defined(KERNEL_COMPUTE)
    printf("Packing interleaved input array..."); fflush(stdout);
    startTime(&timer);

    for (int i = 0; i < num_panels; i++) {
        gt_h[i] = make_double2(irradiance_h[i], temperature_h[i]);
    }

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
#endif

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

#if defined(KERNEL_COALESCED) || defined(KERNEL_COMPUTE)
    cuda_ret = cudaMalloc((void**)&gt_d, sizeof(double2)*panel_sz);
    if (cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&ip_d, sizeof(double2)*panel_sz);
    if (cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&total_power_d, sizeof(double));
    if (cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
#else
    cuda_ret = cudaMalloc((void**)&irradiance_d, sizeof(double)*panel_sz);
    if (cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&temperature_d, sizeof(double)*panel_sz);
    if (cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&current_d, sizeof(double)*panel_sz);
    if (cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&power_d, sizeof(double)*panel_sz);
    if (cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
  #if defined(KERNEL_SHARED)
    cuda_ret = cudaMalloc((void**)&total_power_d, sizeof(double));
    if (cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
  #endif
#endif

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

#if defined(KERNEL_COALESCED) || defined(KERNEL_COMPUTE)
    cuda_ret = cudaMemcpy(gt_d, gt_h, sizeof(double2)*panel_sz,
        cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess) FATAL("Unable to copy to device");
    // Zero the accumulator before the kernel's atomicAdds start.
    cuda_ret = cudaMemset(total_power_d, 0, sizeof(double));
    if (cuda_ret != cudaSuccess) FATAL("Unable to zero device accumulator");
#else
    cuda_ret = cudaMemcpy(irradiance_d, irradiance_h, sizeof(double)*panel_sz,
        cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess) FATAL("Unable to copy to device");
    cuda_ret = cudaMemcpy(temperature_d, temperature_h, sizeof(double)*panel_sz,
        cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess) FATAL("Unable to copy to device");
  #if defined(KERNEL_SHARED)
    // Zero the accumulator before the kernel's atomicAdds start.
    cuda_ret = cudaMemset(total_power_d, 0, sizeof(double));
    if (cuda_ret != cudaSuccess) FATAL("Unable to zero device accumulator");
  #endif
#endif

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel -----------------------------------------------------------

    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

#if defined(KERNEL_COMPUTE)
    solvePVPlantCompute(num_panels, newton_iters, gt_d, params, ip_d, \
        total_power_d);
#elif defined(KERNEL_COALESCED)
    solvePVPlantCoalesced(num_panels, newton_iters, gt_d, params, ip_d, \
        total_power_d);
#elif defined(KERNEL_SHARED)
    solvePVPlantShared(num_panels, newton_iters, irradiance_d, temperature_d, \
        params, current_d, power_d, total_power_d);
#else
    solvePVPlant(num_panels, newton_iters, irradiance_d, temperature_d, \
        params, current_d, power_d);
#endif

    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables to host ------------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

#if defined(KERNEL_COALESCED) || defined(KERNEL_COMPUTE)
    cuda_ret = cudaMemcpy(ip_h, ip_d, sizeof(double2)*panel_sz,
        cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess) FATAL("Unable to copy from device");
    cuda_ret = cudaMemcpy(&total_power_h, total_power_d, sizeof(double),
        cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess) FATAL("Unable to copy from device");
#else
    cuda_ret = cudaMemcpy(current_h, current_d, sizeof(double)*panel_sz,
        cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess) FATAL("Unable to copy from device");
    cuda_ret = cudaMemcpy(power_h, power_d, sizeof(double)*panel_sz,
        cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess) FATAL("Unable to copy from device");
  #if defined(KERNEL_SHARED)
    cuda_ret = cudaMemcpy(&total_power_h, total_power_d, sizeof(double),
        cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess) FATAL("Unable to copy from device");
  #endif
#endif

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

#if defined(KERNEL_COALESCED) || defined(KERNEL_COMPUTE)
    printf("Unpacking interleaved output array..."); fflush(stdout);
    startTime(&timer);

    for (int i = 0; i < num_panels; i++) {
        current_h[i] = ip_h[i].x;
        power_h[i]   = ip_h[i].y;
    }

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
#endif

#if defined(KERNEL_SHARED) || defined(KERNEL_COALESCED) || defined(KERNEL_COMPUTE)
    printf("    total power (GPU-reduced): %.3f kW\n", total_power_h / 1000.0);
#endif

    // Verify correctness -------------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

#if defined(KERNEL_SHARED) || defined(KERNEL_COALESCED) || defined(KERNEL_COMPUTE)
    verify(irradiance_h, temperature_h, &params, num_panels, newton_iters,
        current_h, power_h, total_power_h);
#else
    verify(irradiance_h, temperature_h, &params, num_panels, newton_iters,
        current_h, power_h);
#endif

    // Free memory ---------------------------------------------------------------

    free(irradiance_h);
    free(temperature_h);
    free(current_h);
    free(power_h);
#if defined(KERNEL_COALESCED) || defined(KERNEL_COMPUTE)
    free(gt_h);
    free(ip_h);
#endif

#if defined(KERNEL_COALESCED) || defined(KERNEL_COMPUTE)
    cudaFree(gt_d);
    cudaFree(ip_d);
    cudaFree(total_power_d);
#else
    cudaFree(irradiance_d);
    cudaFree(temperature_d);
    cudaFree(current_d);
    cudaFree(power_d);
  #if defined(KERNEL_SHARED)
    cudaFree(total_power_d);
  #endif
#endif

    return 0;

}

