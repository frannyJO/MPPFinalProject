#include <stdio.h>
#include <stdlib.h>

#include "support.h"

int main (int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    double *irradiance_h, *temperature_h, *current_h, *power_h;

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

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);


    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel -----------------------------------------------------------

    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer); 

    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables to host ------------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer); 

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -------------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    // Free memory ---------------------------------------------------------------

    return 0;

}
