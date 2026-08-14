#include <stdio.h>
#include "pv_model.h"

#define BOLTZMANN_K 1.380649e-23
#define ELECTRON_Q  1.602176634e-19
#define EV_TO_K     11604.518121
#define BLOCK_SIZE  256

__global__ void pvSolveKernelShared(int num_panels, int newton_iters,
    const double *irradiance, const double *temperature,
    PanelParams params, double *out_current, double *out_power,
    double *total_power) {

    __shared__ double s_power[BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double I = 0.0, P = 0.0;

    if (i < num_panels) {
        const double n   = params.n;
        const double Ns  = params.Ns;
        const double Rs  = params.Rs;
        const double Rsh = params.Rsh;
        const double Vop = params.V_op;
        const double Eg_over_nk = (params.Eg * EV_TO_K) / n;

        const double G = irradiance[i];
        const double T = temperature[i];

        const double Vt_module = n * Ns * (BOLTZMANN_K * T / ELECTRON_Q);

        const double IL = params.IL_ref * (G / params.G_ref) *
                           (1.0 + params.alpha_T * (T - params.T_ref));
        const double I0 = params.I0_ref *
                           pow(T / params.T_ref, 3.0) *
                           exp(Eg_over_nk * (1.0 / params.T_ref - 1.0 / T));

        I = IL;
        for (int k = 0; k < newton_iters; k++) {
            const double expo = exp((Vop + I * Rs) / Vt_module);
            const double f  = IL - I0 * (expo - 1.0) - (Vop + I * Rs) / Rsh - I;
            const double df = -I0 * (Rs / Vt_module) * expo - Rs / Rsh - 1.0;
            I = I - f / df;
        }

        P = Vop * I;
        out_current[i] = I;
        out_power[i]   = P;
    }

    s_power[threadIdx.x] = P;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_power[threadIdx.x] += s_power[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(total_power, s_power[0]);
    }
}

void solvePVPlantShared(int num_panels, int newton_iters,
    const double *irradiance_d, const double *temperature_d,
    PanelParams params, double *current_d, double *power_d,
    double *total_power_d)
{
    // Initialize thread block and kernel grid dimensions 

    dim3 dim_grid((num_panels + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 dim_block(BLOCK_SIZE, 1, 1);

    // Invoke CUDA kernel 

    pvSolveKernelShared<<<dim_grid, dim_block>>>(num_panels, newton_iters,
        irradiance_d, temperature_d, params, current_d, power_d, total_power_d);
}

