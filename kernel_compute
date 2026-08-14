#include <stdio.h>
#include "pv_model.h"

#define BOLTZMANN_K 1.380649e-23
#define ELECTRON_Q  1.602176634e-19
#define EV_TO_K     11604.518121
#define BLOCK_SIZE  256

__device__ __forceinline__ double newton_step_optimized(
    double I,
    double IL_plus_I0,
    double I0,
    double Vop_inv_Rsh,
    double Rs_inv_Rsh,
    double Vop_inv_Vt,
    double Rs_inv_Vt)
{
    const double expo  = exp(Vop_inv_Vt + I * Rs_inv_Vt);
    const double diode = I0 * expo;

    const double f = IL_plus_I0
                   - diode
                   - Vop_inv_Rsh
                   - I * Rs_inv_Rsh
                   - I;

    const double df = -diode * Rs_inv_Vt
                    - Rs_inv_Rsh
                    - 1.0;

    return I - f / df;
}

__global__ void pvSolveKernelCompute(int num_panels, int newton_iters,
    const double2 * __restrict__ gt, PanelParams params,
    double2 * __restrict__ ip, double *total_power)
{
    __shared__ double s_power[BLOCK_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    double I = 0.0;
    double P = 0.0;

    if (i < num_panels) {
        const double2 gt_i = gt[i];
        const double G = gt_i.x;
        const double T = gt_i.y;

        const double n   = params.n;
        const double Ns  = params.Ns;
        const double Rs  = params.Rs;
        const double Rsh = params.Rsh;
        const double Vop = params.V_op;

        const double inv_Tref = 1.0 / params.T_ref;
        const double inv_T    = 1.0 / T;
        const double inv_Rsh  = 1.0 / Rsh;

        const double thermal_coeff =
            n * Ns * (BOLTZMANN_K / ELECTRON_Q);
        const double inv_Vt = 1.0 / (thermal_coeff * T);

        const double Eg_over_nk = (params.Eg * EV_TO_K) / n;

        const double IL = params.IL_ref * (G / params.G_ref) *
            (1.0 + params.alpha_T * (T - params.T_ref));

        const double T_ratio  = T * inv_Tref;
        const double T_ratio3 = T_ratio * T_ratio * T_ratio;
        const double I0 = params.I0_ref * T_ratio3 *
            exp(Eg_over_nk * (inv_Tref - inv_T));

        // Newton-loop invariants.
        const double Rs_inv_Vt   = Rs  * inv_Vt;
        const double Vop_inv_Vt  = Vop * inv_Vt;
        const double Rs_inv_Rsh  = Rs  * inv_Rsh;
        const double Vop_inv_Rsh = Vop * inv_Rsh;
        const double IL_plus_I0  = IL + I0;

        I = IL;

        if (newton_iters == 8) {
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                I = newton_step_optimized(I, IL_plus_I0, I0,
                    Vop_inv_Rsh, Rs_inv_Rsh, Vop_inv_Vt, Rs_inv_Vt);
            }
        } else {
            for (int k = 0; k < newton_iters; ++k) {
                I = newton_step_optimized(I, IL_plus_I0, I0,
                    Vop_inv_Rsh, Rs_inv_Rsh, Vop_inv_Vt, Rs_inv_Vt);
            }
        }

        P = Vop * I;
        ip[i] = make_double2(I, P);
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

void solvePVPlantCompute(int num_panels, int newton_iters,
    const double2 *gt_d, PanelParams params, double2 *ip_d,
    double *total_power_d)
{
    dim3 dim_grid((num_panels + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 dim_block(BLOCK_SIZE, 1, 1);

    pvSolveKernelCompute<<<dim_grid, dim_block>>>(num_panels, newton_iters,
        gt_d, params, ip_d, total_power_d);
}
