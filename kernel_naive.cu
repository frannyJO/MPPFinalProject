#include <stdio.h>
#include "pv_model.h"

#define BOLTZMANN_K 1.380649e-23
#define ELECTRON_Q  1.602176634e-19
#define EV_TO_K     11604.518121

__global__ void pvSolveKernel(int num_panels, int newton_iters, const double *irradiance, const double *temperature, PanelParams params, double *out_current, double *out_power) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_panels) return;

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

    double I = IL;
    for (int k = 0; k < newton_iters; k++) {
        const double expo = exp((Vop + I * Rs) / Vt_module);
        const double f  = IL - I0 * (expo - 1.0) - (Vop + I * Rs) / Rsh - I;
        const double df = -I0 * (Rs / Vt_module) * expo - Rs / Rsh - 1.0;
        I = I - f / df;
    }

    out_current[i] = I;
    out_power[i]   = Vop * I;
}

void solvePVPlant()
{

}
