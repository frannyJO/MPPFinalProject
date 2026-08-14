#include "pv_model.h"
#include <math.h>

#define BOLTZMANN_K 1.380649e-23    /* J/K   */
#define ELECTRON_Q  1.602176634e-19 /* C     */
#define EV_TO_K     11604.518121    

PanelParams pv_default_params(void) {
    PanelParams p;
    p.IL_ref  = 8.70;
    p.I0_ref  = 1.2e-9;
    p.Rs      = 0.30;
    p.Rsh     = 250.0;
    p.n       = 1.25;
    p.Ns      = 60.0;
    p.alpha_T = 0.0004;
    p.Eg      = 1.12;
    p.G_ref   = 1000.0;
    p.T_ref   = 298.15;
    p.V_op    = 32.0;
    return p;
}

void solve_panel_currents(const double *irradiance,
                           const double *temperature,
                           const PanelParams *params,
                           int num_panels,
                           int newton_iters,
                           double *out_current,
                           double *out_power) {
    const double n   = params->n;
    const double Ns  = params->Ns;
    const double Rs  = params->Rs;
    const double Rsh = params->Rsh;
    const double Vop = params->V_op;
    const double Eg_over_nk = (params->Eg * EV_TO_K) / n;

    for (int i = 0; i < num_panels; i++) {
        const double G = irradiance[i];
        const double T = temperature[i];

        const double Vt_module = n * Ns * (BOLTZMANN_K * T / ELECTRON_Q);

        const double IL = params->IL_ref * (G / params->G_ref) *
                           (1.0 + params->alpha_T * (T - params->T_ref));
        const double I0 = params->I0_ref *
                           pow(T / params->T_ref, 3.0) *
                           exp(Eg_over_nk * (1.0 / params->T_ref - 1.0 / T));

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
}
