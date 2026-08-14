#ifndef __PV_MODELH__
#define __PV_MODELH__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double IL_ref;   /* reference light-generated current       */
    double I0_ref;   /* reference diode saturation current       */
    double Rs;       /* series resistance (ohm)                             */
    double Rsh;      /* shunt resistance (ohm)                              */
    double n;        /* diode ideality factor                   */
    double Ns;       /* number of cells in series           */
    double alpha_T;  /* temp. coeff. of light current (1/K)                 */
    double Eg;       /* bandgap energy (eV)                                 */
    double G_ref;    /* reference irradiance (W/m^2)      */
    double T_ref;    /* reference temperature (K) */
    double V_op;     /* fixed operating voltage per panel */
} PanelParams;

PanelParams pv_default_params(void);

void solve_panel_currents(const double *irradiance,
                           const double *temperature,
                           const PanelParams *params,
                           int num_panels,
                           int newton_iters,
                           double *out_current,
                           double *out_power);

#ifdef __cplusplus
}
#endif

#endif
