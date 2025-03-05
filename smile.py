import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import erf, sqrt, exp, log, pi

# =============================================================================
# 1. PARÁMETROS DE LA SIMULACIÓN
# =============================================================================
np.random.seed(42)   # Semilla para reproducibilidad

hedge_frequency = 5

T         = 1.0      # 1 año
N_steps   = 252      # ~252 días
dt        = T / N_steps
S0        = 100.0    # Spot inicial
r         = 0.01     # Tasa libre de riesgo
mu        = 0.06     # Drift real
sigma_real= 0.60     # Vol. real
jump_prob = 0.01     # Probabilidad diaria de salto
jump_mean = -0.02    # Media lognormal del salto
jump_std  = 0.08     # Desv. lognormal del salto

# Payoff exótico: paga 1 si S_T < K1 o S_T > K2
K1        = 95.0
K2        = 105.0

# Para replicar cada digital (put en K1, call en K2) usamos spreads
# con anchura +/- epsilon
epsilon_put = 0.5
epsilon_call = 0.5

# Volatilidad implícita que usamos en BSM (mismatch con la real)
sigma_imp = 0.25

# Costo transaccional
transaction_cost_rate = 0.000005  # 0.1%

# =============================================================================
# 2. FUNCIONES AUXILIARES: BSM
#    (Para calls, puts, delta, gamma)
# =============================================================================
def Phi(x):
    """CDF normal estándar."""
    return 0.5*(1.0 + erf(x/np.sqrt(2.0)))

def bsm_d1(S, K, r, sigma, tau):
    if tau <= 0.0:
        return 0.0
    return (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))

def bsm_d2(S, K, r, sigma, tau):
    return bsm_d1(S, K, r, sigma, tau) - sigma*np.sqrt(tau)

def bsm_call_price(S, K, r, sigma, tau):
    """Precio de call BSM."""
    if tau <= 0:
        return max(S-K, 0)
    d1_ = bsm_d1(S, K, r, sigma, tau)
    d2_ = bsm_d2(S, K, r, sigma, tau)
    return S*Phi(d1_) - K*np.exp(-r*tau)*Phi(d2_)

def bsm_put_price(S, K, r, sigma, tau):
    """Precio de put BSM."""
    if tau <= 0:
        return max(K-S, 0)
    d1_ = bsm_d1(S, K, r, sigma, tau)
    d2_ = bsm_d2(S, K, r, sigma, tau)
    return K*np.exp(-r*tau)*Phi(-d2_) - S*Phi(-d1_)

def bsm_call_delta(S, K, r, sigma, tau):
    """Delta de call BSM."""
    if tau <= 0:
        return 1.0 if S>K else 0.0
    d1_ = bsm_d1(S, K, r, sigma, tau)
    return Phi(d1_)

def bsm_put_delta(S, K, r, sigma, tau):
    """Delta de put BSM."""
    if tau <= 0:
        return -1.0 if S<K else 0.0
    d1_ = bsm_d1(S, K, r, sigma, tau)
    return Phi(d1_) - 1.0  # delta put = N(d1)-1

def bsm_vega(S, K, r, sigma, tau):
    """Vega = dPrice/dSigma (no la usamos mucho aquí, pero a veces útil)."""
    # vega = S * sqrt(tau) * phi(d1)
    pass

def bsm_call_gamma(S, K, r, sigma, tau):
    """Gamma de call (o put) BSM, es la misma para calls/puts."""
    if tau <= 0 or S<=0:
        return 0.0
    from math import exp, sqrt, pi
    d1_ = bsm_d1(S, K, r, sigma, tau)
    # densidad normal
    phi_ = np.exp(-0.5*d1_**2)/np.sqrt(2.0*np.pi)
    return phi_ / (S*sigma*np.sqrt(tau))

def bsm_put_gamma(S, K, r, sigma, tau):
    """Gamma de put (idéntico a call, BSM)."""
    return bsm_call_gamma(S, K, r, sigma, tau)

# =============================================================================
# 3. DIGITAL PUT Y CALL REPLICADOS CON SPREADS
# =============================================================================
def replicate_digital_put_spreads(S, K, t, T, r, sigma_imp, eps):
    """
    Aproxima la digital put (paga 1 si S_T < K) con put spreads.
    Retorna (value, delta, gamma).
    """
    tau = max(T - t, 0.0)
    if tau <= 0:
        payoff = 1.0 if S < K else 0.0
        return (payoff, 0.0, 0.0)

    # Spread: (Put(K1) - Put(K2)) / (K2-K1)
    # Elegimos K1 = K - eps, K2 = K + eps
    K1 = K - eps
    K2 = K + eps
    p1 = bsm_put_price(S, K1, r, sigma_imp, tau)
    p2 = bsm_put_price(S, K2, r, sigma_imp, tau)
    d1_ = bsm_put_delta(S, K1, r, sigma_imp, tau)
    d2_ = bsm_put_delta(S, K2, r, sigma_imp, tau)
    g1_ = bsm_put_gamma(S, K1, r, sigma_imp, tau)
    g2_ = bsm_put_gamma(S, K2, r, sigma_imp, tau)

    spread_val   = (p1 - p2)/(K2 - K1)
    spread_delta = (d1_ - d2_)/(K2 - K1)
    spread_gamma = (g1_ - g2_)/(K2 - K1)

    return (spread_val, spread_delta, spread_gamma)

def replicate_digital_call_spreads(S, K, t, T, r, sigma_imp, eps):
    """
    Aproxima la digital call (paga 1 si S_T > K) con call spreads.
    Retorna (value, delta, gamma).
    """
    tau = max(T - t, 0.0)
    if tau <= 0:
        payoff = 1.0 if S > K else 0.0
        return (payoff, 0.0, 0.0)

    K1 = K - eps
    K2 = K + eps
    c1 = bsm_call_price(S, K1, r, sigma_imp, tau)
    c2 = bsm_call_price(S, K2, r, sigma_imp, tau)
    d1_ = bsm_call_delta(S, K1, r, sigma_imp, tau)
    d2_ = bsm_call_delta(S, K2, r, sigma_imp, tau)
    g1_ = bsm_call_gamma(S, K1, r, sigma_imp, tau)
    g2_ = bsm_call_gamma(S, K2, r, sigma_imp, tau)

    spread_val   = (c1 - c2)/(K2 - K1)
    spread_delta = (d1_ - d2_)/(K2 - K1)
    spread_gamma = (g1_ - g2_)/(K2 - K1)

    return (spread_val, spread_delta, spread_gamma)

def replicate_outside_range_digital(S, t, T, r, sigma_imp,
                                    K1, eps_put, K2, eps_call):
    """
    Payoff exótico: 1 si S_T < K1 o S_T > K2, 0 en caso contrario.
    Se compone de:
      - Digital Put en K1 (replicado con put spreads)
      - Digital Call en K2 (replicado con call spreads)
    Retorna:
      - value_total
      - delta_total
      - gamma_total
      - (value_put, delta_put, gamma_put)
      - (value_call, delta_call, gamma_call)
    """
    val_put, d_put, g_put = replicate_digital_put_spreads(S, K1, t, T, r, sigma_imp, eps_put)
    val_call, d_call, g_call = replicate_digital_call_spreads(S, K2, t, T, r, sigma_imp, eps_call)

    val_tot   = val_put + val_call
    delta_tot = d_put + d_call
    gamma_tot = g_put + g_call

    return val_tot, delta_tot, gamma_tot, (val_put, d_put, g_put), (val_call, d_call, g_call)

# =============================================================================
# 4. SIMULACIÓN DEL SUBYACENTE CON SALTOS (JUMP DIFFUSION)
# =============================================================================
N = N_steps
S_path = np.zeros(N+1)
S_path[0] = S0

for i in range(N):
    Z = np.random.randn()
    drift_part = (mu - 0.5*sigma_real**2)*dt
    diff_part  = sigma_real*np.sqrt(dt)*Z

    jump_factor = 1.0
    if np.random.rand() < jump_prob:
        jump_factor = np.exp(np.random.normal(jump_mean, jump_std))

    S_path[i+1] = S_path[i]*np.exp(drift_part + diff_part)*jump_factor

# =============================================================================
# 5. COBERTURA DINÁMICA Y CÁLCULO DE P&L
#    Basado en ecuaciones del cap. 5 (p.ej. (5.22)/(5.26)/(5.27) en forma discreta)
# =============================================================================
t_grid = np.linspace(0, T, N+1)

# Arrays para guardar la evolución
Portfolio_value     = np.zeros(N+1)
Delta_portfolio     = np.zeros(N+1)
Gamma_portfolio     = np.zeros(N+1)
Put_leg_value       = np.zeros(N+1)
Put_leg_delta       = np.zeros(N+1)
Put_leg_gamma       = np.zeros(N+1)
Call_leg_value      = np.zeros(N+1)
Call_leg_delta      = np.zeros(N+1)
Call_leg_gamma      = np.zeros(N+1)
PnL                 = np.zeros(N+1)

# Valor inicial (outside range digital)
(val_0, delta_0, gamma_0,
 (val_put0, d_put0, g_put0),
 (val_call0, d_call0, g_call0)
 ) = replicate_outside_range_digital(S_path[0], 0.0, T, r, sigma_imp,
                                     K1, epsilon_put, K2, epsilon_call)

Portfolio_value[0] = val_0
Delta_portfolio[0] = delta_0
Gamma_portfolio[0] = gamma_0
Put_leg_value[0]   = val_put0
Put_leg_delta[0]   = d_put0
Put_leg_gamma[0]   = g_put0
Call_leg_value[0]  = val_call0
Call_leg_delta[0]  = d_call0
Call_leg_gamma[0]  = g_call0

# Iniciamos la estrategia: compramos la cartera exótica (pagamos val_0),
# vendemos delta_0 en el subyacente, y pagamos costo transaccional:
PnL[0] = -val_0 + delta_0*S_path[0]
cost_0 = transaction_cost_rate * abs(delta_0*S_path[0])
PnL[0] -= cost_0

for i in range(1, N_steps+1):
    # 1) El portfolio anterior sube/baja segun S_path
    val_prev   = Portfolio_value[i-1]
    delta_prev = Delta_portfolio[i-1]
    
    # --- Actualiza el valor del portfolio con la misma DELTA que teníamos ---
    #    (Es decir, sumamos dPnL = (Val_i - Val_{i-1}) - delta_prev*(S_i - S_{i-1}))
    #    Pero SOLO calculamos "Val_i" con la formula si rehedgeamos. 
    #    Si NO rehedgeamos, repetimos la val anterior
    if i % hedge_frequency == 0:
        # Reevaluamos la opción exótica y su delta
        val_i, delta_i, gamma_i, _, _ = replicate_outside_range_digital(
            S_path[i], 
            i*dt, 
            T, 
            r, 
            sigma_imp, 
            K1, 
            epsilon_put, 
            K2, 
            epsilon_call
        )
    else:
        # Mantenemos la misma delta y valor que el día anterior (aunque "valor" teórico
        # podría cambiar, asumamos no recalcularlo. O recalculamos solo "mark to market"
        # sin re-hedge. A ti te toca decidir la lógica.)
        val_i = val_prev  # o, si quieres, revalúas la opción sin cambiar la delta
        delta_i = delta_prev
    
    Portfolio_value[i] = val_i
    
    # 2) Calcula la P&L incremental
    dVal   = val_i - val_prev
    dStock = delta_prev * (S_path[i] - S_path[i-1])
    dPnL   = dVal - dStock
    
    # 3) Ajusta la delta si re-hedgeamos
    shares_traded = 0.0
    if i % hedge_frequency == 0:
        shares_traded = (delta_i - delta_prev)
    else:
        # no cambio de delta
        delta_i = delta_prev
    
    cost_trading = transaction_cost_rate * abs(shares_traded * S_path[i])
    
    Delta_portfolio[i] = delta_i
    
    # 4) Actualiza PnL
    PnL[i] = PnL[i-1] + dPnL - cost_trading

# =============================================================================
# 6. ANIMACIÓN CON 9 SUBPLOTS
#    1) S(t)
#    2) Valor total del portafolio
#    3) P&L
#    4) Delta total
#    5) Gamma total
#    6) Delta put leg
#    7) Gamma put leg
#    8) Delta call leg
#    9) Gamma call leg
# =============================================================================

fig, axs = plt.subplots(3, 3, figsize=(13,10))
# Subplots references:
ax_s    = axs[0,0]
ax_val  = axs[0,1]
ax_pnl  = axs[0,2]
ax_dtot = axs[1,0]
ax_gtot = axs[1,1]
ax_dput = axs[1,2]
ax_gput = axs[2,0]
ax_dcal = axs[2,1]
ax_gcal = axs[2,2]

# 1) Subyacente
ax_s.set_title("Subyacente S(t) con saltos")
ax_s.set_xlim(0, N)
ax_s.set_ylim(min(S_path)*0.9, max(S_path)*1.1)
lineS,  = ax_s.plot([], [], 'b-', lw=2)
pointS, = ax_s.plot([], [], 'ro', ms=4)

# 2) Valor total portafolio
ax_val.set_title("Valor total cartera exótica (outside range)")
ax_val.set_xlim(0, N)
ax_val.set_ylim(min(Portfolio_value)*0.9, max(Portfolio_value)*1.1 + 0.01)
lineVal, = ax_val.plot([], [], 'g-', lw=2)
pointVal,= ax_val.plot([], [], 'mo', ms=4)

# 3) P&L
ax_pnl.set_title("P&L acumulado")
ax_pnl.set_xlim(0, N)
ax_pnl.set_ylim(min(PnL)*1.1, max(PnL)*1.1 + 0.01)
linePnl, = ax_pnl.plot([], [], 'r-', lw=2)
pointPnl,= ax_pnl.plot([], [], 'ko', ms=4)

# 4) Delta total
ax_dtot.set_title("Delta total del portafolio")
ax_dtot.set_xlim(0, N)
delta_min = min(Delta_portfolio) - 0.1
delta_max = max(Delta_portfolio) + 0.1
ax_dtot.set_ylim(delta_min, delta_max)
lineDtot, = ax_dtot.plot([], [], 'c-', lw=2)
pointDtot,= ax_dtot.plot([], [], 'bo', ms=4)

# 5) Gamma total
ax_gtot.set_title("Gamma total del portafolio")
ax_gtot.set_xlim(0, N)
gamma_min = min(Gamma_portfolio) - 0.1
gamma_max = max(Gamma_portfolio) + 0.1
ax_gtot.set_ylim(gamma_min, gamma_max)
lineGtot, = ax_gtot.plot([], [], 'm-', lw=2)
pointGtot,= ax_gtot.plot([], [], 'go', ms=4)

# 6) Delta put leg
ax_dput.set_title("Delta put leg (K1)")
ax_dput.set_xlim(0, N)
dp_min = min(Put_leg_delta) - 0.1
dp_max = max(Put_leg_delta) + 0.1
ax_dput.set_ylim(dp_min, dp_max)
lineDput, = ax_dput.plot([], [], 'y-', lw=2)
pointDput,= ax_dput.plot([], [], 'ro', ms=4)

# 7) Gamma put leg
ax_gput.set_title("Gamma put leg (K1)")
ax_gput.set_xlim(0, N)
gp_min = min(Put_leg_gamma) - 0.1
gp_max = max(Put_leg_gamma) + 0.1
ax_gput.set_ylim(gp_min, gp_max)
lineGput, = ax_gput.plot([], [], 'b-', lw=2)
pointGput,= ax_gput.plot([], [], 'ko', ms=4)

# 8) Delta call leg
ax_dcal.set_title("Delta call leg (K2)")
ax_dcal.set_xlim(0, N)
dc_min = min(Call_leg_delta) - 0.1
dc_max = max(Call_leg_delta) + 0.1
ax_dcal.set_ylim(dc_min, dc_max)
lineDcal, = ax_dcal.plot([], [], 'g-', lw=2)
pointDcal,= ax_dcal.plot([], [], 'mo', ms=4)

# 9) Gamma call leg
ax_gcal.set_title("Gamma call leg (K2)")
ax_gcal.set_xlim(0, N)
gc_min = min(Call_leg_gamma) - 0.1
gc_max = max(Call_leg_gamma) + 0.1
ax_gcal.set_ylim(gc_min, gc_max)
lineGcal, = ax_gcal.plot([], [], 'r-', lw=2)
pointGcal,= ax_gcal.plot([], [], 'co', ms=4)

plt.tight_layout()

def init():
    lineS.set_data([], [])
    pointS.set_data([], [])
    lineVal.set_data([], [])
    pointVal.set_data([], [])
    linePnl.set_data([], [])
    pointPnl.set_data([], [])
    lineDtot.set_data([], [])
    pointDtot.set_data([], [])
    lineGtot.set_data([], [])
    pointGtot.set_data([], [])
    lineDput.set_data([], [])
    pointDput.set_data([], [])
    lineGput.set_data([], [])
    pointGput.set_data([], [])
    lineDcal.set_data([], [])
    pointDcal.set_data([], [])
    lineGcal.set_data([], [])
    pointGcal.set_data([], [])
    return (lineS, pointS, lineVal, pointVal, linePnl, pointPnl,
            lineDtot, pointDtot, lineGtot, pointGtot,
            lineDput, pointDput, lineGput, pointGput,
            lineDcal, pointDcal, lineGcal, pointGcal)

def update(frame):
    xdata = np.arange(frame+1)
    
    # 1) Subyacente
    lineS.set_data(xdata, S_path[:frame+1])
    pointS.set_data(frame, S_path[frame])
    
    # 2) Valor total
    lineVal.set_data(xdata, Portfolio_value[:frame+1])
    pointVal.set_data(frame, Portfolio_value[frame])
    
    # 3) P&L
    linePnl.set_data(xdata, PnL[:frame+1])
    pointPnl.set_data(frame, PnL[frame])
    
    # 4) Delta total
    lineDtot.set_data(xdata, Delta_portfolio[:frame+1])
    pointDtot.set_data(frame, Delta_portfolio[frame])
    
    # 5) Gamma total
    lineGtot.set_data(xdata, Gamma_portfolio[:frame+1])
    pointGtot.set_data(frame, Gamma_portfolio[frame])
    
    # 6) Delta put leg
    lineDput.set_data(xdata, Put_leg_delta[:frame+1])
    pointDput.set_data(frame, Put_leg_delta[frame])
    
    # 7) Gamma put leg
    lineGput.set_data(xdata, Put_leg_gamma[:frame+1])
    pointGput.set_data(frame, Put_leg_gamma[frame])
    
    # 8) Delta call leg
    lineDcal.set_data(xdata, Call_leg_delta[:frame+1])
    pointDcal.set_data(frame, Call_leg_delta[frame])
    
    # 9) Gamma call leg
    lineGcal.set_data(xdata, Call_leg_gamma[:frame+1])
    pointGcal.set_data(frame, Call_leg_gamma[frame])
    
    return (lineS, pointS, lineVal, pointVal, linePnl, pointPnl,
            lineDtot, pointDtot, lineGtot, pointGtot,
            lineDput, pointDput, lineGput, pointGput,
            lineDcal, pointDcal, lineGcal, pointGcal)

ani = FuncAnimation(fig, update, frames=N+1, init_func=init, blit=False, interval=50)
plt.show()
