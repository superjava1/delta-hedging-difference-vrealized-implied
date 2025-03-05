# 📘 Hedging, P&L, and Volatility Mismatch: A Practical Guide

## Overview
This repository provides a detailed exploration of **hedging strategies** under the **Black-Scholes-Merton framework**, focusing on the effects of **implied vs. realized volatility mismatches** on **P&L (Profit and Loss)**. 

We implement a **realistic Monte Carlo simulation** to:
- Replicate an **exotic option** (e.g., a **double digital**) using a portfolio of **vanilla options** (calls and puts).
- **Delta-hedge** the portfolio dynamically while tracking **P&L** evolution.
- Analyze the impact of **gamma trading**, transaction costs, and re-hedging frequency.

This repository also includes a **comprehensive mathematical breakdown** of the **5.xx equations** from a key financial textbook, explaining the theoretical foundations behind **volatility arbitrage and option hedging strategies**.

## 🔥 Features
✅ **Dynamic hedging**: Adjusts position sizes over time based on market conditions.  
✅ **Exotic option replication**: Uses **spreads** to approximate a **double digital** payoff.  
✅ **Realistic volatility modeling**: Incorporates **jumps, high realized vol, and hedging mismatches**.  
✅ **Full P&L decomposition**: Tracks delta, gamma, and cost effects.  
✅ **Mathematical documentation**: Explains each equation with context and derivations.  
✅ **Animations & visualizations**: See **how the hedge portfolio evolves** over time.

## 📊 Equations Covered
We analyze the **5.xx equations** related to hedging strategies:
- **5.27 - 5.30**: Incremental **P&L under implied vs. realized vol**.
- **5.31 - 5.34**: **Present value of P&L** with discounting.
- **5.35 - 5.38**: **Gamma trading effects** when vol mismatches occur.
- **5.42 - 5.43**: How P&L accumulates over time due to **gamma and vol differences**.

# The 5.xx Equations and Their Purpose

This document summarizes the key 5.xx equations related to **delta-hedging** and the difference between **implied** vs. **realized volatility** in a Black–Scholes–Merton framework. Each equation is stated briefly with its main meaning and usage.

---

## 1. (5.27)
\[
\mathrm{d}P\&L(I,R) \;=\; \mathrm{d}V_I \;-\; \Delta_I\,\mathrm{d}S.
\]

- **Meaning**: Incremental profit/loss when the option is valued with implied volatility \(\sigma_I\) and hedged using \(\Delta_I\).
- **Interpretation**: \(\mathrm{d}V_I\) is the option’s value change; \(\Delta_I\,\mathrm{d}S\) is the change in the hedge position.

---

## 2. (5.28)
\[
\mathrm{d}P\&L(R) \;=\; \mathrm{d}V_R \;-\; \Delta_R\,\mathrm{d}S.
\]

- **Meaning**: Incremental P&L if one could perfectly value and hedge the option with the realized volatility \(\sigma_R\).
- **Interpretation**: Hypothetical scenario showing how a “riskless” strategy arises under a known \(\sigma_R\).

---

## 3. (5.29)
\[
\Delta_R \,\bigl[\mathrm{d}S - (r - D)\,S\,\mathrm{d}t\bigr] \;=\; \mathrm{d}V_R \;-\; r\,V_R\,\mathrm{d}t.
\]

- **Meaning**: Rearranges (5.28) to highlight that, in the idealized Black–Scholes sense, the hedge plus the option earn the risk-free rate \(r\).
- **Interpretation**: Formalizes the “perfect hedge” argument when \(\sigma_R\) is assumed known and constant.

---

## 4. (5.30)
\[
\mathrm{d}P\&L(I,R) \;=\; \mathrm{d}V_I \;-\; \mathrm{d}V_R \;-\; \bigl(V_I - V_R\bigr)\,r\,\mathrm{d}t.
\]

- **Meaning**: Difference between the valuation changes using \(\sigma_I\) vs. \(\sigma_R\).
- **Interpretation**: Tracks how \(\mathrm{d}P\&L\) arises when implied vol is different from the realized vol.

---

## 5. (5.31) and (5.32)
\[
\mathrm{d}P\&L(I,R) \;=\; e^{rt}\,\mathrm{d}\Bigl[e^{-rt}\,(V_I - V_R)\Bigr],
\]
\[
\text{PV}\bigl[\mathrm{d}P\&L(I,R)\bigr] \;=\; e^{-r(t-t_0)}\,\mathrm{d}\Bigl[e^{-rt}\,(V_I - V_R)\Bigr].
\]

- **Meaning**: Expresses the incremental P&L in a total differential form, factoring in discounting at rate \(r\).
- **Interpretation**: Makes it easier to integrate the P&L over time and to compute present values.

---

## 6. (5.33) and (5.34)
\[
\text{PV}[P\&L(I,R)] 
= 
e^{r\,t_0}
\Bigl[
e^{-r t}\,(V_I - V_R)
\Bigr]_{t_0}^{T},
\quad
\text{and}
\quad
\text{PV}[P\&L(I,R)]
= 
e^{-r(T-t_0)}\bigl(V_{R,T} - V_{I,T}\bigr).
\]

- **Meaning**: Integrated form showing the present value of total P&L from \(t_0\) to \(T\). 
- **Interpretation**: Demonstrates that at maturity, both valuations converge to the same payoff, so the real difference (and resulting P&L) is generated during the life of the option.

---

## 7. (5.35)–(5.38)
These detail the **Itô expansion** of \(\mathrm{d}V_I\) and highlight how **gamma** and **the difference** \(\sigma_R^2 - \sigma_I^2\) drive the extra P&L.

- **(5.38)** is especially important:
  \[
    \mathrm{d}P\&L(I,R) 
    = 
    \tfrac12\,\Gamma_I\,S^2\,\bigl(\sigma_R^2 - \sigma_I^2\bigr)\,\mathrm{d}t
    \;+\;\dots
  \]
  showing the source of additional gains/losses when realized volatility \(\sigma_R\) differs from \(\sigma_I\).

---

## 8. (5.39)–(5.41)
- **Meaning**: Present theoretical upper and lower **bounds** for the P&L, based on integrating these differences and analyzing payoff constraints.
- **Interpretation**: Provide a range of possible outcomes given different assumptions about \(S\) and \(\sigma\).

---

## 9. (5.42)
\[
\mathrm{d}P\&L(I,I) 
= 
\tfrac12\,\Gamma_I\,S^2\,\bigl(\sigma_R^2 - \sigma_I^2\bigr)\,\mathrm{d}t.
\]

- **Meaning**: A simplified version emphasizing that the gamma term times \((\sigma_R^2 - \sigma_I^2)\) is the key driver of incremental P&L in a delta-hedged position.
- **Interpretation**: Illustrates how “long gamma” positions profit if realized vol is higher than implied vol.

---

## 10. (5.43)
\[
\text{PV}\bigl[P\&L(I,I)\bigr]
= 
\int_0^T
e^{-r\,t}
\Bigl[
\tfrac12\,\Gamma_I\,S^2\,\bigl(\sigma_R^2 - \sigma_I^2\bigr)
\Bigr]
\,\mathrm{d}t.
\]

- **Meaning**: The present value of the total P&L over the option’s lifetime, integrating the mismatch in volatilities.
- **Interpretation**: Captures the idea that the net profit/loss from a gamma position depends on how much actual variance (\(\sigma_R^2\)) exceeds (or falls below) the implied variance (\(\sigma_I^2\)), aggregated and discounted over time.

---

### **In Summary**
- **(5.27)–(5.28)**: Incremental P&L expressions (implied vs. realized).  
- **(5.29)–(5.34)**: Integration and present value, showing how the difference arises and when it is realized.  
- **(5.35)–(5.38)**: Itô expansion, highlighting gamma and \(\sigma_R^2 - \sigma_I^2\).  
- **(5.39)–(5.41)**: Bounds on the P&L.  
- **(5.42)–(5.43)**: Final summary of the gamma-driven extra profit/loss when realized vol differs from implied vol.


