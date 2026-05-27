import numpy as np
import matplotlib.pyplot as plt

# DDPM linear schedule (Section 4 of Ho et al. 2020)
T = 1000
beta = np.linspace(1e-4, 0.02, T)   # β_1 ... β_T
alpha = 1.0 - beta                    # α_t = 1 - β_t
alpha_bar = np.cumprod(alpha)         # ᾱ_t = ∏_{s=1}^t α_s

# σ²_t = β_t (given assumption)
# coefficient = β²_t / (2 σ²_t α_t (1 - ᾱ_t))
#             = β_t   / (2 α_t (1 - ᾱ_t))
coeff = beta**2 / (2 * beta * alpha * (1 - alpha_bar))

t = np.arange(1, T + 1)

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(t, coeff, color='#2563EB', linewidth=1.8,
        label=r'$\dfrac{\beta_t^2}{2\sigma_t^2\,\alpha_t(1-\bar{\alpha}_t)}$')
ax.set_yscale('log')

ax.set_xlabel(r'Timestep $t$', fontsize=13)
ax.set_ylabel(r'Coefficient (log scale)', fontsize=13)
ax.set_title(r'Loss coefficient $\dfrac{\beta_t^2}{2\sigma_t^2\,\alpha_t(1-\bar{\alpha}_t)}$ vs. timestep $t$',
             fontsize=13, pad=14)

ax.axvspan(1,   200,  alpha=0.07, color='red',   label='Small $t$: large coefficient')
ax.axvspan(800, 1000, alpha=0.07, color='green',  label='Large $t$: small coefficient')

ax.legend(fontsize=11, loc='upper right')
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
ax.set_xlim(1, T)

plt.tight_layout()
plt.savefig('ddpm_coeff.png', dpi=150, bbox_inches='tight')
plt.show()