import numpy as np
import matplotlib.pyplot as plt

# USER INPUT SECTION

print("\n--- Enter Healthy Patient Parameters ---")
k_healthy = float(input("Enter k (gain factor): "))
a_healthy = float(input("Enter a (slope sensitivity): "))
P0_healthy = float(input("Enter P0 (critical pressure): "))

print("\n--- Enter Stroke Patient Parameters ---")
k_stroke = float(input("Enter k (gain factor): "))
a_stroke = float(input("Enter a (slope sensitivity): "))
P0_stroke = float(input("Enter P0 (critical pressure): "))

print("\n--- Pressure Range ---")
P_min = float(input("Enter minimum perfusion pressure: "))
P_max = float(input("Enter maximum perfusion pressure: "))

# Generate pressure array

P = np.linspace(P_min, P_max, 500)


# FUNCTIONS

def dF_dP(P, k, a, P0):
    """First derivative (autoregulation slope)"""
    return k / (1 + np.exp(-a*(P - P0)))


def F(P, k, a, P0, F0=50):
    """Integrate derivative to estimate flow"""
    dF = dF_dP(P, k, a, P0)
    return F0 + np.cumsum(dF) * (P[1] - P[0])


def d2F_dP2(P, k, a, P0):
    """Second derivative (curvature)"""
    exp_term = np.exp(-a*(P - P0))
    return (k * a * exp_term) / (1 + exp_term)**2


# CALCULATE CURVES

F_healthy = F(P, k_healthy, a_healthy, P0_healthy)
F_stroke = F(P, k_stroke, a_stroke, P0_stroke)

dF_healthy = dF_dP(P, k_healthy, a_healthy, P0_healthy)
dF_stroke = dF_dP(P, k_stroke, a_stroke, P0_stroke)

d2F_healthy = d2F_dP2(P, k_healthy, a_healthy, P0_healthy)
d2F_stroke = d2F_dP2(P, k_stroke, a_stroke, P0_stroke)


# PLOTS

plt.figure(figsize=(12, 9))

# Flow Curve 
plt.subplot(3,1,1)
plt.plot(P, F_healthy, color='green', label='Healthy')
plt.plot(P, F_stroke, color='red', label='Stroke')

plt.axvline(P0_healthy, color='gray', linestyle='--', label='Critical Pressure')

plt.fill_between(P, F_healthy.min(), F_healthy.max(),
                 where=(P < P0_healthy),
                 color='blue', alpha=0.1, label='Vasodilation')

plt.fill_between(P, F_healthy.min(), F_healthy.max(),
                 where=(P > P0_healthy),
                 color='orange', alpha=0.1, label='Vasoconstriction')

plt.ylabel('Blood Flow F')
plt.title('Cerebral Blood Flow Autoregulation (User Input Model)')
plt.legend()
plt.grid(True)

#  First Derivative 
plt.subplot(3,1,2)
plt.plot(P, dF_healthy, color='green', label='dF/dP Healthy')
plt.plot(P, dF_stroke, color='red', label='dF/dP Stroke')
plt.ylabel('dF/dP')
plt.legend()
plt.grid(True)

#  Second Derivative 
plt.subplot(3,1,3)
plt.plot(P, d2F_healthy, color='green', label='d²F/dP² Healthy')
plt.plot(P, d2F_stroke, color='red', label='d²F/dP² Stroke')
plt.xlabel('Perfusion Pressure P')
plt.ylabel('d²F/dP²')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
