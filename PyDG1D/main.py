# +=======================================================================+
# | 1-Dimensional Discontinuous Galerkin for Maxwell Equations.           |
# | Created by Luis Manuel Diaz Angulo for the University of Granada.     |
# +-----------------------------------------------------------------------+
# | PURPOSE:                                                              |
# | This script simulates propagation of an EM wave in 1-D using a DGTD   |
# | algorithm.                                                            |
# +=======================================================================+
# ======= List Physical and mathematical constants ========================
import scipy.constants
# -------- Physical constants ---------------------------------------------
c0   = scipy.constants.speed_of_light
eps0 = scipy.constants.epsilon_0
mu0  = scipy.constants.mu_0

import numpy as np
# -------- Low storage Runge-Kutta coefficients ---------------------------
rk4a = np.array([                    
                                     0.0, 
        - 567301805773.0/1357537059087.0, 
        -2404267990393.0/2016746695238.0,
        -3550918686646.0/2091501179385.0,
        -1275806237668.0/842570457699.0]);
rk4b = np.array([
         1432997174477.0/ 9575080441755.0, 
         5161836677717.0/13612068292357.0,
         1720146321549.0/ 2090206949498.0,
         3134564353537.0/ 4481467310338.0, 
         2277821191437.0/ 14882151754819.0]);
rk4c = np.array([
                                     0.0,
         1432997174477.0/9575080441755.0,
         2526269341429.0/6820363962896.0,
         2006345519317.0/3224310063776.0,
         2802321613138.0/2924317926251.0]);
# ============== Initialization ===============================================
# -------Simulation general parameters ----------------------------------------
N = 4;        # N    := Degree of polynomials forming the basis.
h = 0.25;     # h    := Spatial resolution. [meters]
alpha = 0.0;  # 1 = Central , 0 = Upwind
xMin = 0.0;
xMax = 1.0;
h = 0.1;
# -----------------------------------------------------------------------------
K = xMax - xMin / h;
vx = np.array([
    np.linspace(xMin  , xMax-h, K),
    np.linspace(xMin+h, xMax  , K)
]);                             # vx := Element vertices.
x = setNodes1D(N,vx); # x    := Nodes in equispaced positions.
# Creates element structure.
#for k=1:K
#    e(k) = struct('x', x(:,k));
#end
# # ------- Material properties -----------------------------------------------
# Ns = 1e3;
# freqMin = 1e2;
# freqMax = 1e9;
# fq = linspace(freqMin, freqMax, Ns);
# omega = 2 * pi * fq;
# matLib = loadMatLib(omega');
# m1 = matLib(1);
# e = setMaterial(e, m1, 1, K);
# # ------- Initial conditions ----------------------------------------------
# [ e ] = setZeroField(e); # Sets initial fields to zero.
# # ------- Analytical matrices definitions ---------------------------------
# addpath 'Analytical matrices';
# T    = TmatA(N,1);     # General Mass Matrix for that order.
# D{1} = Dmatrix1D(N,1); # Derivative matrix for simplex coordinate 1.
# D{2} = Dmatrix1D(N,2); # Derivative matrix for simplex coordiante 2.
# # ------- Computes element parameters -------------------------------------
# for k=1:K
#     M = (e(k).x(Np)-e(k).x(1)) /2 .* T; # M := Mass Matrix.
#     e(k).invM = inv( M );                # invM := Inverse Mass Matrix
#     e(k).S =  (D{2} - D{1})' / (e(k).x(Np)-e(k).x(1));  # S := Stiffness Matrix.
#     e(k).LIFT = e(k).invM ;              # LIFT := Fluxes operator.
# end
# # ------------- Time parameters initialization ----------------------------
# dt     = computeTimeStep(e, CFL); # dt     := Time step.
# Nsteps = floor(finalTime / dt);   # Nsteps := Number of iterations needed. 
# time   = 0:dt:(Nsteps*dt);        # time   := Time vector.
# # ------------- Initialization of aux. variables --------------------------
# for k=1:K
#     # Right hand side variables.
#     e(k).rhsE = zeros(Np,1);
#     e(k).rhsH = zeros(Np,1);
#     # Jumps between elements.
#     e(k).dE = zeros(2,1);
#     e(k).dH = zeros(2,1);
#     # Fluxes.
#     e(k).fluxE = zeros(Np,1);
#     e(k).fluxH = zeros(Np,1);
#     # Residual fields.
#     e(k).resE = zeros(Np,1);
#     e(k).resH = zeros(Np,1);
# end
# # Incident field.
# As = 1;
# Ag = 1;
# freq = 0;
# spread = 3e-10;
# disp   = 20e-10;
# [eS, eT] = getElementsWithBoundaries(e, 0.05);
# [eT2, eS2] = getElementsWithBoundaries(e, 0.45);
# ## ============= Time integration ========================================= 
# # ------------- Main iterative loop ---------------------------------------
# figure(3);
# for tstep = 1:Nsteps
#     # ------------ Computes five stage Runge Kutta time integration -------
#     for INTRK = 1:5
#         # ---------------- RHS terms computation --------------------------
#         e = computeJumps(e);
#         e = imposeBC1D(e, SMABC, SMABC);
#         timeDelayed = time(tstep) - e(eS).x(Np) / c0;
#         e = setIncidentField(e, eT, eS, ... 
#              incidentFieldAmplitude(As, freq, timeDelayed, ...
#                                     Ag, spread, disp));
#         timeDelayed = time(tstep) - e(eT2).x(Np) / c0;
#         e = setIncidentField(e, eT2, eS2, ... 
#              incidentFieldAmplitude(As, freq, timeDelayed, ...
#                                     Ag, spread, disp));
#         e = computeFluxes(e, alpha);
#         e = computeRHS(e);
#         for k = 1:K
#             # ---------------- Updates fields -----------------------------
#             e(k).resE = rk4a(INTRK) * e(k).resE + dt * e(k).rhsE;
#             e(k).resH = rk4a(INTRK) * e(k).resH + dt * e(k).rhsH;
#             e(k).E    = e(k).E + rk4b(INTRK) * e(k).resE;
#             e(k).H    = e(k).H + rk4b(INTRK) * e(k).resH;
#         end
#     end
#     # ------------ Plots fields -------------------------------------------
#     if mod(tstep, framesShown) == 0
#         plotFields(e, time(tstep));
#     end
# end
# 
# 
# quit()

