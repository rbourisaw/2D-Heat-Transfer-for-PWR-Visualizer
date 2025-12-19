import numpy as np
from iapws import IAPWS97
import matplotlib
matplotlib.use('Qt5Agg')
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
from scipy.interpolate import interp1d, interpn
from numba import njit
from CHF_loader import CHF_load
import pyqtgraph as pg

def main(P_initial1, P_end, Period1, Shape1, Flow, progress, cancel):
    #Make sure args are right data type:
    P_initial1 = float(P_initial1)*1e6 #expected to be W, should be passed in as MW
    P_end = float(P_end)*1e6 #expected to be W, should be passed in as MW
    Period1 = float(Period1) #expected to be seconds
    Flow = float(Flow) #expected to be m/s
    Shape1 = str(Shape1) #Has to be Middle, Uniform, Outlet, or Inlet.
    
    #Constants
    P_w = 15.5132  # MPa
    T_sat = IAPWS97(P=P_w, x=0).T #K
    h_f = IAPWS97(P=P_w, x=0).h*1000 #J/kgK
    h_g = IAPWS97(P=P_w, x=1).h*1000
    h_fg = h_g - h_f
    
    #load water props from table file:
    data = np.load("water_props_h.npz")
    h_grid = data['h_grid']
    rho_table = data["rho"]
    cp_table  = data["cp"]
    k_table   = data["k"]
    mu_table  = data["mu"]
    st_table  = data['st']

    #load Pressure values for two-phase flow correlation:
    data_1 = np.load('water_satP_table.npz')
    T_grid = data_1['T_grid']
    P_wall = data_1['P_sat']
    P_sat_interp = interp1d(T_grid, P_wall,
                            kind='linear',
                            fill_value='extrapolate')

    uk_a, uk_b, uk_c = np.polyfit([500, 2150, 2500], [5.5, 2.05, 2.3], 2)

    theta = 516.12  # K
    C1 = 78.215
    C2 = 3.8609e-3
    C3 = 3.4250e8
    Ea = 1.9105
    kB = 8.6144e-5
    M_w = 0.03*235 + 0.97*238 + 32
    rho_273 = 10.963e3
    U_avg = Flow
    time_sol = 0.0
    T_in = 289.33+273.15
    h_in = IAPWS97(P=P_w, T=T_in).h*1000
    G_const = U_avg*IAPWS97(P=P_w, T=T_in).rho
    Pr_T = 0.85 #common value
    CHF_grid, unique_P, unique_G, unique_X = CHF_load()

    #Define constant two-phase flow values:
    rho_l = IAPWS97(P=P_w, x=0).rho
    rho_g = IAPWS97(P=P_w, x=1).rho
    cp_l = IAPWS97(P=P_w, x=0).cp*1000
    cp_g = IAPWS97(P=P_w, x=1).cp*1000
    k_l = IAPWS97(P=P_w, x=0).k
    k_g = IAPWS97(P=P_w, x=1).k
    mu_l = IAPWS97(P=P_w, x=0).mu
    mu_g = IAPWS97(P=P_w, x=1).mu
    st_l = IAPWS97(P=P_w, x=0).sigma

    #Define constants for fuel, clad, and water initial temp profiles, period/dt fit:
    a_f, b_f, c_f = 4.309447056032611e-18, 8.21438869660193e-08, 291.2279365441616
    a_c, b_c, c_c = -4.121335837516241e-19, 1.453563904520749e-08, 292.1740751600232
    a_w, b_w, c_w, d_w = 5.848453058775314e-29, -1.4990297125701472e-18, 1.3407314268828512e-08, 291.8913970824419
    a_p, b_p, c_p = 0.0011472906052745592, -0.002750594702998647, 0.037281497533654034

    #Grid
    fuel_outr = 0.00475
    clad_outr = 0.00532
    flow_outr = 0.00630
    fuel_z = 3.88112
    total_fuel_vol = np.pi*fuel_outr**2*fuel_z
    A_tot = (flow_outr*2)**2 - np.pi*clad_outr**2
    D_h = 4*(((2*flow_outr)**2) - (np.pi*clad_outr**2))/(2*flow_outr*4)
    K_2_const = ((1/2) + ((2*(flow_outr*2 - clad_outr*2))/(clad_outr*2)))
    geom_factor = (0.042*((flow_outr*2)/(clad_outr*2)) - 0.024)

    r_faces = np.linspace(0, flow_outr, 150)
    z_faces = np.linspace(0, fuel_z, 300)
    r_nodes = 0.5*(r_faces[:-1]+r_faces[1:])
    z_nodes = 0.5*(z_faces[:-1]+z_faces[1:])
    Nr, Nz = len(r_nodes), len(z_nodes)
    N = Nr * Nz

    fuel_end = np.searchsorted(r_nodes, fuel_outr, side="right")
    clad_end = np.searchsorted(r_nodes, clad_outr, side="right")
    idx_fuel = np.arange(0, fuel_end)
    idx_clad = np.arange(fuel_end, clad_end)
    idx_water = np.arange(clad_end, len(r_nodes))
    r_fuel = r_nodes[idx_fuel]
    Z_fuel, R_fuel = np.meshgrid(z_nodes, r_fuel, indexing='ij')
    dr = np.diff(r_faces)
    dz = np.diff(z_faces)
    dz_cell = dz[0]
    dr_cell = dr[0]
    area_conv_face = 2*np.pi*clad_outr*dz_cell
    area_gas_gap = 2*np.pi*(fuel_outr + fuel_outr + 0.0001651)*0.5*dz_cell
    h_gas = 1.7e4
    V_cells = np.outer(dz, np.pi*(r_faces[1:]**2 - r_faces[:-1]**2))
    V_v = V_cells.reshape(-1)
    ring_areas = np.pi*(r_faces[1:]**2 - r_faces[:-1]**2)
    ring_areas[-1] = (flow_outr*2)**2 - np.pi*r_faces[-2]**2
    ring_areas_2D = np.tile(ring_areas, (Nz,1))
    ring_areas_v = ring_areas_2D.reshape(-1)
    N_water = len(idx_water)
    r_avg_e = 0.5*(r_nodes[:-1] + r_nodes[1:])
    r_avg_e = np.append(r_avg_e, 0)
    r_avg_w = np.empty_like(r_nodes)
    r_avg_w[1:] = r_avg_e[:-1]
    r_avg_w[0] = r_avg_e[0]
    area_e = 2*np.pi * r_avg_e * dz_cell
    area_w = 2*np.pi * r_avg_w * dz_cell
    area_e_2D = np.tile(area_e, (Nz, 1))
    area_w_2D = np.tile(area_w, (Nz, 1))
    area_e_v = area_e_2D.reshape(-1)
    area_w_v = area_w_2D.reshape(-1)
    cancel()

    #make masks for A_b_matrix builder function for proper boundary conditions
    east_mask_2D  = np.ones((Nz, Nr), dtype=bool)
    west_mask_2D  = np.ones((Nz, Nr), dtype=bool)
    north_mask_2D = np.ones((Nz, Nr), dtype=bool)
    south_mask_2D = np.ones((Nz, Nr), dtype=bool)
    east_mask_2D[:, -1] = False
    west_mask_2D[:, 0] = False
    north_mask_2D[-1, :] = False 
    south_mask_2D[0,  :] = False
    east_mask = east_mask_2D.reshape(-1)
    west_mask = west_mask_2D.reshape(-1)
    north_mask = north_mask_2D.reshape(-1)
    south_mask = south_mask_2D.reshape(-1)
    water_mask_2D = np.zeros((Nz, Nr), dtype=bool)
    water_mask_2D[:, idx_water] = True
    water_mask = water_mask_2D.reshape(-1)
    fuel_mask_2D = np.zeros((Nz, Nr), dtype=bool)
    fuel_mask_2D[:, idx_fuel] = True
    fuel_mask = fuel_mask_2D.reshape(-1)

    #Pre-building of sparse diag matrix:
    diagonals = [np.ones(N),    # main
                    np.ones(N-1), # east
                    np.ones(N-1), # west
                    np.ones(N-Nr),# north
                    np.ones(N-Nr)]# south
    offsets = [0, 1, -1, Nr, -Nr]

    A_template = diags(diagonals, offsets, shape=(N, N), format='csr')

    # --- Precompute CSR positions of each diagonal ---
    main_idx, east_idx, west_idx, north_idx, south_idx = [], [], [], [], []
    for i in range(N):
        row_start = A_template.indptr[i]
        row_end = A_template.indptr[i+1]
        row_cols = A_template.indices[row_start:row_end]

        for k, col in enumerate(row_cols):
            pos = row_start + k
            if col == i:
                main_idx.append(pos)
            elif col == i + 1:
                east_idx.append(pos)
            elif col == i - 1:
                west_idx.append(pos)
            elif col == i + Nr:
                north_idx.append(pos)
            elif col == i - Nr:
                south_idx.append(pos)
                
    main_idx = np.array(main_idx)
    east_idx = np.array(east_idx)
    west_idx = np.array(west_idx)
    north_idx = np.array(north_idx)
    south_idx = np.array(south_idx)
    cancel()

    #Material property functions
    def water_props(h):
        h = h.flatten()
        
        sub_mask = h <= h_f
        mix_mask = ~sub_mask

        rho_out = np.zeros_like(h)
        cp_out  = np.zeros_like(h)
        k_out  = np.zeros_like(h)
        mu_out  = np.zeros_like(h)
        st_out = np.zeros_like(h)

        if np.any(sub_mask):
            h_sub = h[sub_mask]

            idx = np.interp(h_sub, h_grid, np.arange(len(h_grid)))
            i0 = np.floor(idx).astype(int)
            i1 = np.clip(i0 + 1, 0, len(h_grid) - 1)
            t = idx - i0

            rho_out[sub_mask] = (1 - t) * rho_table[i0] + t * rho_table[i1]
            cp_out[sub_mask] = (1 - t) * cp_table[i0]  + t * cp_table[i1]
            k_out[sub_mask] = (1 - t) * k_table[i0]   + t * k_table[i1]
            mu_out[sub_mask] = (1 - t) * mu_table[i0]  + t * mu_table[i1]
            st_out[sub_mask] = (1 - t) * st_table[i0]  + t * st_table[i1]
            
        if np.any(mix_mask):
            h_mix = h[mix_mask]
            x = (h_mix - h_f) / h_fg
            x = np.clip(x, 0.0, 1.0)

            rho_out[mix_mask] = 1.0 / (((1-x)/rho_l) + (x/rho_g))
            cp_out[mix_mask]  = (1-x)*cp_l + x*cp_g
            k_out[mix_mask]   = (1-x)*k_l + x*k_g
            mu_out[mix_mask]  = (1-x)*mu_l + x*mu_g

        shape = h_field.shape

        return (rho_out.reshape(shape),
                cp_out.reshape(shape),
                k_out.reshape(shape),
                mu_out.reshape(shape),
                st_out.reshape(shape))

    def fuel_props(T):
        k = uk_a*T**2 + uk_b*T + uk_c
        
        c_p = np.zeros_like(T)
        low_T_mask = T < 2670
        term1 = (C1 * theta**2 * np.exp(theta/T[low_T_mask])) / (T[low_T_mask]**2 * (np.exp(theta/T[low_T_mask])-1)**2)
        term2 = 2 * C2 * T[low_T_mask]
        term3 = C3 * kB * np.exp(-Ea/(kB*T[low_T_mask])) * (1 + Ea/(kB*T[low_T_mask]))
        c_p[low_T_mask] = (term1 + term2 + term3)/M_w
        c_p[~low_T_mask] = 167.04
        
        L_ratio = np.zeros_like(T)
        mask1 = T <= 923
        L_ratio[mask1] = 9.9734e-1 + 9.802e-6*T[mask1] - 2.705e-10*T[mask1]**2 + 4.291e-10*T[mask1]**3
        L_ratio[~mask1] = 9.9672e-1 + 1.179e-5*T[~mask1] - 2.429e-9*T[~mask1]**2 + 1.219e-12*T[~mask1]**3
        rho = rho_273 / L_ratio**3
        
        return rho, c_p, k

    def Zirc_props(T):
        c_p = np.zeros_like(T, dtype=float)

        # Linear region (T < 1090 K)
        mask_lin = T <= 1090.0
        c_p[mask_lin] = 0.114738542*T[mask_lin] + 252.546336

        # Polynomial region (T >= 1090 K)
        mask_poly = ~mask_lin
        a, b, c, d, e, f = 1.96639875e-07, -1.14646416e-03, 2.67190771, -3.11152233e3, 1.81060455e6, -4.21183933e8
        Tp = T[mask_poly]
        c_p[mask_poly] = a*Tp**5 + b*Tp**4 + c*Tp**3 + d*Tp**2 + e*Tp + f
        
        k = 7.51 + 2.09e-2*T - 1.45e-5*T**2 + 7.67e-9*T**3
        
        rho0 = 6500.0       # Reference density at T0 [kg/m^3]
        T0 = 293.0          # Reference temperature [K]
        alpha = 5.5e-6      # Linear thermal expansion coefficient [1/K]
        
        # Volume expansion: rho = rho0 / (1 + alpha*(T-T0))^3
        rho = rho0 / (1 + alpha*(T - T0))**3

        return rho, c_p, k
    cancel()

    #CHF predicting function:
    def CHF(h_field, u_profile, rho):
        x = (h_field[:,0] - h_f)/(h_fg)
        G = rho[:,idx_water[0]]*u_profile[:,0]
        P = np.full_like(G, P_w*1e6)
        q_chf = interpn((unique_P, unique_G, unique_X), CHF_grid, np.column_stack((P, G, x)), 
                        method='linear', bounds_error = False, fill_value = None)
        return q_chf

    def power(power_i, power_e, period, time_sol):
        p_reactor = power_i*np.exp(time_sol/period)
        if p_reactor > power_e and period > 0.0:
            p_reactor = power_e
        elif p_reactor < power_e and period < 0.0:
            p_reactor = power_e
        return p_reactor

    #Power profile function
    def vol_heat_rate(shape, power):
        power_per_rod = power / 50952

        if shape == 'Uniform':
            q_vol = np.full_like(Z_fuel, power_per_rod / total_fuel_vol)

        elif shape == 'Middle':
            alpha = 0.4
            f_r = (1 - alpha * (R_fuel / fuel_outr)**2)
            f_z = 2.0 * (1 - 0.25 * np.cos(2 * np.pi * Z_fuel / fuel_z))**2
            f = f_r * f_z
            r_faces_local = r_faces[:fuel_end+1]
            dz = np.diff(z_faces)
            V_cell = np.outer(dz, np.pi * (r_faces_local[1:]**2 - r_faces_local[:-1]**2))
            norm_factor = power_per_rod / np.sum(f * V_cell)
            q_vol = f * norm_factor

        elif shape == "Outlet":
            lam = 0.25
            H_c = fuel_z + 2 * lam
            f_z = (np.pi / (fuel_z * H_c)) * (Z_fuel + lam) * np.sin((np.pi * (Z_fuel + lam)) / H_c)
            alpha = 0.4
            f_r = (1 - alpha * (R_fuel / fuel_outr)**2)
            r_faces_local = r_faces[:fuel_end+1]
            dz = np.diff(z_faces)
            V_cell = np.outer(dz, np.pi * (r_faces_local[1:]**2 - r_faces_local[:-1]**2))
            norm_factor = power_per_rod / np.sum(f_r * f_z * V_cell)
            q_vol = f_r * f_z * norm_factor

        elif shape == 'Inlet':
            lam = 0.25
            H_c = fuel_z + 2 * lam
            f_z = (np.pi / (fuel_z * H_c)) * (fuel_z + lam - Z_fuel) * np.sin((np.pi * (fuel_z + lam - Z_fuel)) / H_c)
            alpha = 0.4
            f_r = (1 - alpha * (R_fuel / fuel_outr)**2)
            r_faces_local = r_faces[:fuel_end+1]
            dz = np.diff(z_faces)
            V_cell = np.outer(dz, np.pi * (r_faces_local[1:]**2 - r_faces_local[:-1]**2))
            norm_factor = power_per_rod / np.sum(f_r * f_z * V_cell)
            q_vol = f_r * f_z * norm_factor

        return q_vol

    def v_profile(rho):
        r_channel = flow_outr - clad_outr
        u_max = U_avg*(1 + (1/7))
        r_rel = u_max*(1 - (flow_outr - r_nodes[idx_water]) / r_channel) ** (1/7)
        A_flows = np.empty(len(idx_water))
        A_flows[0] = np.pi*(r_faces[idx_water[1]])**2 - np.pi*clad_outr**2
        A_flows[1:-1] = np.pi*(r_faces[idx_water[2]:-1])**2 - np.pi*(r_faces[idx_water[1]:-2])**2
        A_flows[-1] = (r_faces[-1]*2)**2 - np.pi*(r_faces[-2])**2
        A_total = np.sum(A_flows)
        m_dot = G_const * A_total
        rho_slice = rho[:, idx_water]
        norm_factor = m_dot / np.sum(rho_slice * A_flows[None,:] * r_rel[None,:], axis=1)
        u_profile = norm_factor[:, None] * r_rel[None, :]
        
        return u_profile

    #Convective coefficient function
    def convective_h(T_field, u_profile, h_field, rho, cp, k, mu, st, 
                    h_single_p=0.0, h_two_p=0.0, q_local=0.0, q_chf=0.0):
        h_z = np.zeros(Nz)
        h_single = np.zeros(Nz)
        h_two = np.zeros(Nz)
        x = (h_field[:,0] - h_f)/h_fg

        T_wat = T_field[:, idx_water[0]]
        T_clad = T_field[:, idx_clad[-1]]
        T_inter = T_clad
        
        mask_single = (T_inter < T_sat)
        mask_onset = (T_inter >= T_sat) & (T_wat < T_sat)
        mask_two = (T_wat >= T_sat)
        chf = np.zeros_like(q_local, dtype=bool)
        ratio = np.zeros_like(q_local)
        
        if np.any(q_chf) != 0:
            x_osb = (x >= 0)
            
            if np.any(x_osb):
                i_osb = np.argmax(x_osb)
                q_seg = q_local[i_osb:]
                q_sum = np.cumsum(q_seg)
                counts = np.arange(1, len(q_seg) + 1)
                q_BLA = q_sum/counts
                K_5 = np.ones_like(q_local)
                K_5[i_osb:] = q_local[i_osb:]/q_BLA
                q_chf = q_chf*K_5
            
            K_2 = np.minimum(1,K_2_const*np.exp(-(np.cbrt(x))/2))
            q_chf = q_chf*K_2
            ratio = q_chf/q_local
            #plt.plot(z_nodes[mask_single], q_local[mask_single]) #*2*np.pi*clad_outr
            #plt.plot(z_nodes, q_chf*K_2)
            #plt.plot(z_nodes[mask_onset], q_local[mask_onset])
            #plt.plot(z_nodes[mask_two], q_local[mask_two])
            #plt.axvline(fuel_z/2)
            #plt.ylim(0, 1e6)
            #plt.show()
            chf_raw = (ratio <= 1)
            chf = np.logical_or.accumulate(chf_raw)
            
        mask_single_eff = mask_single & (~chf)
        mask_onset_eff = mask_onset & (~chf)
        mask_two_eff = mask_two & (~chf)
        
        rho_f = rho[:, idx_water[0]]
        cp_f  = cp[:, idx_water[0]]
        k_f   = k[:, idx_water[0]]
        mu_f  = mu[:, idx_water[0]]
        st_f  = st[:, idx_water[0]]
        
        del_t = T_inter - T_sat
        
        if np.any(mask_single_eff):

            #Extract masked values
            rho_s = rho_f[mask_single_eff]
            cp_s = cp_f[mask_single_eff]
            k_s = k_f[mask_single_eff]
            mu_s = mu_f[mask_single_eff]

            Re_s = rho_s * u_profile[mask_single_eff, 0] * D_h / mu_s
            Pr_s = cp_s * mu_s / k_s

            #Allocate masked Nu array
            Nu_s = np.zeros_like(Re_s)

            #Laminar
            lam = Re_s < 2300
            Nu_s[lam] = 3.66

            #Transitional
            tr = (Re_s >= 2300) & (Re_s < 5000)
            if np.any(tr):
                Nu_lam = 3.66
                Nu_turb_tr = 0.023 * Re_s[tr]**0.8 * Pr_s[tr]**0.4
                Nu_s[tr] = Nu_lam + (Nu_turb_tr - Nu_lam)*(Re_s[tr] - 2300)/2700

            #Fully turbulent
            turb = Re_s >= 5000
            if np.any(turb):
                Nu_s[turb] = (
                    geom_factor *
                    Re_s[turb]**0.8 *
                    Pr_s[turb]**(1/3)
                )

            h_s = Nu_s * k_s / D_h
            h_z[mask_single_eff] = h_s
            h_single[mask_single_eff] = h_s

            
        if np.any(mask_onset_eff):

            rho_o = rho_f[mask_onset_eff]
            cp_o  = cp_f[mask_onset_eff]
            k_o   = k_f[mask_onset_eff]
            mu_o  = mu_f[mask_onset_eff]
            st_o  = st_f[mask_onset_eff]
            st_o = np.where(st_o > 0.0, st_o, st_l)

            u_o = u_profile[mask_onset_eff, 0]

            del_t_o = del_t[mask_onset_eff]
            T_inter_o = T_inter[mask_onset_eff]
            T_inter_clipped = np.minimum(T_inter_o, 647.0)
            del_P_o = np.maximum(P_sat_interp(T_inter_clipped) - P_w, 0.0)*1e6

            Re_o = rho_o * u_o * D_h / mu_o
            Pr_o = cp_o * mu_o / k_o

            h_c_o = geom_factor * Re_o**0.8 * Pr_o**(1/3) * (k_o/D_h)

            S = 1.0 / (
                1.0 + (2.53e-6 *
                    ((rho_o * u_o * D_h) / mu_o)**1.17)
            )

            h_nb_o = (
                (k_o**0.79 * cp_o**0.45 * rho_o**0.49)
                / (st_o**0.5 * mu_o**0.29 * h_fg**0.24 * rho_g**0.24)
                * del_t_o**0.24
                * del_P_o**0.75
                * S * 0.000122 * geom_factor
            )
            
            h_z[mask_onset_eff] = h_c_o + h_nb_o
            h_single[mask_onset_eff] = h_c_o
            h_two[mask_onset_eff] = h_nb_o

        if np.any(mask_two_eff):
            T_inter_min = np.minimum(T_inter, 647.0)
            del_t_two = T_inter[mask_two_eff] - T_sat
            del_P_two = (P_sat_interp(T_inter_min[mask_two_eff]) - P_w)*1e6
            
            x = np.clip(x,0.001,1.0)
            x_t = x[mask_two_eff]

            u_two = u_profile[mask_two_eff, 0]
            
            rho_f_two = 1.0 / (((1 - x_t)/rho_l) + (x_t/rho_g))
            k_f_two = (1 - x_t)*k_l  + x_t*k_g
            cp_f_two = (1 - x_t)*cp_l + x_t*cp_g
            
            X_H = ((1 - x_t)/x_t)**0.9 * (rho_g/rho_l)**0.5 * (mu_l/mu_g)**0.1
            X_H = np.clip(X_H,0.001,11)
            F = np.empty_like(X_H)
            high_mask = X_H > 10
            low_mask  = ~high_mask
            F[high_mask] = 1.0
            F[low_mask]  = 2.35*(0.213 + 1.0/X_H[low_mask])**0.736
            S = 1.0 / (
                1.0 + (2.53e-6 *
                    (((rho_f_two * u_two * (1 - x_t) * D_h) / mu_l) * F**1.25)**1.17)
            )

            Re_two = rho_f_two * u_two * (1-x_t) * D_h / mu_l
            Pr_two = cp_l * mu_l / k_l

            h_c_two = (
                geom_factor * Re_two**0.8 * Pr_two**(1/3) * (k_l/D_h) * F
            )

            h_nb_two = (
                (k_f_two**0.79 * cp_f_two**0.45 * rho_f_two**0.49)
                / (st_l**0.5 * mu_l**0.29 * h_fg**0.24 * rho_g**0.24)
                * del_t_two**0.24
                * del_P_two**0.75
                * S * 0.000122 * geom_factor
            )
            
            h_z[mask_two_eff] = h_c_two + h_nb_two
            h_single[mask_two_eff] = h_c_two
            h_two[mask_two_eff] = h_nb_two
            
        if np.any(chf):
            #Extract masked values
            rho_s = rho_f[chf]
            cp_s = cp_f[chf]
            k_s = k_f[chf]
            mu_s = mu_f[chf]

            Re_s = rho_s * u_profile[chf, 0] * D_h / mu_s
            Pr_s = cp_s * mu_s / k_s

            #Allocate masked Nu array
            Nu_s = np.zeros_like(Re_s)

            #Laminar
            lam = Re_s < 2300
            Nu_s[lam] = 3.66

            #Transitional
            tr = (Re_s >= 2300) & (Re_s < 5000)
            if np.any(tr):
                Nu_lam = 3.66
                Nu_turb_tr = 0.023 * Re_s[tr]**0.8 * Pr_s[tr]**0.4
                Nu_s[tr] = Nu_lam + (Nu_turb_tr - Nu_lam)*(Re_s[tr] - 2300)/2700

            #Fully turbulent
            turb = Re_s >= 5000
            if np.any(turb):
                Nu_s[turb] = (
                    geom_factor *
                    Re_s[turb]**0.8 *
                    Pr_s[turb]**(1/3)
                )

            h_s = Nu_s * k_s / D_h
            h_z[chf] = h_s*0.7
            h_single[chf] = h_s

        #print(h_single[mask_onset_eff], h_two[mask_onset_eff])
        #print(h_z[mask_single_eff], h_z[mask_onset_eff], h_z[mask_two_eff], h_z[chf])
        return h_z, ratio, h_single, h_two, q_chf

    def initial_temp(P):
        T = np.zeros((len(z_nodes), len(r_nodes)))
        T[:,idx_fuel] = a_f*P**2 + b_f*P + c_f
        T[:,idx_clad] = a_c*P**2 + b_c*P + c_c
        T[:,idx_water] = a_w*P**3 + b_w*P**2 + c_w*P + d_w
        return T + 273.15

    def A_b_matrix(k, rho, c_p, q_vol, h_z, u_profile, T_prev, x_field, in_time=False): 
        N = Nr * Nz 
        b_true = np.zeros(N) 
        #Flatten fields
        rho_v = rho.ravel() 
        cp_v = c_p.ravel() 
        q_vol_v = np.zeros((Nz, Nr)) 
        q_vol_v[:, idx_fuel] = q_vol 
        q_vol_v = q_vol_v.ravel() 
        
        #Do second temp field for proper temp tracking
        cap_mask = (T_prev[:,idx_water] >= T_sat) & (x_field <= 1.0)
        if (Period < 0.0) and in_time:
            cap_mask = (x_field >= -0.02) & (x_field <= 1.0)   
        full_cap_mask = np.zeros((Nz, Nr), dtype=bool) 
        full_cap_mask[:,idx_water] = cap_mask 
        full_cap_mask = full_cap_mask.ravel()

        #Apply BCs
        j = np.arange(Nz)
        inlet_rows = idx_water
        east_rows = j*Nr + idx_water[-1]
        b_true[inlet_rows] = T_in 
        b_true[fuel_mask] = q_vol_v[fuel_mask] * V_v[fuel_mask]

        #Compute k coefficients
        k_ip = np.zeros_like(k); k_ip[:, :-1] = 2*k[:, :-1]*k[:, 1:]/(k[:, :-1]+k[:, 1:]) 
        k_im = np.zeros_like(k); k_im[:, 1:] = 2*k[:, 1:]*k[:, :-1]/(k[:, 1:]+k[:, :-1]) 
        k_jp = np.zeros_like(k); k_jp[:-1, :] = 2*k[:-1, :]*k[1:, :]/(k[:-1, :]+k[1:, :]) 
        k_jm = np.zeros_like(k); k_jm[1:, :] = 2*k[1:, :]*k[:-1, :]/(k[1:, :]+k[:-1, :]) 

        k_ip_v = k_ip.ravel(); k_im_v = k_im.ravel() 
        k_jp_v = k_jp.ravel(); k_jm_v = k_jm.ravel() 

        #Conductances
        G_conv = h_z*area_conv_face 
        G_e = area_e_v*k_ip_v/dr_cell 
        G_w = area_w_v*k_im_v/dr_cell 
        G_n = ring_areas_v*k_jp_v/dz_cell 
        G_s = ring_areas_v*k_jm_v/dz_cell 

        #Add clad-water convection
        p_clad = j*Nr + idx_clad[-1] 
        p_water = j*Nr + idx_water[0] 
        G_e[p_clad] = G_conv 
        G_w[p_water] = G_conv

        #Advection 
        u_full = np.zeros((Nz, Nr)) 
        u_full[:, idx_water] = u_profile 
        u_full_v = u_full.ravel() 
        m_dot_cp = rho_v * u_full_v * ring_areas_v * cp_v 
        adv_south = np.zeros(N) 
        adv_south[water_mask] = m_dot_cp[water_mask] 
        adv_south[idx_water] = 0.0 
        G_s += adv_south 

        #Gas gap 
        fuel_surface = idx_fuel[-1] 
        clad_inner = idx_clad[0] 
        p_fuel = j*Nr + fuel_surface 
        p_clad = j*Nr + clad_inner 
        G_e[p_fuel] = h_gas*area_gas_gap 
        G_w[p_clad] = h_gas*area_gas_gap 

        #Main diagonal 
        G_e[~east_mask] = 0.0 
        G_w[~west_mask] = 0.0 
        G_n[~north_mask] = 0.0 
        G_s[~south_mask] = 0.0 
        G_main = G_s + G_n + G_e + G_w 
        #set inlet row
        G_main[inlet_rows] = 1.0 
        G_e[inlet_rows] = 0.0 
        G_w[inlet_rows] = 0.0 
        G_n[inlet_rows] = 0.0 
        G_s[inlet_rows] = 0.0  
        #Set far right side nodes to previous node
        G_e[east_rows] = 0.0
        G_n[east_rows] = 0.0
        G_s[east_rows] = 0.0
        G_w[east_rows] = 1.0
        G_main[east_rows] = 1.0
        b_true[east_rows] = 0.0

        if np.any(cap_mask): 
            G_main[full_cap_mask] = 1.0 
            G_e[full_cap_mask] = 0.0 
            G_w[full_cap_mask] = 0.0 
            G_n[full_cap_mask] = 0.0 
            G_s[full_cap_mask] = 0.0 
            b_true[full_cap_mask] = T_sat
        
        A_true = A_template.copy() 
        A_true.data[main_idx]  = G_main 
        A_true.data[east_idx]  = -G_e[:-1] 
        A_true.data[west_idx]  = -G_w[1:] 
        A_true.data[north_idx] = -G_n[:-Nr] 
        A_true.data[south_idx] = -G_s[Nr:] 

        return A_true, b_true 
    cancel()

    P_initial = P_initial1
    P_final = P_end
    Period = Period1
    dt_nice = np.array([0.1, 0.2, 0.25, 0.5, 1, 2, 3, 4])
    dt = dt_nice[np.argmin(np.abs(dt_nice - (a_p*Period**2 + b_p*Period + c_p)))]
    if P_initial > P_final:
        Period *= -1

    q_vol = vol_heat_rate(Shape1, P_final) 
    q_prime = np.sum(q_vol*V_cells[:,idx_fuel], axis=1)/dz_cell
    T_curr = initial_temp(P_final) 
    rho = np.zeros_like(T_curr) 
    c_p = np.zeros_like(T_curr) 
    k   = np.zeros_like(T_curr) 
    mu  = np.zeros_like(T_curr) 
    st  = np.zeros_like(T_curr) 

    h_initial = IAPWS97(P=P_w, T=T_curr[0,idx_water[0]]).h*1000 
    h_field = np.full((len(z_nodes), len(idx_water)), h_initial) 

    rho[:,idx_fuel], c_p[:,idx_fuel], k[:,idx_fuel] = fuel_props(T_curr[:,idx_fuel]) 
    rho[:,idx_clad], c_p[:,idx_clad], k[:,idx_clad] = Zirc_props(T_curr[:,idx_clad]) 
    rho[:,idx_water], c_p[:,idx_water], k[:,idx_water], mu[:,idx_water], st[:,idx_water] = water_props(h_field) 

    u_profile = v_profile(rho) 
    h_z, _, h_single, h_two, _ = convective_h(T_curr, u_profile, h_field, rho, c_p, k, mu, st) 

    #Calculate effective thermal conductivity for water for turbulent approximation: 
    k_mol = k[:, idx_water] 
    rho_w = rho[:, idx_water] 
    cp_w  = c_p[:, idx_water] 
    r_w   = r_nodes[idx_water] 

    du_dr = np.zeros_like(u_profile) 
    #interior points 
    ui_p = u_profile[:, 2:] 
    ui_m = u_profile[:, :-2] 
    ri_p = r_w[2:] 
    ri_m = r_w[:-2] 
    du_dr[:, 1:-1] = (ui_p - ui_m) / (ri_p - ri_m) 

    #left boundary (i == idx_water[0]) 
    u0 = u_profile[:, 0] 
    u1 = u_profile[:, 1] 
    u2 = u_profile[:, 2] 
    h = r_nodes[1] - r_nodes[0] 
    du_dr[:, 0] = (-3*u0 + 4*u1 - u2) / (2*h) 

    d_wall = r_w - clad_outr 
    lm = 0.25*d_wall 
    mu_t = rho_w*lm**2*np.abs(du_dr) 
    k[:, idx_water] = k_mol + (mu_t * cp_w) / Pr_T 

    x_field = (h_field - h_f)/h_fg 
    progress()
    cancel()
    A_true, b_true = A_b_matrix(k, rho, c_p, q_vol, h_z, u_profile, T_curr, x_field) 
    T_candidate = spsolve(A_true,b_true).reshape((Nz, Nr))
    relax = 0.5
    max_dT = 30.0  #max temperature change per iteration in K
    dT = T_candidate - T_curr
    dT = np.clip(dT, -max_dT, max_dT)
    T_true = T_curr + relax * dT

    SS = False

    @njit(fastmath=True)
    def update_h(h_field, q_prime, u_profile, rho):

        h_field[:,:] = 0.0
        h_field[0,:] = h_in

        for j in range(1, Nz):

            m_dot = 0.0
            for local_i in range(N_water):
                i = idx_water[local_i]
                m_dot += rho[j,i] * u_profile[j,local_i] * ring_areas[i]

            dh = (q_prime[j] * dz_cell) / m_dot

            for local_i in range(N_water):
                h_field[j,local_i] = h_field[j-1,local_i] + dh

    update_h(h_field, q_prime, u_profile, rho) 

    T_prev = T_true.copy() 
    h_prev = h_field.copy() 

    while not SS: 

        #Store previous fields
        T_curr = T_true.copy() 

        #Update material properties
        rho[:,idx_fuel], c_p[:,idx_fuel], k[:,idx_fuel] = fuel_props(T_curr[:,idx_fuel]) 
        rho[:,idx_clad], c_p[:,idx_clad], k[:,idx_clad] = Zirc_props(T_curr[:,idx_clad]) 
        rho[:,idx_water], c_p[:,idx_water], k[:,idx_water], mu[:,idx_water], st[:,idx_water] = water_props(h_field) 

        #Velocity profile
        u_profile = v_profile(rho)  

        q_chf = CHF(h_field, u_profile, rho) 

        #Convective coefficient
        h_z, ratio, h_single, h_two, _ = convective_h(
            T_curr, u_profile, h_field, rho, c_p, k, mu, st,
            h_single_p = h_single, h_two_p = h_two, q_local=(q_prime/(np.pi*clad_outr*2)), q_chf=q_chf
        ) 

        #Effective turbulent conductivity
        k_mol = k[:, idx_water] 
        rho_w = rho[:, idx_water] 
        cp_w  = c_p[:, idx_water] 
        r_w   = r_nodes[idx_water] 

        du_dr = np.zeros_like(u_profile) 
        du_dr[:, 1:-1] = (u_profile[:, 2:] - u_profile[:, :-2]) / (r_w[2:] - r_w[:-2]) 
        h_r = r_nodes[1] - r_nodes[0] 
        du_dr[:, 0] = (-3*u_profile[:,0] + 4*u_profile[:,1] - u_profile[:,2]) / (2*h_r) 

        d_wall = r_w - clad_outr 
        lm = 0.35 * d_wall 
        mu_t = rho_w * lm**2 * np.abs(du_dr) 
        k[:, idx_water] = k_mol + (mu_t * cp_w) / Pr_T 

        #Build matrix and RHS
        x_field = (h_field - h_f)/h_fg 
        A_true, b_true = A_b_matrix(k, rho, c_p, q_vol, h_z, u_profile, T_prev, x_field) 

        #Solve linear system with under-relaxation and max delta T
        T_candidate = spsolve(A_true,b_true).reshape((Nz, Nr))
        relax = 1.0
        max_dT = 30.0  #max temperature change per iteration in K
        dT = T_candidate - T_prev
        dT = np.clip(dT, -max_dT, max_dT)
        T_true = T_prev + relax * dT

        #Update enthalpy
        update_h(h_field, q_prime, u_profile, rho)  

        #Check convergence
        dT_max = np.max(np.abs(T_true - T_prev)) 
        dH_max = np.max(np.abs(h_field - h_prev)) 

        cancel()
        
        #Convergence check
        if dT_max < 1e-2 and dH_max < 1e-1: 
            SS = True 

        T_prev[:] = T_true 
        h_prev[:] = h_field 
        
    progress()
    cancel()
        
    T_final = T_true

    q_vol = vol_heat_rate(Shape1, P_initial) 
    q_prime = np.sum(q_vol*V_cells[:,idx_fuel], axis=1)/dz_cell
    T_curr = initial_temp(P_initial) 
    rho = np.zeros_like(T_curr) 
    c_p = np.zeros_like(T_curr) 
    k   = np.zeros_like(T_curr) 
    mu  = np.zeros_like(T_curr) 
    st  = np.zeros_like(T_curr) 

    h_initial = IAPWS97(P=P_w, T=T_curr[0,idx_water[0]]).h*1000 
    h_field = np.full((len(z_nodes), len(idx_water)), h_initial) 

    rho[:,idx_fuel], c_p[:,idx_fuel], k[:,idx_fuel] = fuel_props(T_curr[:,idx_fuel]) 
    rho[:,idx_clad], c_p[:,idx_clad], k[:,idx_clad] = Zirc_props(T_curr[:,idx_clad]) 
    rho[:,idx_water], c_p[:,idx_water], k[:,idx_water], mu[:,idx_water], st[:,idx_water] = water_props(h_field) 

    u_profile = v_profile(rho) 
    h_z, _, h_single, h_two, _ = convective_h(T_curr, u_profile, h_field, rho, c_p, k, mu, st) 

    #Calculate effective thermal conductivity for water for turbulent approximation: 
    k_mol = k[:, idx_water] 
    rho_w = rho[:, idx_water] 
    cp_w  = c_p[:, idx_water] 
    r_w   = r_nodes[idx_water] 

    du_dr = np.zeros_like(u_profile) 
    #interior points 
    ui_p = u_profile[:, 2:] 
    ui_m = u_profile[:, :-2] 
    ri_p = r_w[2:] 
    ri_m = r_w[:-2] 
    du_dr[:, 1:-1] = (ui_p - ui_m) / (ri_p - ri_m) 

    #left boundary (i == idx_water[0]) 
    u0 = u_profile[:, 0] 
    u1 = u_profile[:, 1] 
    u2 = u_profile[:, 2] 
    h = r_nodes[1] - r_nodes[0] 
    du_dr[:, 0] = (-3*u0 + 4*u1 - u2) / (2*h) 

    d_wall = r_w - clad_outr 
    lm = 0.25*d_wall 
    mu_t = rho_w*lm**2*np.abs(du_dr) 
    k[:, idx_water] = k_mol + (mu_t * cp_w) / Pr_T 

    x_field = (h_field - h_f)/h_fg 
    progress()
    cancel()
    A_true, b_true = A_b_matrix(k, rho, c_p, q_vol, h_z, u_profile, T_curr, x_field) 
    T_candidate = spsolve(A_true,b_true).reshape((Nz, Nr))
    relax = 0.5
    max_dT = 30.0  #max temperature change per iteration in K
    dT = T_candidate - T_curr
    dT = np.clip(dT, -max_dT, max_dT)
    T_true = T_curr + relax * dT

    SS = False

    update_h(h_field, q_prime, u_profile, rho) 

    T_prev = T_true.copy() 
    h_prev = h_field.copy() 

    while not SS:  

        #Store previous fields
        T_curr = T_true.copy()  

        #Update material properties
        rho[:,idx_fuel], c_p[:,idx_fuel], k[:,idx_fuel] = fuel_props(T_curr[:,idx_fuel]) 
        rho[:,idx_clad], c_p[:,idx_clad], k[:,idx_clad] = Zirc_props(T_curr[:,idx_clad]) 
        rho[:,idx_water], c_p[:,idx_water], k[:,idx_water], mu[:,idx_water], st[:,idx_water] = water_props(h_field) 

        #Velocity profile
        u_profile = v_profile(rho) 

        q_chf = CHF(h_field, u_profile, rho) 

        #Convective coefficient
        h_z, ratio, h_single, h_two, _ = convective_h(
            T_curr, u_profile, h_field, rho, c_p, k, mu, st,
            h_single_p = h_single, h_two_p = h_two, q_local=(q_prime/(np.pi*clad_outr*2)), q_chf=q_chf
        ) 

        #Effective turbulent conductivity
        k_mol = k[:, idx_water] 
        rho_w = rho[:, idx_water] 
        cp_w  = c_p[:, idx_water] 
        r_w   = r_nodes[idx_water] 

        du_dr = np.zeros_like(u_profile) 
        du_dr[:, 1:-1] = (u_profile[:, 2:] - u_profile[:, :-2]) / (r_w[2:] - r_w[:-2]) 
        h_r = r_nodes[1] - r_nodes[0] 
        du_dr[:, 0] = (-3*u_profile[:,0] + 4*u_profile[:,1] - u_profile[:,2]) / (2*h_r) 

        d_wall = r_w - clad_outr 
        lm = 0.35 * d_wall 
        mu_t = rho_w * lm**2 * np.abs(du_dr) 
        k[:, idx_water] = k_mol + (mu_t * cp_w) / Pr_T 

        #Build matrix and RHS
        x_field = (h_field - h_f)/h_fg 
        A_true, b_true = A_b_matrix(k, rho, c_p, q_vol, h_z, u_profile, T_prev, x_field)

        #Solve linear system with under-relaxation and max delta T
        T_candidate = spsolve(A_true,b_true).reshape((Nz, Nr))
        relax = 1.0
        max_dT = 30.0  #max temperature change per iteration in K
        dT = T_candidate - T_prev
        dT = np.clip(dT, -max_dT, max_dT)
        T_true = T_prev + relax * dT

        #Update enthalpy
        update_h(h_field, q_prime, u_profile, rho)  

        #Check convergence
        dT_max = np.max(np.abs(T_true - T_prev)) 
        dH_max = np.max(np.abs(h_field - h_prev)) 

        cancel()

        #Convergence check
        if dT_max < 1e-2 and dH_max < 1e-1: 
            SS = True 

        T_prev[:] = T_true 
        h_prev[:] = h_field
        
    progress()
    cancel()
        
    v_max_fuel = np.max(np.maximum(T_final[:,idx_fuel], T_true[:,idx_fuel]))-273.15
    v_max_clad = np.max(np.maximum(T_final[:,idx_clad], T_true[:,idx_clad]))-273.15
    v_max_wat = np.max(np.maximum(T_final[:,idx_water], T_true[:,idx_water]))-273.15
    v_min_fuel = np.min(np.minimum(T_final[:,idx_fuel], T_true[:,idx_fuel]))-273.15
    v_min_clad = np.min(np.minimum(T_final[:,idx_clad], T_true[:,idx_clad]))-273.15

    #Initialize time
    time_solver = True
    time_break = 0
    frame_counter = 0
    P_reactor = P_initial
    frames = []

    progress()
    cancel()

    while time_solver:

        #Update material properties
        rho[:,idx_fuel], c_p[:,idx_fuel], k[:,idx_fuel] = fuel_props(T_prev[:,idx_fuel])
        rho[:,idx_clad], c_p[:,idx_clad], k[:,idx_clad] = Zirc_props(T_prev[:,idx_clad])
        rho[:,idx_water], c_p[:,idx_water], k[:,idx_water], mu[:,idx_water], st[:,idx_water] = water_props(h_field)
        rho_v = rho.ravel()
        cp_v  = c_p.ravel()

        #Velocity profile
        u_profile = v_profile(rho)

        q_chf = CHF(h_field, u_profile, rho)

        #Convective coefficient
        h_z, ratio, h_single, h_two, q_chf = convective_h(
            T_prev, u_profile, h_field, rho, c_p, k, mu, st,
            h_single_p=h_single, h_two_p=h_two, q_local=(q_prime/(np.pi*clad_outr*2)), q_chf=q_chf
        )

        #Effective turbulent conductivity
        k_mol = k[:, idx_water]
        rho_w = rho[:, idx_water]
        cp_w  = c_p[:, idx_water]
        r_w   = r_nodes[idx_water]

        du_dr = np.zeros_like(u_profile)
        du_dr[:, 1:-1] = (u_profile[:, 2:] - u_profile[:, :-2]) / (r_w[2:] - r_w[:-2])
        h_r = r_nodes[1] - r_nodes[0]
        du_dr[:, 0] = (-3*u_profile[:,0] + 4*u_profile[:,1] - u_profile[:,2]) / (2*h_r)

        d_wall = r_w - clad_outr
        lm = 0.35 * d_wall
        mu_t = rho_w * lm**2 * np.abs(du_dr)
        k[:, idx_water] = k_mol + (mu_t * cp_w) / Pr_T

        #Build matrix and RHS
        x_field = (h_field - h_f)/h_fg
        A_true, b_true = A_b_matrix(k, rho, c_p, q_vol, h_z, u_profile, T_prev, x_field, in_time=True)

        #Implicit Euler solve: (M/dt + A) T_next = M/dt * T_prev + b
        T_prev_v = T_prev.ravel()
        M_over_dt = diags(rho_v * cp_v * V_v / dt)  #mass matrix / dt
        lhs = M_over_dt + A_true
        rhs = M_over_dt.dot(T_prev_v) + b_true

        T_next_v = spsolve(lhs, rhs)
        T_true = T_next_v.reshape((Nz, Nr))

        #Update enthalpy
        update_h(h_field, q_prime, u_profile, rho)

        #Save frame
        frame_data = {
            'T': T_true.copy(),
            'h': h_field.copy(),
            'ratio': ratio.copy(),
            'time': time_sol,
            'chf': q_chf.copy(),
            'local': (q_prime.copy()/(np.pi*clad_outr*2)),
            'power': P_reactor
        }
        frames.append(frame_data)

        cancel()

        dT_max = np.max(np.abs(T_final - T_true))
        dT_maxi = np.max(np.abs(T_true - T_prev))
        if dT_max < 1e-2 or (dT_maxi < 0.5 and P_reactor == P_final) or (time_break > 30):
            time_solver = False
        
        if P_reactor == P_final:
            time_break += 1

        frame_counter += 1
        T_prev[:] = T_true
        h_prev[:] = h_field

        time_sol += dt
        P_reactor = power(P_initial, P_final, Period, time_sol)
        q_vol = vol_heat_rate(Shape1, P_reactor)
        q_prime = np.sum(q_vol*V_cells[:,idx_fuel], axis=1)/dz_cell
    progress()
    cancel()

    #Stack original frames into arrays
    T_stack = np.array([f['T'] for f in frames])        # shape: (n_frames, Nz, Nr)
    h_stack = np.array([f['h'] for f in frames])
    ratio_stack = np.array([f['ratio'] for f in frames])
    time_stack = np.array([f['time'] for f in frames])
    q_chf = np.array([f['chf'] for f in frames])
    q_local = np.array([f['local'] for f in frames])
    P_reactor = np.array([f['power'] for f in frames])

    progress()
    cancel()
    return T_stack, h_stack, ratio_stack, time_stack, T_true, v_min_fuel, v_max_fuel, v_min_clad, v_max_clad, \
            v_max_wat, q_chf, q_local, P_reactor
    
def generate_image(plot_widget, cbar_panel, chf_graph, T_stack, h_stack, ratio_stack, time_stack,
                   T_true, v_min_fuel, v_max_fuel, v_min_clad, v_max_clad, v_max_wat, q_chf, q_local, power):

    #Geometry
    fuel_outr = 0.00475
    clad_outr = 0.00532
    flow_outr = 0.00630
    fuel_z = 3.88112

    r_faces = np.linspace(0, flow_outr, 150)
    z_faces = np.linspace(0, fuel_z, 300)
    r_nodes = 0.5*(r_faces[:-1]+r_faces[1:])
    z_nodes = 0.5*(z_faces[:-1]+z_faces[1:])
    Nr, Nz = len(r_nodes), len(z_nodes)
    dr = r_nodes[1] - r_nodes[0]
    dz = z_nodes[1] - z_nodes[0]

    fuel_end = np.searchsorted(r_nodes, fuel_outr, side="right")
    clad_end = np.searchsorted(r_nodes, clad_outr, side="right")
    idx_fuel = np.arange(fuel_end)
    idx_clad = np.arange(fuel_end, clad_end)
    idx_water = np.arange(clad_end, Nr)

    #Thermo
    h_f = IAPWS97(P=15.5132, x=0).h*1000
    h_g = IAPWS97(P=15.5132, x=1).h*1000
    h_fg = h_g - h_f
    T_sat = IAPWS97(P=15.5132, x=0).T

    #Time
    fps = 10.0
    dt_new = 1.0/fps
    time_new = np.arange(time_stack[0], time_stack[-1], dt_new)
    n_frames = len(time_new)
    chunk = 300

    #Interpolators
    T_interp_func = interp1d(time_stack, T_stack, axis=0, bounds_error=False, fill_value=(T_stack[0], T_stack[-1]))
    h_interp_func = interp1d(time_stack, h_stack, axis=0, bounds_error=False, fill_value=(h_stack[0], h_stack[-1]))
    ratio_interp_func = interp1d(time_stack, ratio_stack, axis=0, bounds_error=False, fill_value=(ratio_stack[0], ratio_stack[-1]))
    q_local_interp_func = interp1d(time_stack, q_local, axis=0, bounds_error=False, fill_value=(q_local[0], q_local[-1]))
    q_chf_interp_func = interp1d(time_stack, q_chf, axis=0, bounds_error=False, fill_value=(q_chf[0], q_chf[-1]))
    power_interp_func = interp1d(time_stack, power, axis=0, bounds_error=False, fill_value=(power[0], power[-1]))

    T_chunk = h_chunk = ratio_chunk = q_local_chunk = q_chf_chunk = power_chunk = None
    chunk_start = 0
    def load_chunk(start):
        nonlocal T_chunk, h_chunk, ratio_chunk, q_local_chunk, q_chf_chunk, power_chunk, chunk_start
        end = min(start+chunk, n_frames)
        t_slice = time_new[start:end]
        T_chunk = T_interp_func(t_slice)
        h_chunk = h_interp_func(t_slice)
        ratio_chunk = ratio_interp_func(t_slice)
        q_local_chunk = q_local_interp_func(t_slice)
        q_chf_chunk = q_chf_interp_func(t_slice)
        power_chunk = power_interp_func(t_slice)
        chunk_start = start

    #Two-phase fraction
    x_two = np.ones((Nz, len(idx_water)), dtype=np.float32)*-1.0
    h_frame0 = h_stack[0]
    for j in range(Nz):
        for ii in range(len(idx_water)):
            if T_true[j, ii+idx_water[0]] >= T_sat-0.75:
                x_two[j, ii] = max((h_frame0[j, ii]-h_f)/h_fg, 0.01)

    #Plot
    plot_widget.clear()
    plot_widget.setLabel('bottom', "Radius [m]")
    plot_widget.setLabel('left', "Axial z [m]")

    img_fuel = pg.ImageItem()
    img_clad = pg.ImageItem()
    img_water = pg.ImageItem()
    img_quality = pg.ImageItem(opacity=0.4)
    img_fuel.setColorMap(pg.colormap.get('inferno'))
    img_clad.setColorMap(pg.colormap.get('magma'))
    img_water.setColorMap(pg.colormap.get('inferno'))
    img_quality.setColorMap(pg.colormap.get('viridis'))
    
    #Set physical rects (z vertical, r horizontal)
    def set_rect(img, r_idx):
        r0, r1 = r_nodes[r_idx[0]], r_nodes[r_idx[-1]]
        img.setRect(pg.QtCore.QRectF(r0, 0, r1-r0, fuel_z))

    set_rect(img_fuel, idx_fuel)
    set_rect(img_clad, idx_clad)
    set_rect(img_water, idx_water)
    set_rect(img_quality, idx_water)
    
    plot_widget.addItem(img_fuel)
    plot_widget.addItem(img_clad)
    plot_widget.addItem(img_water)
    plot_widget.addItem(img_quality)

    cbar_panel.clear()

    #Fuel
    lbl_fuel = pg.LabelItem("Fuel Temp [°C]", angle=0)
    lut_fuel = pg.HistogramLUTItem()
    lut_fuel.gradient.loadPreset("inferno")
    lut_fuel.setImageItem(img_fuel)
    lut_fuel.setLevels(v_min_fuel, v_max_fuel)

    #Clad
    lbl_clad = pg.LabelItem("Clad Temp [°C]", angle=0)
    lut_clad = pg.HistogramLUTItem()
    lut_clad.gradient.loadPreset("magma")
    lut_clad.setImageItem(img_clad)
    lut_clad.setLevels(v_min_clad, v_max_clad)

    #Water
    lbl_wat = pg.LabelItem("Water Temp [°C]", angle=0)
    lut_wat = pg.HistogramLUTItem()
    lut_wat.gradient.loadPreset("inferno")
    lut_wat.setImageItem(img_water)
    lut_wat.setLevels(289.33, v_max_wat)

    #Stack vertically
    for lbl, lut in [(lbl_fuel, lut_fuel),
                    (lbl_clad, lut_clad),
                    (lbl_wat, lut_wat)]:
        cbar_panel.addItem(lbl)
        cbar_panel.nextRow()
        cbar_panel.addItem(lut)
        cbar_panel.nextRow()
        
    lut_fuel.plot.hide()
    lut_clad.plot.hide()
    lut_wat.plot.hide()

    #Markers
    onb_scatter = pg.ScatterPlotItem(pen='k', brush='w', size=8)
    osb_scatter = pg.ScatterPlotItem(pen='k', brush='g', size=8)
    chf_scatter = pg.ScatterPlotItem(pen='k', brush='r', size=8)
    plot_widget.addItem(onb_scatter)
    plot_widget.addItem(osb_scatter)
    plot_widget.addItem(chf_scatter)
    legend = plot_widget.addLegend(offset=(10, 10))
    legend.addItem(onb_scatter, "ONB (white)")
    legend.addItem(osb_scatter, "OSB (green)")
    legend.addItem(chf_scatter, "CHF (red)")
    
    #Side chf graph definition:
    chf_graph.clear()
    plot = chf_graph.addPlot()
    plot.setLabel('bottom', "Axial z [m]")
    plot.setLabel('left',   "Heat Flux [W/m²]")

    curve_local = plot.plot(z_nodes, np.zeros_like(z_nodes), pen=pg.mkPen('w', width=2), name='Local q"')
    curve_chf = plot.plot(z_nodes, np.zeros_like(z_nodes), pen=pg.mkPen('r', width=2), name='Predicted q"')

    legend = pg.LegendItem(offset=(-30, 30))
    legend.setParentItem(plot.graphicsItem())

    legend.addItem(curve_local, 'Local q"')
    legend.addItem(curve_chf, 'Predicted q"')

    load_chunk(0)

    hatch_lines = []  #store all the temporary line items

    def update(i):
        nonlocal T_chunk, h_chunk, ratio_chunk, q_local_chunk, q_chf_chunk, power_chunk, chunk_start, hatch_lines
        if i < chunk_start or i >= chunk_start + T_chunk.shape[0]:
            load_chunk((i // chunk) * chunk)
        local_i = i - chunk_start
        T_frame = T_chunk[local_i]
        h_frame = h_chunk[local_i]
        ratio_frame = ratio_chunk[local_i]
        q_local_frame = q_local_chunk[local_i]
        q_chf_frame = q_chf_chunk[local_i]
        power_frame = power_chunk[local_i]
        T_C = T_frame - 273.15

        #Water x fraction
        water_T = T_frame[:, idx_water]
        boiling_mask = water_T >= (T_sat - 0.75)
        x_two[:] = -1.0

        x_vals = np.maximum((h_frame[boiling_mask] - h_f) / h_fg, 0.01)
        x_two[boiling_mask] = np.minimum(x_vals, 1.0)

        #Fuel
        x0, x1 = r_nodes[idx_fuel[0]], r_nodes[idx_fuel[-1]]
        y0, y1 = z_nodes[0], z_nodes[-1]
        img_fuel.setImage(T_C[:, idx_fuel].T, autoLevels=False, levels=(v_min_fuel, v_max_fuel))
        img_fuel.setRect(pg.QtCore.QRectF(x0, y0, x1-x0, y1-y0))

        #Clad
        x0, x1 = r_nodes[idx_clad[0]], r_nodes[idx_clad[-1]]
        img_clad.setImage(T_C[:, idx_clad].T, autoLevels=False, levels=(v_min_clad, v_max_clad))
        img_clad.setRect(pg.QtCore.QRectF(x0, y0, x1-x0, y1-y0))

        #Water
        x0, x1 = r_nodes[idx_water[0]], r_nodes[idx_water[-1]]
        img_water.setImage(T_C[:, idx_water].T, autoLevels=False, levels=(T_C[:, idx_water].min(), v_max_wat))
        img_water.setRect(pg.QtCore.QRectF(x0, y0, x1-x0, y1-y0))

        #Quality overlay
        img_quality.setImage((x_two > -1).astype(float).T, autoLevels=False, levels=(0,1))
        img_quality.setRect(pg.QtCore.QRectF(x0, y0, x1-x0, y1-y0))

        #Hatch lines for real two-phase
        for line in hatch_lines:
            plot_widget.removeItem(line)
        hatch_lines.clear()

        ys, xs = np.where((x_two > 0) & (x_two < 1.0))
        n_lines = 4
        if len(xs) > 0:
            all_x = []
            all_y = []
            k_offsets = np.linspace(0, dr, n_lines + 1)
            for y_idx, x_idx in zip(ys, xs):
                x0_cell = r_nodes[idx_water[x_idx]]
                y0_cell = z_nodes[y_idx]
                y1_cell = y0_cell + dz
                for k in range(n_lines):
                    all_x.extend([x0_cell + k_offsets[k], x0_cell + 0.5*k_offsets[k+1], np.nan])
                    all_y.extend([y0_cell, y1_cell, np.nan])
            hatch_item = pg.PlotDataItem(all_x, all_y, pen=pg.mkPen('k', width=1))
            plot_widget.addItem(hatch_item)
            hatch_lines.append(hatch_item)

        #Markers
        j_onb = np.argmax(T_frame[:, idx_clad[-1]] >= T_sat)
        if not (T_frame[j_onb, idx_clad[-1]] >= T_sat):
            j_onb = None

        j_osb = np.argmax(T_frame[:, idx_water[0]] >= T_sat)
        if not (T_frame[j_osb, idx_water[0]] >= T_sat):
            j_osb = None

        j_chf = np.argmax(ratio_frame <= 1)
        if not (ratio_frame[j_chf] <= 1):
            j_chf = None

        onb_scatter.setData([clad_outr] if j_onb is not None else [], 
                            [z_nodes[j_onb]] if j_onb is not None else [])
        osb_scatter.setData([clad_outr] if j_osb is not None else [], 
                            [z_nodes[j_osb]] if j_osb is not None else [])
        chf_scatter.setData([clad_outr] if j_chf is not None else [], 
                            [z_nodes[j_chf]] if j_chf is not None else [])

        plot_widget.setTitle(
            f"Time={time_new[i]:.1f}s | Power={(power_frame/1e6):.1f}MW | Max Fuel={np.max(T_C[:, idx_fuel]):.1f}C "
            f"| Max Clad={np.max(T_C[:, idx_clad]):.1f}C | Max Water={np.max(T_C[:, idx_water]):.1f}C"
        )
        
        curve_local.setData(z_nodes, q_local_frame)
        curve_chf.setData(z_nodes, q_chf_frame)
        plot.setTitle(f'Heat Flux | Min DNBR ratio={np.min(q_chf_frame/q_local_frame):.2f}')

    #Timer
    timer = pg.QtCore.QTimer()
    timer.frame = 0
    def step():
        update(timer.frame)
        timer.frame += 1
        if timer.frame >= n_frames:
            timer.frame = 0
    timer.timeout.connect(step)
    timer.start(int(1000/fps))
    return timer
    
#if __name__ == '__main__':
    #main(100, 3700, 5, 'Outlet', 4.0)