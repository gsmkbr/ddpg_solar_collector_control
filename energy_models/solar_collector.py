from CoolProp.CoolProp import PropsSI
from math import pi

# list of constants
EPSILON_COV = 0.9
EPSILON_R = 0.2
A_RO = 1.715
A_RI = 1.617
A_CO = 2.818
A_CI = 2.671
A_AP = 39.0
D_RI = 0.066
D_CO = 0.115
RHO_CON = 0.83
GAMMA = 0.99
TAU = 0.95
ALPHA = 0.96
SIGMA = 5.67e-8
PI = pi
HTF_FLUID = "INCOMP::S800"  #'INCOMP::TVP1'


def calc_solar_outlet_temp(theta, v_wind, t_in, p_in, t_amb, m_dot_htf, g_b):

    k_htf = PropsSI('L', 'P', p_in, 'T', t_in, HTF_FLUID)
    c_p_htf = PropsSI('C', 'P', p_in, 'T', t_in, HTF_FLUID)
    mu_htf = PropsSI('V', 'P', p_in, 'T', t_in, HTF_FLUID)


    k_theta = 1-2.2307e-4*theta-1.1e-4*theta**2+3.18596e-6*theta**3-4.85509e-8*theta**4
    eta_opt = k_theta*RHO_CON*GAMMA*TAU*ALPHA

    epsilon_star_r = (1/EPSILON_R+(1-EPSILON_COV)*(A_RO/A_CI)/EPSILON_COV)**(-1)
    
    re_htf = 4*m_dot_htf/(PI*mu_htf*D_RI)
    pr_htf = mu_htf*c_p_htf/k_htf
    # print(f"pr_htf: {pr_htf}, re_htf: {re_htf}")
    h_fm = 0.023*k_htf*(re_htf**0.8*pr_htf**0.4)/D_RI

    #h_cov_o = 4*v_wind**0.58/(D_CO**(-0.42))
    h_cov_o = 10

    k5 = 4*A_CO*EPSILON_COV*SIGMA*t_amb**3+A_CO*h_cov_o
    k3 = A_RO*epsilon_star_r*SIGMA*(1+(4*t_amb**3*A_RO*epsilon_star_r*SIGMA)/k5)**(-1)
    k4 = (1/(h_fm*A_RI)+1/(2*m_dot_htf*c_p_htf))**(-1)
    k1 = eta_opt * (1+(4*t_in**3*k3)/k4)**(-1)
    k2 = k3*(1+(4*t_in**3*k3)/k4)**(-1)

    q_dot_s = A_AP*g_b
    t_out = t_in+q_dot_s*k1/(m_dot_htf*c_p_htf)-k2/(m_dot_htf*c_p_htf)*(t_in**4-t_amb**4)
    q_dot_in = k1*q_dot_s-k2*(t_in**4-t_amb**4)

    eta_th = k1-k2*(t_in**4-t_amb**4)/(A_AP*g_b)
    return t_out, q_dot_in, eta_th

t_htf_in = 500
p_htf_in = 1e5
t_ambient = 300
volumetric_flow_rate = 0.00001
m_dot_htf = PropsSI('D', 'P', p_htf_in, 'T', t_htf_in, HTF_FLUID)*volumetric_flow_rate
input_solar_flux = 30
t_htf_out, absorbed_heat, efficiency = calc_solar_outlet_temp(
    theta=0, 
    v_wind=20, 
    t_in=t_htf_in, 
    p_in=p_htf_in, 
    t_amb=t_ambient, 
    m_dot_htf=m_dot_htf,
    g_b=input_solar_flux)


print(f"Outlet temperature: {t_htf_out}\nHeat absorption: {absorbed_heat}\nEfficiency: {efficiency}")