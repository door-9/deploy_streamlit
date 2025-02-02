import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.stats import probplot

# 페이지 설정
st.set_page_config(layout="wide", page_title="PK/PD Analyzer Pro")
st.title("Advanced Pharmacokinetic Modeling")

# 세션 상태 초기화
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['Time', 'Concentration'])
if 'params' not in st.session_state:
    st.session_state.params = {
        'model_type': '1-Compartment',
        'admin_route': 'IV',
        'dose': 100.0,
        'CL': 5.0,
        'Vc': 10.0,
        'CLd': 2.0,
        'Vp': 20.0,
        'Ka': 1.5,
        'F': 1.0,
        'step_size': 0.1
    }
if 'param_bounds' not in st.session_state:
    st.session_state.param_bounds = {
        'CL': (0.1, 20.0),
        'Vc': (1.0, 100.0),
        'CLd': (0.1, 10.0),
        'Vp': (5.0, 200.0),
        'Ka': (0.1, 5.0),
        'F': (0.1, 1.0)
    }

# ODE 시스템 정의
def pk_ode(t, y):
    p = st.session_state.params
    model = p['model_type']
    route = p['admin_route']
    
    if model == '1-Compartment':
        if route == 'IV':
            dA_c = -(p['CL']/p['Vc']) * y[0]
            return [dA_c]
        else:  # PO
            dA_g = -p['Ka'] * y[0]
            dA_c = p['Ka'] * y[0] - (p['CL']/p['Vc']) * y[1]
            return [dA_g, dA_c]
    else:  # 2-Compartment
        if route == 'IV':
            C_c = y[0] / p['Vc']
            C_p = y[1] / p['Vp']
            dA_c = p['CLd']*(C_p - C_c) - p['CL']*C_c
            dA_p = p['CLd']*(C_c - C_p)
            return [dA_c, dA_p]
        else:  # PO
            C_c = y[1] / p['Vc']
            C_p = y[2] / p['Vp']
            dA_g = -p['Ka'] * y[0]
            dA_c = p['Ka'] * y[0] + p['CLd']*(C_p - C_c) - p['CL']*C_c
            dA_p = p['CLd']*(C_c - C_p)
            return [dA_g, dA_c, dA_p]

def solve_ode():
    p = st.session_state.params
    try:
        # 초기 조건 설정
        if p['admin_route'] == 'IV':
            if p['model_type'] == '1-Compartment':
                y0 = [p['dose']]
            else:
                y0 = [p['dose'], 0]
        else:  # PO
            dose = p['dose'] * p['F']
            if p['model_type'] == '1-Compartment':
                y0 = [dose, 0]
            else:
                y0 = [dose, 0, 0]

        t_max = max(st.session_state.data['Time'])*1.5 if not st.session_state.data.empty else 24
        t_eval = np.arange(0, t_max + p['step_size'], p['step_size'])
        
        # RK4 Solver
        y = np.zeros((len(t_eval), len(y0)))
        y[0] = y0
        for i in range(1, len(t_eval)):
            h = p['step_size']
            k1 = pk_ode(t_eval[i-1], y[i-1])
            k2 = pk_ode(t_eval[i-1] + h/2, y[i-1] + h/2*np.array(k1))
            k3 = pk_ode(t_eval[i-1] + h/2, y[i-1] + h/2*np.array(k2))
            k4 = pk_ode(t_eval[i-1] + h, y[i-1] + h*np.array(k3))
            y[i] = y[i-1] + h/6*(np.array(k1) + 2*np.array(k2) + 2*np.array(k3) + np.array(k4))
        
        return t_eval, y
    except Exception as e:
        st.error(f"ODE Solver Error: {str(e)}")
        return None, None

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ Model Configuration")
    st.session_state.params['model_type'] = st.selectbox(
        "Model Type", ["1-Compartment", "2-Compartment"])
    st.session_state.params['admin_route'] = st.selectbox(
        "Administration Route", ["IV", "PO"])
    
    st.subheader("Dosing Parameters")
    st.session_state.params['dose'] = st.number_input(
        "Dose", 0.0, 10e+10, 100.0, 0.1)
    
    st.subheader("PK Parameters & Bounds")
    
    def param_input(param, default, min_val, max_val):
        cols = st.columns([3,1,1])
        with cols[0]:
            st.session_state.params[param] = st.number_input(
                f"{param}", 
                min_val, max_val, default, 0.1,
                key=f"val_{param}"
            )
        with cols[1]:
            min_bound = st.number_input(
                "Min", min_val, max_val, st.session_state.param_bounds[param][0], 0.1,
                key=f"min_{param}"
            )
        with cols[2]:
            max_bound = st.number_input(
                "Max", min_val, max_val, st.session_state.param_bounds[param][1], 0.1,
                key=f"max_{param}"
            )
        st.session_state.param_bounds[param] = (min_bound, max_bound)
    
    param_input('CL', 5.0, 0.0, 10e+10)
    param_input('Vc', 10.0, 0.0, 10e+10)
    
    if st.session_state.params['model_type'] == '2-Compartment':
        param_input('CLd', 2.0, 0.0, 10e+10)
        param_input('Vp', 20.0, 0.0, 10e+10)
    
    if st.session_state.params['admin_route'] == 'PO':
        param_input('Ka', 1.5, 0.0, 10e+3)
        param_input('F', 1.0, 0.0, 1.0)
    
    st.subheader("Solver Settings")
    st.session_state.params['step_size'] = st.number_input(
        "Step Size (h)", 0.01, 2.0, 0.1, 0.01)

# 메인 화면
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📥 Data Input")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
            if {'Time', 'Concentration'}.issubset(df.columns):
                st.session_state.data = df.sort_values('Time')
                st.success(f"Loaded {len(df)} data points")
            else:
                st.error("Missing required columns: Time & Concentration")
        except Exception as e:
            st.error(f"File Error: {str(e)}")
    
    with st.expander("Manual Data Entry"):
        edited_df = st.data_editor(
            st.session_state.data,
            num_rows="dynamic",
            column_config={
                "Time": {"format": "%.2f"},
                "Concentration": {"format": "%.4f"}
            },
            height=300
        )
        st.session_state.data = edited_df

with col2:
    st.subheader("📊 Simulation Results")
    t_sim, y_sim = solve_ode()
    
    # 농도 계산
    conc_sim = []
    if y_sim is not None:
        p = st.session_state.params
        if p['admin_route'] == 'IV':
            conc_sim = y_sim[:,0] / p['Vc']
        else:
            conc_sim = y_sim[:,1] / p['Vc']

    # 메인 플롯
    fig = plt.figure(figsize=(10, 4))
    if not st.session_state.data.empty:
        t_obs = st.session_state.data['Time'].values
        c_obs = st.session_state.data['Concentration'].values
        plt.scatter(t_obs, c_obs, c='red', label='Observed')
    if len(conc_sim) > 0:
        plt.plot(t_sim, conc_sim, 'b--', label='Simulated')
    plt.xlabel('Time (h)')
    plt.ylabel('Concentration (mg/L)')
    plt.title('Concentration-Time Profile')
    plt.legend()
    plt.grid(alpha=0.3)
    st.pyplot(fig)

    # 파라미터 추정
    if st.button("🔍 Run Parameter Estimation"):
        if len(st.session_state.data) < 3:
            st.warning("Minimum 3 data points required")
            st.stop()
            
        try:
            p = st.session_state.params
            t_obs = st.session_state.data['Time'].values
            c_obs = st.session_state.data['Concentration'].values

            # 추정 파라미터 선택
            param_info = []
            if p['model_type'] == '1-Compartment':
                param_info += [('CL', p['CL']), ('Vc', p['Vc'])]
                if p['admin_route'] == 'PO':
                    param_info += [('Ka', p['Ka']), ('F', p['F'])]
            else:
                param_info += [('CL', p['CL']), ('Vc', p['Vc']), 
                              ('CLd', p['CLd']), ('Vp', p['Vp'])]
                if p['admin_route'] == 'PO':
                    param_info += [('Ka', p['Ka']), ('F', p['F'])]

            # 경계값 설정
            bounds = (
                [st.session_state.param_bounds[name][0] for name, _ in param_info],
                [st.session_state.param_bounds[name][1] for name, _ in param_info]
            )
            x0 = [val for _, val in param_info]
            param_names = [name for name, _ in param_info]

            def residual(params):
                try:
                    # 파라미터 업데이트
                    for name, val in zip(param_names, params):
                        p[name] = val

                    # 시뮬레이션 실행
                    t_sim, y_sim = solve_ode()
                    if p['admin_route'] == 'IV':
                        c_sim = y_sim[:,0] / p['Vc']
                    else:
                        c_sim = y_sim[:,1] / p['Vc']

                    # 보간
                    f = interp1d(t_sim, c_sim, bounds_error=False, fill_value=0)
                    return f(t_obs) - c_obs
                except:
                    return np.ones_like(c_obs)*1e6

            # 최적화 실행
            result = least_squares(residual, x0=x0, bounds=bounds, method='trf')

            # 결과 저장
            for name, val in zip(param_names, result.x):
                p[name] = val

            # 리포트 데이터 생성
            t_sim, y_sim = solve_ode()
            c_sim = y_sim[:,1]/p['Vc'] if p['admin_route']=='PO' else y_sim[:,0]/p['Vc']
            f = interp1d(t_sim, c_sim, bounds_error=False, fill_value=0)
            c_pred = f(t_obs)
            
            st.session_state.report = {
                'params': p.copy(),
                't_sim': t_sim,
                'c_sim': c_sim,
                't_obs': t_obs,
                'c_obs': c_obs,
                'c_pred': c_pred,
                'residuals': c_obs - c_pred,
                'metrics': {
                    'R²': 1 - np.sum((c_obs - c_pred)**2)/np.sum((c_obs - np.mean(c_obs))**2),
                    'RMSE': np.sqrt(np.mean((c_obs - c_pred)**2)),
                    'AIC': len(c_obs)*np.log(np.sum((c_obs - c_pred)**2)/len(c_obs)) + 2*len(result.x)
                }
            }
            st.success("Parameter estimation completed!")
            st.rerun()
        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")

# 리포트 표시
if 'report' in st.session_state:
    st.divider()
    st.subheader("📝 Analysis Report")
    subcol1, subcol2 = st.columns([1, 2])
    with subcol1:
        st.write("Model Parameters")
        
        # 파라미터 테이블
        p = st.session_state.report['params']
        param_df = pd.DataFrame({
            'Parameter': ['CL', 'Vc', 'CLd', 'Vp', 'Ka ', 'F'],
            'Value': [p['CL'], p['Vc'], p.get('CLd', np.nan), 
                    p.get('Vp', np.nan), p.get('Ka', np.nan), p.get('F', np.nan)],
        })
        if p['model_type'] == '1-Compartment':
            param_df = param_df[~param_df['Parameter'].isin(['CLd', 'Vp'])]
        if p['admin_route'] == 'IV':
            param_df = param_df[~param_df['Parameter'].isin(['Ka', 'F'])]
        st.dataframe(param_df, hide_index=True)

    with subcol2:
        st.write("Model Diagnostics")

        # 성능 지표
        evaluation_df = pd.DataFrame({
            'Evaluation Metric': ['R²', 'RMSE', 'AIC'],
            'Value':[f"{st.session_state.report['metrics']['R²']:0.4f}", 
                    f"{st.session_state.report['metrics']['RMSE']:0.4f}", 
                    f"{st.session_state.report['metrics']['AIC']:0.4f}"]
        })
        st.dataframe(evaluation_df, hide_index=True)
    
    # 진단 플롯
    fig = plt.figure(figsize=(12, 8))

    # 농도 프로파일
    plt.subplot(2,2,1)
    plt.plot(st.session_state.report['t_sim'], st.session_state.report['c_sim'], 'b--', label='Simulated')
    plt.scatter(st.session_state.report['t_obs'], st.session_state.report['c_obs'], c='red', label='Observed')
    plt.title('Concentration-Time Profile')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.legend()

    # 잔차 플롯
    plt.subplot(2,2,2)
    plt.scatter(st.session_state.report['t_obs'], st.session_state.report['residuals'], alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.title('Residuals vs Time')
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    
    # 예측 vs 관측
    plt.subplot(2,2,3)
    plt.scatter(st.session_state.report['c_obs'], st.session_state.report['c_pred'], alpha=0.6)
    lims = [0, max(st.session_state.report['c_obs'])*1.1]
    plt.plot(lims, lims, 'r--')
    plt.title('Predicted vs Observed')
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    
    # Q-Q 플롯 (Scipy 버전 호환성 처리)
    plt.subplot(2,2,4)
    res = st.session_state.report['residuals']
    try:
        (osm, osr), _ = probplot(res, dist="norm", fit=True)
    except:
        prob_res = probplot(res, dist="norm", fit=False)
        osm, osr = prob_res[0], prob_res[1]
    plt.scatter(osm, osr, alpha=0.6)
    plt.plot([-3, 3], [-3, 3], 'r--')
    plt.title('Normal Q-Q Plot')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    
    plt.tight_layout()
    st.pyplot(fig)