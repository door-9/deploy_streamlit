##############################################
# KPML_R_translate_css 파일에서 일부 수식을 수정한 버전
# 2개 그래프를 가로로 배치(subplots)하도록 수정
##############################################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

##############################################
# 1) 공통 계산 함수
##############################################

def calc_half_life(vd, cl):
    """
    R Shiny 식:
      t1/2 = ln(2) * vd / cl
    (cl: L/hr, vd: L)
    """
    try:
        if cl == 0:
            return 0.0
        return np.log(2) * vd / cl
    except:
        return 0.0

def create_two_subplots(x_cl, y_cl, b_cl, log_a_cl, 
                        x_vd, y_vd, b_vd, log_a_vd, 
                        x_human_cl, x_human_vd,
                        x_label_cl, y_label_cl
                        ):
    """
    하나의 Figure에 subplot(1행2열)을 만들고,
    왼쪽 그래프(CL), 오른쪽 그래프(Vd)를 각각 그림.
    x_cl, y_cl: 동물 CL 데이터 (log10 BW, log10 CL)
    b_cl, log_a_cl: CL 회귀 계수
    x_vd, y_vd: 동물 Vd 데이터 (log10 BW, log10 Vd)
    b_vd, log_a_vd: Vd 회귀 계수
    x_human_cl, x_human_vd: 사람 BW 값 log10 scale
      .
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- 왼쪽 subplot (CL)
    # 동물 데이터 산점도
    axes[0].scatter(x_cl, y_cl, label=f'Animal {y_label_cl} data points', color='blue')
    # 회귀선
    xs = np.linspace(min(x_cl)-0.2, x_human_cl+0.2, 50)
    ys = b_cl * xs + log_a_cl
    axes[0].plot(xs, ys, color='red', label=f"Reg line: y={b_cl:.2f}x+{log_a_cl:.2f}")

    # 사람 BW (log scale) → 예측값
    y_human_cl = b_cl * x_human_cl + log_a_cl
    axes[0].scatter([x_human_cl], [y_human_cl], color='green', marker='x', s=80, label=f'Human {y_label_cl} predicted')

    axes[0].set_xlabel(f'{x_label_cl}')
    axes[0].set_ylabel(f'{y_label_cl}')
    axes[0].set_title('CL regression')
    axes[0].legend()

    # --- 오른쪽 subplot (Vd)
    # 동물 데이터 산점도
    axes[1].scatter(x_vd, y_vd, label='Animal Vd data points', color='purple')
    # 회귀선
    xs_vd = np.linspace(min(x_vd)-0.2, x_human_vd+0.2, 50)
    ys_vd = b_vd * xs_vd + log_a_vd
    axes[1].plot(xs_vd, ys_vd, color='red', label=f"Reg line: y={b_vd:.2f}x+{log_a_vd:.2f}")

    # 사람 BW (log scale) → 예측값
    y_human_vd = b_vd * x_human_vd + log_a_vd
    axes[1].scatter([x_human_vd], [y_human_vd], color='green', marker='x', s=80, label='Human Vd predicted')

    axes[1].set_xlabel('Log10(BW(kg))')
    axes[1].set_ylabel('Log10(Vd(L))')
    axes[1].set_title('Vd regression')
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)  # 메모리 해제

def predict_species1(species, BW_human, BW_animal, cl_animal, vd_animal):  
    """
    Single species scaling:
      Rat:    CL_h= 0.152 × CL_rat
      Dog:    CL_h= 0.41  × CL_dog
      Monkey: CL_h= 0.407 × CL_monkey
      Other:  CL_h= CL_animal * ( (BW_human/BW_animal)^0.75 )
    Vd_h= vd_animal * ( (BW_human/BW_animal)^1 )
    t1/2= calc_half_life(...)
    """
    if species.lower()=="rat":
        cl_h = (cl_animal/BW_animal)*0.152*BW_human
    elif species.lower()=="dog":
        cl_h = (cl_animal/BW_animal)*0.41*BW_human
    elif species.lower()=="monkey":
        cl_h = (cl_animal/BW_animal)*0.407*BW_human
    else:
        cl_h = cl_animal * ((BW_human/BW_animal)**0.75)

    vd_h= vd_animal*((BW_human/BW_animal)**1)

    t12= calc_half_life(vd_h, cl_h)
    return cl_h, vd_h, t12

def calc_sa(animal_df, human_df):
    """
    Simple Allometry (SA):
    - 동물: log10(CL_animal) ~ log10(BW_animal)
    - 사람: CL_h= a*(BW_h^b)
            Vd_h= a_vd*(BW_h^b_vd)
    """
    dfA = animal_df.dropna(subset=["BW(kg)","CL(L/hr)","Vd(L)"])
    dfA = dfA[(dfA["BW(kg)"]>0)&(dfA["CL(L/hr)"]>0)&(dfA["Vd(L)"]>0)]
    if len(dfA)<2:
        return None

    # CL part
    x_cl= np.log10(dfA["BW(kg)"].values)
    y_cl= np.log10(dfA["CL(L/hr)"].values)
    b_cl, log_a_cl= np.polyfit(x_cl, y_cl, 1)
    a_cl= 10**log_a_cl

    # Vd part
    x_vd= np.log10(dfA["BW(kg)"].values)
    y_vd= np.log10(dfA["Vd(L)"].values)
    b_vd, log_a_vd= np.polyfit(x_vd, y_vd, 1)
    a_vd= 10**log_a_vd

    # 사람 DF(1줄)에서 BW_h
    dfH = human_df.dropna(subset=["BW(kg)"])
    rowH= dfH.iloc[0]
    BW_h= float(rowH["BW(kg)"]) if rowH["BW(kg)"]>0 else 70.0

    CL_h= (a_cl*(BW_h**b_cl))
    Vd_h= (a_vd*(BW_h**b_vd))
    t12= calc_half_life(Vd_h,CL_h)

    # side-by-side subplot
    create_two_subplots(
        x_cl, y_cl, b_cl, log_a_cl,
        x_vd, y_vd, b_vd, log_a_vd,
        np.log10(BW_h), np.log10(BW_h),  # 사람 BW는 동일
        "BW(kg)","CL"
    )

    return (b_cl,b_vd,CL_h,Vd_h,t12)

def calc_mlp(animal_df, human_df):
    """
    ROE(MLP):
    - 동물: log10(CL_i * MLP_i) ~ log10(BW_i)
    - 사람: CL_h= (a_roe*(BW_h^b_roe)) / MLP_h
    - Vd => SA
    """
    dfA= animal_df.dropna(subset=["BW(kg)","CL(L/hr)","Vd(L)","MLP(year)"])
    dfA= dfA[(dfA["BW(kg)"]>0)&(dfA["CL(L/hr)"]>0)&(dfA["Vd(L)"]>0)&(dfA["MLP(year)"]>0)]
    if len(dfA)<2:
        return None

    # CL part
    x_cl= np.log10(dfA["BW(kg)"].values)
    cl_times_mlp= dfA["CL(L/hr)"].values * dfA["MLP(year)"].values * 365 * 24
    y_cl= np.log10(cl_times_mlp)
    b_roe, log_a_roe= np.polyfit(x_cl, y_cl, 1)
    a_roe= 10**log_a_roe

    # Vd => SA
    x_vd= np.log10(dfA["BW(kg)"].values)
    y_vd= np.log10(dfA["Vd(L)"].values)
    b_vd, log_a_vd= np.polyfit(x_vd, y_vd, 1)
    a_vd= 10**log_a_vd

    # Human row
    dfH= human_df.dropna(subset=["BW(kg)","MLP(year)"])
    rowH= dfH.iloc[0]
    BW_h= float(rowH["BW(kg)"]) if rowH["BW(kg)"]>0 else 70.0
    MLP_h= float(rowH["MLP(year)"]) if rowH["MLP(year)"]>0 else 93.0
    MLP_h_hours = MLP_h * 365 * 24

    CL_h= (a_roe*(BW_h**b_roe))/MLP_h_hours
    Vd_h= (a_vd*(BW_h**b_vd))
    t12= calc_half_life(Vd_h,CL_h)

    # side-by-side subplot
    create_two_subplots(
        x_cl, y_cl, b_roe, log_a_roe,
        x_vd, y_vd, b_vd, log_a_vd,
        np.log10(BW_h), np.log10(BW_h),
        "BW(kg)","MLP*CL"
    )

    return (b_roe, b_vd, CL_h, Vd_h, t12)

def calc_brw(animal_df, human_df):
    """
    ROE(BrW):
    - 동물: log10(CL_i * BrW_i) ~ log10(BW_i)
    - 사람: CL_h= (a_brw*(BW_h^b_brw)) / BrW_h
    - Vd => SA
    """
    dfA= animal_df.dropna(subset=["BW(kg)","CL(L/hr)","Vd(L)","BrW(kg)"])
    dfA= dfA[(dfA["BW(kg)"]>0)&(dfA["CL(L/hr)"]>0)&(dfA["Vd(L)"]>0)&(dfA["BrW(kg)"]>0)]
    if len(dfA)<2:
        return None

    x_cl= np.log10(dfA["BW(kg)"].values)
    cl_times_brw= dfA["CL(L/hr)"].values * dfA["BrW(kg)"].values
    y_cl= np.log10(cl_times_brw)
    b_brw, log_a_brw= np.polyfit(x_cl, y_cl, 1)
    a_brw= 10**log_a_brw

    # Vd => SA
    x_vd= np.log10(dfA["BW(kg)"].values)
    y_vd= np.log10(dfA["Vd(L)"].values)
    b_vd, log_a_vd= np.polyfit(x_vd, y_vd, 1)
    a_vd= 10**log_a_vd

    dfH= human_df.dropna(subset=["BW(kg)","BrW(kg)"])
    rowH= dfH.iloc[0]
    BW_h= float(rowH["BW(kg)"]) if rowH["BW(kg)"]>0 else 70.0
    BrW_h= float(rowH["BrW(kg)"]) if rowH["BrW(kg)"]>0 else 1.53

    CL_h= (a_brw*(BW_h**b_brw)) / BrW_h
    Vd_h= (a_vd*(BW_h**b_vd))
    t12= calc_half_life(Vd_h, CL_h)

    # side-by-side subplot
    create_two_subplots(
        x_cl, y_cl, b_brw, log_a_brw,
        x_vd, y_vd, b_vd, log_a_vd,
        np.log10(BW_h), np.log10(BW_h),
        "BW(kg)","BrW*CL"
    )

    return (b_brw, b_vd, CL_h, Vd_h, t12)


##############################################
# 2) 메인
##############################################
def main():
    st.set_page_config(page_title="Allometric Scaling: Single species + (SA/MLP/BrW)", layout="wide")
    st.title("Allometric Scaling - Single species / SA / MLP / BrW")
    st.write("""
    - **Single species** 모드: Animal table 1행(Rat/Dog/Monkey/Other) + Human table (1행)  
    - **SA/MLP/BrW** 모드: Animal table에 2행 이상(각 동물) + Human table (1행)  
    """)

    #-----------------------------------------
    # A) Animal table
    #-----------------------------------------
    st.subheader("1) Animal Data Table")
    animal_init = pd.DataFrame(columns=[
        "Species",            
        "BW(kg)",            
        "CL(L/hr)",
        "Vd(L)",
        "MLP(year)",         
        "BrW(kg)"           
    ])
    animal_df = st.data_editor(
        animal_init,
        num_rows="dynamic",
        use_container_width=True,
        key="animal_editor"
    )

    #-----------------------------------------
    # B) Human table
    #-----------------------------------------
    st.subheader("2) Human Data Table (1 row)")
    human_init = pd.DataFrame({
        "BW(kg)": [70.0],
        "MLP(year)": [93.0],
        "BrW(kg)": [1.53]
    })
    human_df = st.data_editor(
        human_init,
        num_rows=1,
        use_container_width=True,
        key="human_editor"
    )

    #-----------------------------------------
    # C) Method selection
    #-----------------------------------------
    st.subheader("3) Choose Method and Calculate")
    method = st.selectbox(
        "Select Method",
        ["Single species", "Simple Allometry (SA)", "ROE (MLP)", "ROE (BrW)"],
        key="method_selection",
    )
   
    # 안내 메시지
    if method == "Single species":
        st.write("Animal data required: **Species, BW, CL, Vd** (1행)")
        st.write("Human data required: **BW** (1행)")
    elif method == "Simple Allometry (SA)":
        st.write("Animal data required: **Species, BW, CL, Vd** (>=2행)")
        st.write("Human data required: **BW** (1행)")
    elif method == "ROE (MLP)":
        st.write("Animal data required: **Species, BW, CL, Vd, MLP** (>=2행)")
        st.write("Human data required: **BW, MLP** (1행)")
    elif method == "ROE (BrW)":
        st.write("Animal data required: **Species, BW, CL, Vd, BrW** (>=2행)")
        st.write("Human data required: **BW, BrW** (1행)")

    if st.button("Calculate"):
        dfA = animal_df.copy()
        for c in ["BW(kg)","CL(L/hr)","Vd(L)","MLP(year)","BrW(kg)"]:
            if c in dfA.columns:
                dfA[c] = pd.to_numeric(dfA[c], errors='coerce')

        dfH = human_df.copy()
        for c in ["BW(kg)","MLP(year)","BrW(kg)"]:
            if c in dfH.columns:
                dfH[c] = pd.to_numeric(dfH[c], errors='coerce')

        if method=="Single species":
            if len(dfA)<1:
                st.error("Need at least 1 row in Animal table!")
                return
            rowA= dfA.iloc[0]
            if not isinstance(rowA["Species"], str):
                st.error("Animal row must have a valid 'Species'!")
                return
            sp_name= rowA["Species"]
            BW_an = float(rowA["BW(kg)"]) if rowA["BW(kg)"]>0 else 0.0
            cl_an= float(rowA["CL(L/hr)"]) if rowA["CL(L/hr)"]>0 else 0.0
            vd_an= float(rowA["Vd(L)"]) if rowA["Vd(L)"]>0 else 0.0

            if BW_an<=0 or cl_an<=0 or vd_an<=0:
                st.error("Animal BW, CL, Vd must be >0 for single species method.")
                return

            if len(dfH)<1:
                st.error("Need at least 1 row in Human table!")
                return
            rowH= dfH.iloc[0]
            BW_h = float(rowH["BW(kg)"]) if rowH["BW(kg)"]>0 else 0.0
            
            if BW_h<=0:
                st.error("Human BW must be >0 for single species method.")
                return

            c_h, v_h, t12= predict_species1(sp_name, BW_h, BW_an, cl_an, vd_an)
            results = pd.DataFrame({
                "Species": [sp_name],
                "CL_h": [f"{c_h:.3f}"],
                "Vd_h": [f"{v_h:.3f}"],
                "t1/2": [f"{t12:.3f}"],
            })
            st.subheader("Calculation Results")
            st.table(results)

        elif method=="Simple Allometry (SA)":
            res= calc_sa(dfA, dfH)
            if res is None:
                st.error("calc_sa returned None (need >=2 valid Animal lines).")
            else:
                b_cl,b_vd,CL_h,Vd_h,t12= res
                results = pd.DataFrame({
                    "CL_h": [f"{CL_h:.3f}"],
                    "Vd_h": [f"{Vd_h:.3f}"],
                    "t1/2": [f"{t12:.3f}"],
                })
                st.subheader("Calculation Results [SA]")
                st.table(results)

        elif method=="ROE (MLP)":
            res= calc_mlp(dfA, dfH)
            if res is None:
                st.error("calc_mlp returned None (need >=2 Animal lines + MLP?).")
            else:
                b_roe,b_vd,CL_h,Vd_h,t12= res
                results = pd.DataFrame({
                    "CL_h": [f"{CL_h:.3f}"],
                    "Vd_h": [f"{Vd_h:.3f}"],
                    "t1/2": [f"{t12:.3f}"],
                })
                st.subheader("Calculation Results [MLP]")
                st.table(results)

        else:  # BrW
            res= calc_brw(dfA, dfH)
            if res is None:
                st.error("calc_brw returned None (need >=2 Animal lines + BrW?).")
            else:
                b_brw,b_vd,CL_h,Vd_h,t12= res
                results = pd.DataFrame({
                    "CL_h": [f"{CL_h:.3f}"],
                    "Vd_h": [f"{Vd_h:.3f}"],
                    "t1/2": [f"{t12:.3f}"],
                })
                st.subheader("Calculation Results [BrW]")
                st.table(results)


########################################
if __name__=="__main__":
    main()
