import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyomo.environ import *
from pyomo.opt import SolverFactory

st.set_page_config(
    page_title="총괄생산계획 최적화",
    page_icon="🏭",
    layout="wide"
)

# ─────────────────────────────────────────────
# Pyomo 최적화 함수
# ─────────────────────────────────────────────
def solve_app(demand, params, model_type="LP"):
    TH = len(demand)
    TIME = range(0, TH + 1)
    T = range(1, TH + 1)

    reg_wage   = params["reg_wage"]
    ot_wage    = params["ot_wage"]
    hire_cost  = params["hire_cost"]
    fire_cost  = params["fire_cost"]
    inv_cost   = params["inv_cost"]
    back_cost  = params["back_cost"]
    mat_cost   = params["mat_cost"]
    sub_cost   = params["sub_cost"]
    W0         = params["W0"]
    I0         = params["I0"]
    If_min     = params["If_min"]
    std_time   = params["std_time"]
    ot_max     = params["ot_max"]
    days       = params["days"]
    hours      = params["hours"]

    reg_prod = hours * days / std_time
    reg_labor_cost = reg_wage * hours * days

    m = ConcreteModel()
    type_var = NonNegativeIntegers if model_type == "IP" else NonNegativeReals

    m.W = Var(TIME, domain=type_var)
    m.H = Var(TIME, domain=type_var)
    m.L = Var(TIME, domain=type_var)
    m.P = Var(TIME, domain=NonNegativeReals)
    m.I = Var(TIME, domain=NonNegativeReals)
    m.S = Var(TIME, domain=NonNegativeReals)
    m.C = Var(TIME, domain=NonNegativeReals)
    m.O = Var(TIME, domain=NonNegativeReals)

    m.Cost = Objective(
        expr=sum(
            reg_labor_cost * m.W[t] +
            ot_wage        * m.O[t] +
            hire_cost      * m.H[t] +
            fire_cost      * m.L[t] +
            inv_cost       * m.I[t] +
            back_cost      * m.S[t] +
            mat_cost       * m.P[t] +
            sub_cost       * m.C[t]
            for t in T
        ),
        sense=minimize
    )

    m.labor     = Constraint(T, rule=lambda m, t: m.W[t] == m.W[t-1] + m.H[t] - m.L[t])
    m.capacity  = Constraint(T, rule=lambda m, t: m.P[t] <= reg_prod * m.W[t] + m.O[t] / std_time)
    m.inventory = Constraint(T, rule=lambda m, t:
        m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - demand[t-1] - m.S[t-1] + m.S[t])
    m.overtime  = Constraint(T, rule=lambda m, t: m.O[t] <= ot_max * m.W[t])
    m.W_0 = Constraint(rule=m.W[0] == W0)
    m.I_0 = Constraint(rule=m.I[0] == I0)
    m.S_0 = Constraint(rule=m.S[0] == 0)
    m.last_inv      = Constraint(rule=m.I[TH] >= If_min)
    m.last_shortage = Constraint(rule=m.S[TH] == 0)

    solver = SolverFactory("glpk")
    result = solver.solve(m, tee=False)

    if result.solver.termination_condition != TerminationCondition.optimal:
        return None

    monthly = []
    for t in T:
        monthly.append({
            "월": f"{t}월",
            "인원(명)": round(value(m.W[t]), 1),
            "고용(명)": round(value(m.H[t]), 1),
            "해고(명)": round(value(m.L[t]), 1),
            "생산(개)": round(value(m.P[t]), 1),
            "재고(개)": round(value(m.I[t]), 1),
            "부재고(개)": round(value(m.S[t]), 1),
            "하청(개)": round(value(m.C[t]), 1),
            "초과시간(h)": round(value(m.O[t]), 1),
            "수요(개)": demand[t-1],
        })

    cost_breakdown = {}
    for t in T:
        cost_breakdown[f"{t}월"] = {
            "정규노동비": reg_labor_cost * value(m.W[t]),
            "초과근무비": ot_wage * value(m.O[t]),
            "고용비":    hire_cost * value(m.H[t]),
            "해고비":    fire_cost * value(m.L[t]),
            "재고유지비": inv_cost * value(m.I[t]),
            "부재고비":  back_cost * value(m.S[t]),
            "재료비":    mat_cost * value(m.P[t]),
            "하청비":    sub_cost * value(m.C[t]),
        }

    return {
        "total_cost": value(m.Cost),
        "monthly": monthly,
        "cost_breakdown": cost_breakdown,
    }


# ─────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 총괄생산계획")
    st.markdown("**원예장비 제조업체** · Pyomo LP/IP")
    st.divider()

    st.markdown("### 📊 월별 수요 예측 (개)")
    scenario = st.selectbox("시나리오 프리셋", ["직접 입력", "기본 시나리오", "피크 수요", "변동 수요"])
    presets = {
        "기본 시나리오": [1600, 3000, 3200, 3800, 2200, 2200],
        "피크 수요":     [1600, 5000, 3200, 5800, 2200, 2200],
        "변동 수요":     [1600, 5000, 3200, 5800, 2200, 6500],
    }
    default_demands = presets.get(scenario, [1600, 3000, 3200, 3800, 2200, 2200])

    months = ["1월", "2월", "3월", "4월", "5월", "6월"]
    demand = []
    cols = st.columns(2)
    for i, mo in enumerate(months):
        with cols[i % 2]:
            demand.append(st.number_input(mo, value=default_demands[i], min_value=0, step=100, key=f"d{i}"))

    st.divider()
    st.markdown("### 💰 비용 파라미터")
    reg_wage  = st.number_input("정규임금 (원/시간)",      value=4000, step=500)
    ot_wage   = st.number_input("초과근무임금 (원/시간)",  value=6000, step=500)
    hire_cost = st.number_input("고용비용 (천원/인)",      value=300,  step=50) * 1000
    fire_cost = st.number_input("해고비용 (천원/인)",      value=500,  step=50) * 1000
    inv_cost  = st.number_input("재고유지비 (천원/개/월)", value=2,    step=1)  * 1000
    back_cost = st.number_input("부재고비용 (천원/개/월)", value=5,    step=1)  * 1000
    mat_cost  = st.number_input("재료비 (천원/개)",        value=10,   step=1)  * 1000
    sub_cost  = st.number_input("하청비용 (천원/개)",      value=30,   step=5)  * 1000

    st.divider()
    st.markdown("### ⚙️ 생산 파라미터")
    W0       = st.number_input("초기 인원 (명)",           value=80,   step=5)
    I0       = st.number_input("초기 재고 (개)",           value=1000, step=100)
    If_min   = st.number_input("최종 재고 최소 (개)",      value=500,  step=100)
    std_time = st.number_input("작업표준시간 (시간/개)",   value=4,    step=1)
    ot_max   = st.number_input("초과시간 한도 (시간/인/월)", value=10, step=1)
    days     = st.number_input("작업일수 (일/월)",         value=20,   step=1)
    hours    = st.number_input("작업시간 (시간/일)",       value=8,    step=1)

    st.divider()
    model_type   = st.radio("모델 유형", ["LP (실수해)", "IP (정수해)"], index=0)
    run_compare  = st.checkbox("LP vs IP 동시 비교", value=True)
    run_btn      = st.button("⚡ 최적화 실행", type="primary", use_container_width=True)


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
st.title("📊 총괄생산계획 최적화 대시보드")
st.caption("원예장비 제조업체 · Pyomo GLPK 솔버 · 6개월 계획")

if not run_btn:
    st.info("👈 왼쪽에서 파라미터를 설정하고 **최적화 실행** 버튼을 누르세요.")
    st.stop()

params = dict(
    reg_wage=reg_wage, ot_wage=ot_wage,
    hire_cost=hire_cost, fire_cost=fire_cost,
    inv_cost=inv_cost, back_cost=back_cost,
    mat_cost=mat_cost, sub_cost=sub_cost,
    W0=W0, I0=I0, If_min=If_min,
    std_time=std_time, ot_max=ot_max,
    days=days, hours=hours
)
mt = "LP" if model_type.startswith("LP") else "IP"

with st.spinner("Pyomo + GLPK로 최적화 계산 중..."):
    result = solve_app(demand, params, mt)
    result_lp = solve_app(demand, params, "LP")
    result_ip  = solve_app(demand, params, "IP") if run_compare else None

if result is None:
    st.error("최적해를 찾을 수 없습니다. 파라미터를 확인해주세요.")
    st.stop()

df  = pd.DataFrame(result["monthly"])
cb  = result["cost_breakdown"]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 계획 개요", "📋 월별 상세", "⚖️ LP vs IP 비교", "🔬 민감도 분석", "⚠️ 병목 분석"
])

# ══ TAB 1 ══
with tab1:
    total_shortage = df["부재고(개)"].sum()
    total_hire     = df["고용(명)"].sum()
    total_fire     = df["해고(명)"].sum()
    avg_inv        = df["재고(개)"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 최소 총 비용",  f"{result['total_cost']/1e6:.1f}백만원", f"{mt} 최적해")
    c2.metric("⚠️ 총 부재고",    f"{total_shortage:,.0f}개",
              "없음 ✅" if total_shortage == 0 else f"{int((df['부재고(개)']>0).sum())}개월 발생")
    c3.metric("👥 인력 변동",    f"+{total_hire:.0f} / -{total_fire:.0f}명", "고용 / 해고")
    c4.metric("📦 평균 재고",    f"{avg_inv:,.0f}개")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="생산", x=months, y=df["생산(개)"].tolist(), marker_color="#22c55e", opacity=0.8))
        fig.add_trace(go.Bar(name="하청", x=months, y=df["하청(개)"].tolist(), marker_color="#a855f7", opacity=0.8))
        fig.add_trace(go.Scatter(name="수요", x=months, y=demand,
                                 mode="lines+markers", line=dict(color="#6366f1", width=2.5, dash="dot")))
        fig.update_layout(title="수요 vs 생산량", barmode="stack", height=320,
                          plot_bgcolor="white", legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(name="인원(명)", x=months, y=df["인원(명)"].tolist(),
                              marker_color="#f59e0b", opacity=0.8), secondary_y=False)
        fig2.add_trace(go.Scatter(name="초과시간(h)", x=months, y=df["초과시간(h)"].tolist(),
                                  mode="lines+markers", line=dict(color="#8b5cf6", width=2)),
                       secondary_y=True)
        fig2.update_layout(title="인원 및 초과근무", height=320,
                           plot_bgcolor="white", legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="재고", x=months, y=df["재고(개)"].tolist(), marker_color="#0ea5e9", opacity=0.8))
        fig3.add_trace(go.Bar(name="부재고", x=months, y=df["부재고(개)"].tolist(), marker_color="#ef4444", opacity=0.8))
        fig3.update_layout(title="재고 / 부재고", barmode="group", height=300,
                           plot_bgcolor="white", legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        cost_names = ["정규노동비","초과근무비","고용비","해고비","재고유지비","부재고비","재료비","하청비"]
        cost_vals  = [sum(cb[f"{t+1}월"][k] for t in range(6)) for k in cost_names]
        fig4 = go.Figure(go.Pie(
            labels=cost_names, values=cost_vals, hole=0.5,
            marker_colors=["#6366f1","#8b5cf6","#22c55e","#ef4444","#0ea5e9","#f59e0b","#d97706","#dc2626"]
        ))
        fig4.update_layout(title="비용 구성", height=300, legend=dict(orientation="h", y=-0.3, font_size=11))
        st.plotly_chart(fig4, use_container_width=True)

# ══ TAB 2 ══
with tab2:
    st.markdown("### 📋 월별 결정변수 결과표")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 💸 월별 비용 분해")
    cost_keys = ["정규노동비","초과근무비","고용비","해고비","재고유지비","부재고비","재료비","하청비"]
    cost_data = {k: [cb[f"{t+1}월"][k] / 1000 for t in range(6)] for k in cost_keys}
    cost_df   = pd.DataFrame(cost_data, index=months)

    fig_cost = go.Figure()
    colors = ["#6366f1","#8b5cf6","#22c55e","#ef4444","#0ea5e9","#f59e0b","#d97706","#dc2626"]
    for i, col in enumerate(cost_df.columns):
        fig_cost.add_trace(go.Bar(name=col, x=months, y=cost_df[col].tolist(), marker_color=colors[i]))
    fig_cost.update_layout(barmode="stack", title="월별 비용 구성 (천원)",
                           height=380, plot_bgcolor="white",
                           legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_cost, use_container_width=True)

    monthly_totals = [sum(cb[f"{t+1}월"].values()) / 1000 for t in range(6)]
    cum_costs = []
    total = 0
    for v in monthly_totals:
        total += v
        cum_costs.append(total)

    fig_cum = make_subplots(specs=[[{"secondary_y": True}]])
    fig_cum.add_trace(go.Bar(name="월 비용(천원)", x=months, y=monthly_totals,
                             marker_color="#6366f1", opacity=0.7), secondary_y=False)
    fig_cum.add_trace(go.Scatter(name="누적 비용(천원)", x=months, y=cum_costs,
                                 mode="lines+markers", line=dict(color="#22c55e", width=2.5)),
                      secondary_y=True)
    fig_cum.update_layout(title="월별 비용 및 누적 비용", height=320,
                          plot_bgcolor="white", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_cum, use_container_width=True)

# ══ TAB 3 ══
with tab3:
    if result_lp is None or result_ip is None:
        st.info("사이드바에서 **'LP vs IP 동시 비교'** 체크 후 다시 실행하세요.")
    else:
        df_lp = pd.DataFrame(result_lp["monthly"])
        df_ip = pd.DataFrame(result_ip["monthly"])
        lp_cost = result_lp["total_cost"]
        ip_cost = result_ip["total_cost"]
        diff    = ip_cost - lp_cost

        c1, c2, c3 = st.columns(3)
        c1.metric("LP 총 비용 (실수해)", f"{lp_cost/1e6:.2f}백만원", "이론적 최솟값")
        c2.metric("IP 총 비용 (정수해)", f"{ip_cost/1e6:.2f}백만원", "현실적 구현 가능")
        c3.metric("비용 차이 (Integrality Gap)", f"{diff/1000:,.0f}천원", f"+{diff/lp_cost*100:.2f}%")

        st.info(f"💡 LP는 소수점 인원(예: 64.58명)을 허용해 **{lp_cost/1e6:.2f}백만원** 달성. "
                f"IP는 정수 인원으로 **{ip_cost/1e6:.2f}백만원** 필요. "
                f"차이 **{diff/1000:,.0f}천원**이 정수화 비용입니다.")

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(name="LP 인원", x=months, y=df_lp["인원(명)"].tolist(),
                                     mode="lines+markers", line=dict(color="#6366f1", width=2.5, dash="dot")))
            fig.add_trace(go.Scatter(name="IP 인원", x=months, y=df_ip["인원(명)"].tolist(),
                                     mode="lines+markers", line=dict(color="#22c55e", width=2.5)))
            fig.update_layout(title="LP vs IP — 인원 비교", height=320,
                               plot_bgcolor="white", legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(name="LP 재고", x=months, y=df_lp["재고(개)"].tolist(),
                                      mode="lines+markers", line=dict(color="#6366f1", width=2, dash="dot")))
            fig2.add_trace(go.Scatter(name="IP 재고", x=months, y=df_ip["재고(개)"].tolist(),
                                      mode="lines+markers", line=dict(color="#22c55e", width=2)))
            fig2.update_layout(title="LP vs IP — 재고 비교", height=320,
                                plot_bgcolor="white", legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 📊 LP vs IP 상세 비교표")
        compare_data = {
            "항목": ["총 비용(천원)", "평균 인원(명)", "총 고용(명)", "총 해고(명)",
                     "평균 재고(개)", "총 부재고(개)", "총 하청(개)"],
            "LP (실수해)": [
                f"{lp_cost/1000:,.0f}",
                f"{df_lp['인원(명)'].mean():.2f}",
                f"{df_lp['고용(명)'].sum():.2f}",
                f"{df_lp['해고(명)'].sum():.2f}",
                f"{df_lp['재고(개)'].mean():.0f}",
                f"{df_lp['부재고(개)'].sum():.0f}",
                f"{df_lp['하청(개)'].sum():.0f}",
            ],
            "IP (정수해)": [
                f"{ip_cost/1000:,.0f}",
                f"{df_ip['인원(명)'].mean():.1f}",
                f"{df_ip['고용(명)'].sum():.0f}",
                f"{df_ip['해고(명)'].sum():.0f}",
                f"{df_ip['재고(개)'].mean():.0f}",
                f"{df_ip['부재고(개)'].sum():.0f}",
                f"{df_ip['하청(개)'].sum():.0f}",
            ]
        }
        st.dataframe(pd.DataFrame(compare_data), use_container_width=True, hide_index=True)

# ══ TAB 4 ══
with tab4:
    st.markdown("### 🔬 비용 파라미터 민감도 분석")
    st.caption("슬라이더를 조정하면 비용 구성 변화를 시뮬레이션합니다 (재계산 없이 즉시 반영)")

    col1, col2 = st.columns([1, 2])
    with col1:
        fire_m = st.slider("해고비용 배율",    0.5, 3.0, 1.0, 0.1)
        inv_m  = st.slider("재고유지비 배율",  0.5, 4.0, 1.0, 0.1)
        back_m = st.slider("부재고비용 배율",  0.5, 4.0, 1.0, 0.1)
        ot_m   = st.slider("초과근무임금 배율",0.5, 3.0, 1.0, 0.1)
        hire_m = st.slider("고용비용 배율",    0.5, 3.0, 1.0, 0.1)

    with col2:
        orig = {k: sum(cb[f"{t+1}월"][k] for t in range(6)) for k in
                ["정규노동비","초과근무비","고용비","해고비","재고유지비","부재고비","재료비","하청비"]}
        mults = {"정규노동비":1,"초과근무비":ot_m,"고용비":hire_m,
                 "해고비":fire_m,"재고유지비":inv_m,"부재고비":back_m,"재료비":1,"하청비":1}
        new   = {k: orig[k] * mults[k] for k in orig}
        orig_total = sum(orig.values())
        new_total  = sum(new.values())
        diff_total = new_total - orig_total

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("기존 총 비용",    f"{orig_total/1e6:.2f}백만원")
        mc2.metric("조정 후 총 비용", f"{new_total/1e6:.2f}백만원",
                   f"{'+' if diff_total>0 else ''}{diff_total/1000:,.0f}천원")
        mc3.metric("변화율", f"{diff_total/orig_total*100:+.1f}%")

        fig_s = go.Figure()
        cats = list(orig.keys())
        clrs = ["#6366f1","#8b5cf6","#22c55e","#ef4444","#0ea5e9","#f59e0b","#d97706","#dc2626"]
        fig_s.add_trace(go.Bar(name="기존",    x=cats, y=[orig[k]/1000 for k in cats],
                                marker_color=[c+"55" for c in clrs]))
        fig_s.add_trace(go.Bar(name="조정 후", x=cats, y=[new[k]/1000 for k in cats],
                                marker_color=clrs))
        fig_s.update_layout(barmode="group", title="비용 구성 변화 (천원)", height=320,
                             plot_bgcolor="white", xaxis_tickangle=-30,
                             legend=dict(orientation="h", y=-0.35))
        st.plotly_chart(fig_s, use_container_width=True)

        if diff_total > 0:
            st.warning(f"파라미터 조정으로 비용이 {diff_total/1000:,.0f}천원 증가합니다.")
        else:
            st.success(f"파라미터 조정으로 비용이 {abs(diff_total)/1000:,.0f}천원 절감됩니다.")

# ══ TAB 5 ══
with tab5:
    st.markdown("### ⚠️ 병목 및 리스크 분석")
    reg_prod_pw = hours * days / std_time

    alerts = []
    for i, row in df.iterrows():
        mo   = months[i]
        mp   = reg_prod_pw * row["인원(명)"] + row["초과시간(h)"] / std_time
        util = row["생산(개)"] / mp * 100 if mp > 0 else 0
        if row["부재고(개)"] > 0:
            alerts.append(("danger", f"🚨 {mo}: 부재고 {row['부재고(개)']:,.0f}개 발생"))
        if row["초과시간(h)"] > ot_max * row["인원(명)"] * 0.85:
            alerts.append(("warning", f"⚠️ {mo}: 초과근무 한도 {row['초과시간(h)']/max(row['인원(명)'],1):.1f}h/인 — 한도({ot_max}h) 85% 초과"))
        if util > 90:
            alerts.append(("warning", f"⚠️ {mo}: 생산 가동률 {util:.0f}% — 과부하 위험"))
        if row["재고(개)"] < I0 * 0.2 and i < 5:
            alerts.append(("warning", f"⚠️ {mo}: 재고 {row['재고(개)']:,.0f}개 — 안전재고 낮음"))

    if not alerts:
        st.success("✅ 모든 제약조건을 충분히 만족하는 우수한 계획입니다!")
    else:
        for level, msg in alerts:
            if level == "danger":
                st.error(msg)
            else:
                st.warning(msg)

    st.markdown("---")
    util_data = []
    for i, row in df.iterrows():
        mp = reg_prod_pw * row["인원(명)"] + row["초과시간(h)"] / std_time
        util_data.append(round(row["생산(개)"] / mp * 100 if mp > 0 else 0, 1))

    col1, col2 = st.columns(2)
    with col1:
        clr_util = ["#ef4444" if u > 90 else "#f59e0b" if u > 75 else "#22c55e" for u in util_data]
        fig_u = go.Figure(go.Bar(
            x=months, y=util_data,
            marker_color=clr_util,
            text=[f"{u:.0f}%" for u in util_data],
            textposition="outside"
        ))
        fig_u.add_hline(y=90, line_dash="dash", line_color="#ef4444", annotation_text="위험(90%)")
        fig_u.add_hline(y=75, line_dash="dash", line_color="#f59e0b", annotation_text="주의(75%)")
        fig_u.update_layout(title="월별 생산 가동률 (%)", height=320,
                             plot_bgcolor="white", yaxis_range=[0, 115])
        st.plotly_chart(fig_u, use_container_width=True)

    with col2:
        st.markdown("**월별 종합 현황**")
        status_data = []
        for i, row in df.iterrows():
            mp   = reg_prod_pw * row["인원(명)"] + row["초과시간(h)"] / std_time
            util = round(row["생산(개)"] / mp * 100 if mp > 0 else 0, 1)
            has_danger = row["부재고(개)"] > 0
            has_warn   = util > 75
            status = "🔴 부재고" if has_danger else "🟡 주의" if has_warn else "🟢 정상"
            status_data.append({
                "월": months[i],
                "인원(명)": row["인원(명)"],
                "생산(개)": row["생산(개)"],
                "가동률(%)": util,
                "상태": status
            })
        st.dataframe(pd.DataFrame(status_data), use_container_width=True, hide_index=True)
