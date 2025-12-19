import os
import streamlit as st
import pandas as pd
import math
import json
import google.generativeai as genai
from PIL import Image

# ================= 配置区 =================
# 1. 代理设置 (请确认您的端口号)
os.environ["HTTP_PROXY"] = "http://127.0.0.1:8890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8890"

# 2. 公司品牌配置
COMPANY_NAME = "GenAI Health Tech"
LOGO_URL = "https://img.icons8.com/color/96/caduceus.png" 

# ================= 0. 页面与样式配置 =================
st.set_page_config(
    page_title=f"{COMPANY_NAME} - CKD Agent",
    page_icon="🧬",
    layout="wide"
)

# 注入 CSS 美化字体和界面 (罗氏蓝风格)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', 'SF Pro SC', 'Microsoft YaHei', sans-serif;
        }
        h1, h2, h3 { color: #0E4D92; font-weight: 600; }
        [data-testid="stSidebar"] { background-color: #F4F8FB; }
        div.stButton > button:first-child {
            background-color: #0E4D92; color: white; border-radius: 8px; border: none; padding: 10px 24px; font-size: 16px;
        }
        div.stButton > button:hover { background-color: #083060; color: white; }
        [data-testid="stMetricValue"] { font-size: 24px; color: #0E4D92; }
        .expert-card { background-color:#FFF9C4; padding:20px; border-radius:12px; border-left: 8px solid #FBC02D; margin-bottom:25px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

# ================= 1. 核心算法区 (KFRE Model 3) =================
def calculate_kfre_precise(age, sex, egfr, acr):
    """同时返回 2年 和 5年 风险 (Tangri et al. 2011)"""
    try:
        age = float(age)
        egfr = float(egfr)
        acr = float(acr)
        is_male = 1.0 if str(sex).lower() in ['male', '男', 'm', '1'] else 0.0
        acr_val = acr if acr > 0 else 1.0
        
        log_acr = math.log(acr_val)
        age_scaled = age / 10.0
        egfr_scaled = egfr / 5.0

        # 系数
        beta_age = -0.2167; mean_age = 7.0355
        beta_sex = 0.2694; mean_sex = 0.56422
        beta_egfr = -0.55418; mean_egfr = 7.2216
        beta_acr = 0.45608; mean_acr = 5.2774

        lp = (beta_age * (age_scaled - mean_age)) + \
             (beta_sex * (is_male - mean_sex)) + \
             (beta_egfr * (egfr_scaled - mean_egfr)) + \
             (beta_acr * (log_acr - mean_acr))

        S0_5yr = 0.9240; risk_5yr = 1.0 - math.pow(S0_5yr, math.exp(lp))
        S0_2yr = 0.9832; risk_2yr = 1.0 - math.pow(S0_2yr, math.exp(lp))

        return {"2yr": round(risk_2yr * 100, 2), "5yr": round(risk_5yr * 100, 2)}
    except Exception as e:
        return {"error": str(e)}

# ================= 2. 辅助计算工具 (CKD-EPI 2021) =================
def calculate_egfr_ckdepi(scr_umol, age, sex_str):
    """根据肌酐(umol/L)计算 eGFR"""
    try:
        scr_mgdl = float(scr_umol) / 88.4
        age_val = float(age)
        is_male = str(sex_str).lower() in ['male', '男', 'm', '1']
        
        kappa = 0.9 if is_male else 0.7
        alpha = -0.302 if is_male else -0.241
        
        factor1 = min(scr_mgdl / kappa, 1) ** alpha
        factor2 = max(scr_mgdl / kappa, 1) ** -1.209
        factor3 = 0.993 ** age_val
        factor4 = 1.018 if not is_male else 1.0
        
        egfr = 142 * factor1 * factor2 * factor3 * factor4
        return round(egfr, 1)
    except:
        return None

# ================= 2.1 新增：单位标准化工具函数 =================
def standardize_uacr(value, unit):
    if value is None: return None
    try:
        val = float(value)
        u = str(unit).lower().strip()
        if u in ["mg/g", "ug/mg", "μg/mg"]: return val
        elif any(x in u for x in ["mg/mmol", "g/mol", "mg/mm"]): return round(val * 8.84, 2)
        elif "g/g" in u: return val * 1000
        else: return val
    except: return None

# ================= 3. 智能提取助手 (支持多图 & 单位换算) =================
def extract_data_with_gemini(user_input, image_list=None):
    extraction_prompt = """
    你是一个专业的医疗数据录入员。请综合阅读【所有上传的图片】和文本，将分散在不同图片上的信息拼凑成一个完整的患者档案。
    【数值与单位必须分离】
    在提取 JSON 时，"value" 字段只能包含纯数字（支持小数点），严禁包含文字符号。"unit" 字段单独存放单位。
    
    【🔍 视觉扫描策略】
    1. **基本信息**：寻找 年龄 (Age) 和 性别 (Sex)。
    2. **生化指标**：寻找 血肌酐 (Creatinine) 和 eGFR。
    3. **血压和血糖**：
       - 血压 (BP): 寻找如 "135/85", "BP: 120/70" 等。分离为 sbp 和 dbp。
       - 血糖: 寻找 糖化血红蛋白 (HbA1c) 或 空腹血糖 (Glucose/Glu)。
    4. **uACR**: 寻找 "uACR"、"尿微量白蛋白/肌酐比值"。
    5. **尿液组分**: 尿微量白蛋白 (u_albumin_raw), 尿肌酐 (u_creatinine_raw)。

    【返回 JSON 结构】
    {
        "age": 数字, "sex": "Str", "egfr_stated": 数字,
        "blood_pressure": { "sbp": 数字, "dbp": 数字 }, 
        "hba1c": { "value": 数字, "unit": "%" },
        "glucose": { "value": 数字, "unit": "Str" },
        "creatinine_raw": { "value": 数字, "unit": "Str" },
        "uacr_raw": { "value": 数字, "unit": "Str" },
        "u_albumin_raw": { "value": 数字, "unit": "Str" },
        "u_creatinine_raw": { "value": 数字, "unit": "Str" },
        "report_summary": "简述提取情况，若血压/血糖缺失请注明。"
    } 
    请直接返回 JSON 字符串，不要包含 Markdown 格式。
    """
    try:
        inputs = [extraction_prompt]
        if user_input: inputs.append(f"用户补充描述: {user_input}")
        if image_list:
            for i, img in enumerate(image_list):
                inputs.append(f"【图片 {i+1}】")
                inputs.append(img)
        
        response = model.generate_content(inputs)
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        st.error(f"AI 提取失败: {e}")
        return None

# ================= 4. 主界面逻辑 =================
# 不要直接写 KEY，改为从 Streamlit 的“秘密管理”中读取
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    # 如果本地运行没有配置 secrets，可以留个后手手动输入
    API_KEY = st.sidebar.text_input("🔑 请输入 API Key", type="password")

if not API_KEY:
    st.warning("👈 请在左侧侧边栏输入 API Key，或者在 Streamlit 后台配置 Secrets")
    st.stop()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-3-pro-preview') 

st.title("🧬 智能慢性肾病早筛系统")
st.caption(f"Benchmark: Roche KlinRisk | Powered by {COMPANY_NAME}")

tab1, tab2 = st.tabs(["📂 数据库选择", "✨ 智能录入 (多图识别)"])
current_patient = None

# --- Tab 1: 数据库模式 ---
with tab1:
    try:
        df = pd.read_csv("cleaned_kidney_data.csv")
        patient_list = df['id'].tolist()
        selected_id = st.selectbox("选择标准病例 ID", patient_list)
        
        raw_patient = df[df['id'] == selected_id].iloc[0]
        
        current_patient = {
            "age": raw_patient['age'], "sex": raw_patient['sex'],
            "egfr": raw_patient['egfr'], "uacr": raw_patient['uacr'],
            "source": f"Database ID: {selected_id}",
            "htn": raw_patient.get('htn', 'Unknown'), 
            "dm": raw_patient.get('dm', 'Unknown'), 
            "bp": {"sbp": raw_patient.get('sbp'), "dbp": raw_patient.get('dbp')} if 'sbp' in raw_patient else None,
            "glucose": None, 
            "hba1c": None
        }
    except Exception as e:
        pass

# --- Tab 2: 智能录入模式 (支持多图) ---
with tab2:
    col_input, col_preview = st.columns([2, 1])
    with col_input:
        # 【修复点 1】 修正了这里缺失的开头双引号
        user_text = st.text_area("✍️ 备注/病历描述", height=80)
        uploaded_files = st.file_uploader("📷 拖入多张化验单 (支持同时上传)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
        extract_btn = st.button("🔍 AI 综合提取", type="primary")

    if extract_btn:
        with st.spinner("Gemini 正在阅读所有化验单..."):
            image_contents = []
            if uploaded_files:
                for f in uploaded_files:
                    image_contents.append(Image.open(f))
                with col_preview:
                    st.image(image_contents[0], caption=f"共上传 {len(image_contents)} 张", use_column_width=True)
            st.session_state['raw_data_cache'] = extract_data_with_gemini(user_text, image_contents)

    if st.session_state.get('raw_data_cache'):
        raw_data = st.session_state['raw_data_cache']
        
        temp_patient = {
            "age": raw_data.get("age"),
            "sex": raw_data.get("sex"),
            "egfr": raw_data.get("egfr_stated"),
            "uacr": None,
            "bp": raw_data.get("blood_pressure"),  
            "glucose": raw_data.get("glucose"),    
            "hba1c": raw_data.get("hba1c"),        
            "source": "AI Extraction"
        }
        
        # uACR 计算策略
calc_success = False
        
        # 【核心修复】增加 or {}，防止 AI 返回 null 导致程序崩溃
        u_alb = raw_data.get("u_albumin_raw") or {}
        u_cre = raw_data.get("u_creatinine_raw") or {}
        
        # 1. 优先逻辑：利用原始白蛋白 + 原始肌酐进行精准重算
        # 只有当两个字典都不为空，且都有 "value" 时才计算
        if isinstance(u_alb, dict) and isinstance(u_cre, dict) and u_alb.get("value") and u_cre.get("value"):
            try:
                alb_val = float(u_alb["value"])
                cre_val = float(u_cre["value"])
                # 预处理单位字符串：去除空格、转小写、处理特殊字符
                cre_unit = str(u_cre.get("unit", "")).lower().replace(" ", "")
                
                # 识别 umol/L 并强制转换为 mmol/L
                if any(x in cre_unit for x in ["umol", "μmol", "um/l", "μm/l"]):
                    cre_val = cre_val / 1000.0
                
                # 计算比值 (mg/mmol)
                if cre_val > 0: # 防止除以0
                    ratio_mg_mmol = alb_val / cre_val
                    # 转换为标准单位 mg/g
                    temp_patient["uacr"] = round(ratio_mg_mmol * 8.84, 2)
                    calc_success = True
                    
                    st.success(f"✅ 原始值重算成功：uACR 为 **{temp_patient['uacr']} mg/g**")
                    
                    if temp_patient["uacr"] > 300:
                        st.error("🚨 警告：该患者处于 A3 期 (重度增加)，属于极高危状态！")
                    elif temp_patient["uacr"] >= 30:
                        st.warning("⚠️ 提示：该患者处于 A2 期 (中度增加)。")
            except Exception as e:
                # 计算出错也不要崩，直接pass
                print(f"uACR calc error: {e}")
                pass

        # 2. 备选逻辑：如果无法重算，则使用 AI 直接提取的 uACR 值
        if not calc_success:
            u_raw = raw_data.get("uacr_raw") or {} # 同样加保险
            if isinstance(u_raw, dict) and u_raw.get("value"):
                std_val = standardize_uacr(u_raw["value"], u_raw.get("unit"))
                if std_val:
                    temp_patient["uacr"] = std_val
                    if abs(std_val - float(u_raw["value"])) > 0.1:
                        st.warning(f"🔄 采用提取值并完成换算：{u_raw['value']} {u_raw.get('unit')} → {std_val} mg/g")
                    else:
                        st.info(f"✅ 提取 uACR 成功: {std_val} mg/g")

        # 3. 冲突核验：如果 AI 提取的汇总值与我们重算的值差异过大，发出预警
        if calc_success and raw_data.get("uacr_raw", {}).get("value"):
            extracted_val = standardize_uacr(raw_data["uacr_raw"]["value"], raw_data["uacr_raw"].get("unit"))
            if extracted_val and abs(temp_patient["uacr"] - extracted_val) / extracted_val > 0.2:
                st.warning(f"⚖️ 数据冲突提醒：重算值 ({temp_patient['uacr']}) 与报告汇总值 ({extracted_val}) 差异较大，请手动核对原始图片。")

        # eGFR 自动补算
        cr = raw_data.get("creatinine_raw", {})
        if temp_patient["egfr"] is None and cr and cr.get("value") and temp_patient["age"] and temp_patient["sex"]:
            temp_patient["egfr"] = calculate_egfr_ckdepi(cr["value"], temp_patient["age"], temp_patient["sex"])
            if temp_patient["egfr"]: st.info(f"💡 eGFR 自动补算: {temp_patient['egfr']}")

        st.write(f"**识别摘要**: {raw_data.get('report_summary')}")

        # 缺项检查
        missing = []
        if not temp_patient["age"]: missing.append("年龄")
        if not temp_patient["egfr"]: missing.append("eGFR")
        if missing: st.warning(f"⚠️ 信息不全: {' / '.join(missing)}")

        # 【修复点 2】 完整的表单防崩溃逻辑，保留了所有变量名
        with st.form("supplement_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                # 修复：防止 temp_patient["age"] 为 None 导致崩溃
                val_age = temp_patient.get("age")
                safe_age = int(val_age) if val_age is not None else 50
                new_age = st.number_input("补全年龄", value=safe_age)
                
                # 修复：防止性别索引错误
                val_sex = temp_patient.get("sex")
                idx_sex = 0 if str(val_sex).lower() in ['male', '男'] else 1
                new_sex = st.selectbox("补全性别", ["Male", "Female"], index=idx_sex)

            with c2:
                # 修复：eGFR 防空
                val_egfr = temp_patient.get("egfr")
                safe_egfr = float(val_egfr) if val_egfr is not None else 90.0
                new_egfr = st.number_input("补全 eGFR", value=safe_egfr)
                
                # 修复：uACR 防空
                val_uacr = temp_patient.get("uacr")
                safe_uacr = float(val_uacr) if val_uacr is not None else 30.0
                new_uacr = st.number_input("补全 uACR (mg/g)", value=safe_uacr)

            with c3:
                # 修复：血压防空 (最关键的修复)
                bp_dict = temp_patient.get('bp') or {}
                raw_sbp = bp_dict.get('sbp')
                safe_sbp = int(raw_sbp) if raw_sbp is not None else 0
                new_sbp = st.number_input("收缩压 (mmHg)", value=safe_sbp)
                
                raw_dbp = bp_dict.get('dbp')
                safe_dbp = int(raw_dbp) if raw_dbp is not None else 0
                new_dbp = st.number_input("舒张压 (mmHg)", value=safe_dbp)

                hba1c_dict = temp_patient.get('hba1c') or {}
                raw_a1c = hba1c_dict.get('value')
                safe_a1c = float(raw_a1c) if raw_a1c is not None else 0.0
                new_hba1c = st.number_input("HbA1c (%)", value=safe_a1c)

            if st.form_submit_button("✅ 提交并分析"):
                bp_data = {"sbp": new_sbp, "dbp": new_dbp} if new_sbp > 0 else None
                hba1c_data = {"value": new_hba1c, "unit": "%"} if new_hba1c > 0 else None
                st.session_state['confirmed_patient'] = {
                    "age": new_age, "sex": new_sex, "egfr": new_egfr, "uacr": new_uacr,
                    "bp": bp_data, "hba1c": hba1c_data, 
                    "glucose": temp_patient.get("glucose"),
                    "source": "AI + Manual Edit"
                }
                st.rerun()

    if st.session_state.get('confirmed_patient'):
        current_patient = st.session_state['confirmed_patient']

# ================= 5. 分析与报告生成 =================
if current_patient:
    st.markdown("---")
    st.subheader("👤 患者核心指标基线")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("年龄 / 性别", f"{int(current_patient['age'])} / {current_patient['sex']}")
    with c2: st.metric("eGFR", f"{current_patient['egfr']} ml/min")
    with c3: st.metric("uACR", f"{current_patient['uacr']} mg/g")
    
    bp_dict = current_patient.get('bp') or {}
    bp_str = f"{bp_dict.get('sbp', '?')}/{bp_dict.get('dbp', '?')}" if bp_dict.get('sbp') else "未见数据"
    with c4: st.metric("当前血压", bp_str)

    glu_str_list = []
    if current_patient.get('glucose'): glu_str_list.append(f"Glu: {current_patient['glucose']['value']}")
    if current_patient.get('hba1c'): glu_str_list.append(f"A1c: {current_patient['hba1c']['value']}%")
    with c5: st.metric("代谢指标", " / ".join(glu_str_list) if glu_str_list else "未见数据")

    with st.spinner("正在进行 KFRE 风险演算..."):
        risks = calculate_kfre_precise(current_patient['age'], current_patient['sex'], current_patient['egfr'], current_patient['uacr'])
    
    if "error" not in risks:
        risk_5yr = risks['5yr']
        col_dash, col_rpt = st.columns([1, 2.5])
        
        with col_dash:
            st.markdown("### 📉 5年肾衰风险")
            risk_color = "#2E7D32" if risk_5yr < 3 else "#F9A825" if risk_5yr < 5 else "#D32F2F"
            st.markdown(f"<h1 style='color:{risk_color};font-size:72px;margin:0;'>{risk_5yr}%</h1>", unsafe_allow_html=True)
            st.info(f"🚨 **2年近期风险**: {risks['2yr']}%")
            
            chart_data = pd.DataFrame({
                "Year": ["Y0","Y1","Y2","Y3","Y4","Y5"], 
                "Risk": [0, risk_5yr*0.2, risks['2yr'], risk_5yr*0.6, risk_5yr*0.8, risk_5yr]
            })
            st.area_chart(chart_data, x="Year", y="Risk", color=risk_color)

        with col_rpt:
            st.markdown("### 📋 AI 临床决策支持报告")
            
            # 读取知识库
            try:
                with open("kdigo_guidelines_2024.txt", "r", encoding="utf-8") as f: kb_kdigo = f.read()
                with open("中国慢性肾脏病早期评价与管理指南 (2023).txt", "r", encoding="utf-8") as f: kb_china = f.read()
                with open("Comprehensive Clinical Nephrology .txt", "r", encoding="utf-8") as f: kb_ccn = f.read()
                kb_all = f"{kb_kdigo}\n\n{kb_china}\n\n{kb_ccn}"
            except: 
                kb_all = "知识库文件缺失，请检查路径。"
            
            # 【关键修复】构建详细结构的 Prompt，强制模型输出对象而非字符串
            expert_prompt = f"""
            你是一位专业的肾脏病专家。请基于以下知识库分析患者情况。
            知识库：{kb_all}
            患者数据：Age {current_patient['age']}, eGFR {current_patient['egfr']}, uACR {current_patient['uacr']}, BP {bp_str}
            
            请严格按照以下 JSON 结构输出 (不要 Markdown):
            {{
                "expert_assessment": {{ "content": "专家深度综述..." }},
                "diagnosis": {{ "summary": "诊断结论", "detail": "详细分期说明", "citation": "依据" }},
                "referral": {{ "advice": "转诊建议", "citation": "依据" }},
                "medications": [
                    {{ "drug": "SGLT2i/RASi等", "status": "推荐/不推荐/待评估", "reason": "...", "citation": "..." }}
                ],
                "lifestyle": {{ "advice": "...", "citation": "..." }}
            }}
            """

            try:
                # 配置模型输出
                safe_config = genai.types.GenerationConfig(
                    temperature=0.2,
                    response_mime_type="application/json"
                )

                with st.spinner("Gemini 正在进行深度推理..."):
                    res = model.generate_content(expert_prompt, generation_config=safe_config)
                    report = json.loads(res.text)
                
                # --- 0. 专家点评 ---
                assess = report.get('expert_assessment', {})
                # 容错处理：如果 assess 是字符串（虽然不太可能），转为字典
                if isinstance(assess, str): assess = {"content": assess}
                
                st.markdown(f"""
                <div class="expert-card">
                    <h4 style="margin:0 0 10px 0; color:#F57F17; font-size:1.1em;">🧠 首席专家深度综述</h4>
                    <p style="margin:0; color:#333; line-height:1.6; font-size:1.05em; font-weight:500;">
                        {assess.get('content', '未生成点评内容')}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # --- 1. 诊断与转诊 ---
                # 【容错修复】检查类型，如果是字符串，手动包装成字典，防止报错
                diag = report.get('diagnosis', {})
                if isinstance(diag, str): diag = {"summary": diag, "detail": "详见综述", "citation": "N/A"}
                
                ref = report.get('referral', {})
                if isinstance(ref, str): ref = {"advice": ref, "citation": "N/A"}

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div style="background-color:#E3F2FD; padding:15px; border-radius:10px; border-left: 5px solid #2196F3; margin-bottom:15px;">
                        <h4 style="margin:0; color:#0D47A1;">🩺 诊断: {diag.get('summary', '未知')}</h4>
                        <p style="margin:8px 0; color:#333;">{diag.get('detail', '暂无详情')}</p>
                        <div style="font-size:0.85em; color:#546E7A; border-top:1px dashed #BBDEFB; padding-top:5px;">📚 {diag.get('citation', 'N/A')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div style="background-color:#E8F5E9; padding:15px; border-radius:10px; border-left: 5px solid #4CAF50; margin-bottom:15px;">
                        <h4 style="margin:0; color:#1B5E20;">🏥 转诊建议</h4>
                        <p style="margin:8px 0; color:#333;">{ref.get('advice', '暂无建议')}</p>
                        <div style="font-size:0.85em; color:#558B2F; border-top:1px dashed #C8E6C9; padding-top:5px;">📚 {ref.get('citation', 'N/A')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # --- 2. 药物列表 ---
                st.markdown("#### 💊 循证用药筛查")
                meds = report.get("medications", [])
                
                for drug in meds:
                    # 【容错修复】如果 drug 是字符串（比如 AI 返回了文本列表），将其转化为对象
                    if isinstance(drug, str):
                        drug = {"drug": drug, "status": "提示", "reason": "详情请见综述", "citation": "N/A"}
                        
                    is_positive = "推荐" in drug.get('status', '') and "不" not in drug.get('status', '')
                    icon, color = ("✅", "#1B5E20") if is_positive else ("⚠️", "#B71C1C")
                    st.markdown(f"""
                    <div style="border:1px solid #eee; background-color:#FAFAFA; padding:12px; border-radius:8px; margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <strong>{icon} {drug.get('drug')}</strong>
                            <span style="background-color:{color}; color:white; padding:2px 8px; border-radius:12px; font-size:0.8em;">{drug.get('status')}</span>
                        </div>
                        <div style="margin-top:8px; color:#444;">{drug.get('reason')}</div>
                        <div style="margin-top:5px; font-size:0.8em; color:#999; text-align:right;">📖 依据: {drug.get('citation', 'N/A')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # --- 3. 生活方式 ---
                life = report.get('lifestyle', {})
                # 【容错修复】
                if isinstance(life, str): life = {"advice": life, "citation": "N/A"}
                
                st.markdown("#### 🥗 生活方式管理")
                st.markdown(f"""
                <div style="border-left: 3px solid #FF9800; padding-left:10px; color:#555;">
                    {life.get('advice', '暂无建议')}<br>
                    <span style="font-size:0.8em; color:#999;">📖 {life.get('citation', 'N/A')}</span>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"决策引擎异常: {e}")

                if 'res' in locals(): st.text_area("原始响应内容", res.text)
