import streamlit as st
import pandas as pd
import google.generativeai as genai
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries
import pandas_ta as ta
import time
from datetime import datetime
import os
# 如果在本地运行且使用 .env 文件，需要安装 python-dotenv
# pip install python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- 1. 基础配置与 API Key 管理 ---
st.set_page_config(page_title="AI 动态美股筛选器", layout="wide")
st.title("⏱️ AI 动态美股筛选器 (Secrets/Env 集成版)")

def get_api_key(key_name):
    """
    获取 API Key 的通用函数
    优先级: 1. Streamlit Secrets (云端/toml) -> 2. 环境变量 (.env) -> 3. 空
    """
    if key_name in st.secrets:
        return st.secrets[key_name]
    elif os.getenv(key_name):
        return os.getenv(key_name)
    else:
        return ""

# --- 2. 侧边栏配置 (安全版) ---
st.sidebar.header("⚙️ 参数与密钥")

# --- 安全加载逻辑 ---
# 定义一个辅助函数来处理 Key 的显示逻辑
def load_key_securely(key_name, display_name):
    # 1. 尝试从 Secrets 或 Env 获取
    env_key = get_api_key(key_name)
    
    if env_key:
        # 如果找到了，显示绿色的成功状态，不显示具体 Key，也不渲染输入框
        st.sidebar.success(f"✅ {display_name} 已配置")
        return env_key
    else:
        # 如果没找到，显示空的输入框让用户手动填
        return st.sidebar.text_input(
            f"{display_name}", 
            type="password",
            help="未检测到配置文件，请在此手动输入"
        )

# 调用函数加载 Key
av_api_key = load_key_securely("ALPHA_VANTAGE_KEY", "Alpha Vantage Key")
llm_api_key = load_key_securely("GOOGLE_API_KEY", "Google Gemini Key")

# 检查最终状态
if not av_api_key or not llm_api_key:
    st.sidebar.warning("⚠️ 缺少必要的 API Key，程序无法运行。")
    st.stop() # 强制停止后续代码运行，防止报错

st.sidebar.markdown("---")

# 2.2 动态时间设置
system_date = datetime.now().strftime("%Y年%m月%d日")
analysis_date = st.sidebar.text_input("分析时间锚点", value=system_date, help="AI 将基于此时间点分析市场环境")
st.sidebar.caption(f"系统当前日期: {system_date}")

# --- 3. 核心逻辑: 动态 Prompt ---
STRATEGY_PROMPT = f"""
Role: 资深美股量化分析师。
Context: 假设现在的市场时间是 **{analysis_date}**。
Task: 请基于这个时间点的宏观环境，筛选出 5-8 只纳斯达克，道琼斯或标普500成分股。
Criteria:
1. 错杀型 (Deep Value): 股价较{analysis_date}前的高点下跌超过15%，但基本面（营收/EPS）依然健康。
2. 资金流 (Money Flow): 近期成交量有异动，或处于行业轮动（Sector Rotation）的受益区。
3. 行业偏好: 重点扫描 SaaS、半导体、医疗器械或金融科技。
Output Format: 仅输出股票代码(Ticker)，用英文逗号隔开，不要包含任何解释或Markdown格式。
Example: AAPL, MSFT, TTD, BAX
"""

# --- 4. 功能函数 ---

def get_ai_picks(api_key, prompt):
    """调用 LLM 生成股票名单"""
    try:
        if not api_key:
            return []
        
        # 配置 Google Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        with st.spinner(f'AI 正在基于 [{analysis_date}] 的市场环境进行思考...'):
            response = model.generate_content(prompt)
            text = response.text
            # 清洗数据
            tickers = [t.strip().upper() for t in text.replace('\n', '').replace('`', '').split(',') if t.strip()]
            return tickers
    except Exception as e:
        st.error(f"AI 调用错误: {str(e)}")
        return []

def verify_stock_data(symbol, api_key):
    """调用 Alpha Vantage 验证数据"""
    try:
        fd = FundamentalData(key=api_key, output_format='pandas')
        ts = TimeSeries(key=api_key, output_format='pandas')
        
        # 基本面
        overview, _ = fd.get_company_overview(symbol=symbol)
        if overview.empty: return None
        
        pe = float(overview['ForwardPE'].iloc[0]) if 'ForwardPE' in overview.columns and overview['ForwardPE'].iloc[0] != 'None' else 0
        sector = overview['Sector'].iloc[0]
        
        # 技术面 (取最近60天)
        df, _ = ts.get_daily_adjusted(symbol=symbol)
        df = df.head(60)
        
        curr = df['5. adjusted close'].iloc[0]
        high = df['5. adjusted close'].max()
        drop = (curr - high) / high
        rsi = ta.rsi(df['5. adjusted close'], length=14).iloc[0]
        
        # 评分逻辑
        score = 0
        if drop < -0.15: score += 40
        if rsi < 45: score += 30
        if 0 < pe < 35: score += 30
        
        return {
            "代码": symbol,
            "行业": sector,
            "当前价": round(curr, 2),
            "动态PE": pe,
            "距高点跌幅": f"{round(drop*100, 1)}%",
            "RSI (14)": round(rsi, 1),
            "AI 推荐分": score,
            "状态": "✅ 重点关注" if score >= 70 else "👀 观察"
        }
    except Exception as e:
        # st.error(f"{symbol} 数据获取失败: {e}") # 调试用，生产环境可注释
        return None

# --- 5. 主界面逻辑 (修复重点) ---

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1️⃣ AI 策略")
    st.info(f"时间: {analysis_date}")
    
    # 按钮 1: AI 筛选
    if st.button("开始 AI 筛选"):
        picks = get_ai_picks(llm_api_key, STRATEGY_PROMPT)
        if picks:
            # 【关键修复】存入 Session State
            st.session_state['ai_picks'] = picks
            st.success(f"已生成: {', '.join(picks)}")

with col2:
    st.subheader("2️⃣ 量化验证结果")
    
    # 检查是否有 AI 筛选结果
    if 'ai_picks' in st.session_state:
        picks = st.session_state['ai_picks']
        st.write(f"待验证: {picks}")
        
        # 按钮 2: 运行数据验证
        if st.button("运行 Alpha Vantage 验证"):
            results = []
            progress = st.progress(0)
            
            for i, ticker in enumerate(picks):
                data = verify_stock_data(ticker, av_api_key)
                if data: results.append(data)
                # 避免 API 速率限制 (免费版)
                time.sleep(12) if len(picks) > 2 else time.sleep(1)
                progress.progress((i+1)/len(picks))
            
            if results:
                df = pd.DataFrame(results).sort_values(by="AI评分", ascending=False)
                # 【关键修复】将最终结果存入 Session State，而不是只在按钮内部显示
                st.session_state['final_df'] = df
        
        # --- 显示区域 (在按钮外部渲染) ---
        # 只要 session_state 里有结果，就一直显示表格
        if 'final_df' in st.session_state:
            final_df = st.session_state['final_df']
            
            # 样式高亮
            def highlight(row):
                return ['background-color: #d4edda' if row['建议'] == '✅ 关注' else '' for _ in row]
            
            st.dataframe(final_df.style.apply(highlight, axis=1), use_container_width=True)
            
            # 添加下载按钮
            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 下载 CSV", csv, "market_analysis.csv", "text/csv")
            
    else:
        st.info("请先在左侧运行 AI 筛选")
