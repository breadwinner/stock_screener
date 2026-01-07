import streamlit as st
import pandas as pd
import google.generativeai as genai
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries
import pandas_ta as ta
import time
from datetime import datetime

# --- 页面配置 ---
st.set_page_config(page_title="AI 动态美股筛选器", layout="wide")
st.title("⏱️ AI 动态美股筛选器 (自动同步当前时间)")

# --- 1. 动态时间逻辑 (核心修改点) ---
# 获取系统当前日期，格式如：2025年05月
system_date = datetime.now().strftime("%Y年%m月%d日")

st.sidebar.header("⚙️ 参数设置")
# 默认使用系统时间，但允许用户手动修改以进行回测或模拟
analysis_date = st.sidebar.text_input("分析时间锚点", value=system_date, help="AI 将基于此时间点分析市场环境")

st.sidebar.caption(f"系统检测当前时间: {system_date}")

# --- API 配置 ---
av_api_key = st.sidebar.text_input("Alpha Vantage Key", type="password")
llm_api_key = st.sidebar.text_input("Gemini/OpenAI Key", type="password")

# --- 2. 动态 Prompt 构建 ---
# 使用 f-string 将 analysis_date 动态嵌入
STRATEGY_PROMPT = f"""
Role: 资深美股量化分析师。
Context: 假设现在的市场时间是 **{analysis_date}**。
Task: 请基于这个时间点的宏观环境，筛选出 5-8 只纳斯达克, 道琼斯或标普500成分股。
Criteria:
1. 错杀型 (Deep Value): 股价较{analysis_date}前的高点下跌超过15%，但基本面（营收/EPS）依然健康。
2. 资金流 (Money Flow): 近期成交量有异动，或处于行业轮动（Sector Rotation）的受益区。
3. 行业偏好: 重点扫描 SaaS、半导体、医疗器械或金融科技。
Output Format: 仅输出股票代码(Ticker)，用英文逗号隔开，不要包含任何解释或Markdown格式。
Example: AAPL, MSFT, TTD, BAX
"""

# --- 3. 核心功能函数 ---

def get_ai_picks(api_key, prompt):
    """调用 LLM 生成股票名单"""
    try:
        if not api_key:
            return []
        
        # 配置 Google Gemini (或根据需要改为 OpenAI)
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3.0-flash')
        
        with st.spinner(f'AI 正在基于 [{analysis_date}] 的市场环境进行思考...'):
            response = model.generate_content(prompt)
            text = response.text
            # 清洗数据：移除换行、空格，只保留代码
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
        
        # 获取基本面 (PE)
        overview, _ = fd.get_company_overview(symbol=symbol)
        if overview.empty: return None
        
        pe = float(overview['ForwardPE'].iloc[0]) if 'ForwardPE' in overview.columns else 0
        sector = overview['Sector'].iloc[0]
        
        # 获取技术面 (RSI, 跌幅)
        df, _ = ts.get_daily_adjusted(symbol=symbol)
        df = df.head(60) # 取最近60个交易日
        
        curr = df['5. adjusted close'].iloc[0]
        high = df['5. adjusted close'].max()
        drop = (curr - high) / high
        rsi = ta.rsi(df['5. adjusted close'], length=14).iloc[0]
        
        # 简单的评分逻辑
        score = 0
        if drop < -0.15: score += 40      # 跌幅够深
        if rsi < 45: score += 30          # 处于超卖区
        if 0 < pe < 35: score += 30       # 估值合理
        
        return {
            "代码": symbol,
            "行业": sector,
            "当前价": round(curr, 2),
            "动态PE": pe,
            "距高点跌幅": f"{round(drop*100, 1)}%",
            "RSI (14)": round(rsi, 1),
            "AI 推荐分": score,
            "状态": "✅ 值得关注" if score >= 70 else "⚠️ 需谨慎"
        }
    except Exception:
        return None

# --- 4. 主界面逻辑 ---

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1️⃣ AI 策略生成")
    st.info(f"当前 Prompt 时间设定: **{analysis_date}**")
    st.code(STRATEGY_PROMPT, language='markdown')
    
    if st.button("开始 AI 筛选"):
        if not llm_api_key:
            st.warning("请先在左侧输入 LLM API Key")
        else:
            picks = get_ai_picks(llm_api_key, STRATEGY_PROMPT)
            if picks:
                st.session_state['picks'] = picks
                st.success(f"AI 筛选出 {len(picks)} 只标的: {', '.join(picks)}")

with col2:
    st.subheader("2️⃣ 量化数据验证")
    if 'picks' in st.session_state:
        target_tickers = st.session_state['picks']
        st.write(f"待验证列表: {target_tickers}")
        
        if st.button("运行 Alpha Vantage 验证"):
            if not av_api_key:
                st.error("请输入 Alpha Vantage API Key")
            else:
                results = []
                my_bar = st.progress(0)
                
                for i, ticker in enumerate(target_tickers):
                    data = verify_stock_data(ticker, av_api_key)
                    if data: results.append(data)
                    time.sleep(12) # 免费 Key 频率限制
                    my_bar.progress((i+1)/len(target_tickers))
                
                if results:
                    df = pd.DataFrame(results).sort_values(by="AI 推荐分", ascending=False)
                    st.dataframe(df.style.highlight_max(axis=0, subset=['AI 推荐分'], color='#d4edda'))
