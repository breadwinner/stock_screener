import streamlit as st
import pandas as pd
import google.generativeai as genai
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries
import pandas_ta as ta
import time
import yfinance as yf
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
llm_api_key = load_key_securely("GOOGLE_API_KEY", "Google Gemini Key")

# 检查最终状态
if not llm_api_key:
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

# --- 替换原有的 verify_stock_data 函数 (yfinance 版) ---
def verify_stock_data(symbol, api_key=None): 
    # 注意：yfinance 不需要 api_key，这里保留参数是为了兼容之前的调用格式
    try:
        # 1. 初始化 Ticker
        ticker = yf.Ticker(symbol)
        
        # 2. 获取基本面数据 (Info)
        # yfinance 的 info 有时请求较慢，但这步是必须的
        info = ticker.info
        
        # 提取关键指标
        current_price = info.get('currentPrice', 0.0)
        # 如果没有 currentPrice，尝试获取 previousClose
        if current_price == 0:
            current_price = info.get('previousClose', 0.0)

        pe = info.get('forwardPE', 0.0)
        # 如果 Forward PE 为 None (比如亏损股), 设为 0
        if pe is None: pe = 0.0
            
        sector = info.get('sector', 'Unknown')
        
        # 3. 获取技术面数据 (History)
        # 获取过去 3 个月数据用于计算 RSI 和 回撤
        hist = ticker.history(period="3mo")
        
        if hist.empty:
            return None
            
        # 计算技术指标
        # 52周高点 (用3个月高点近似，或者用 info['fiftyTwoWeekHigh'])
        high_52 = info.get('fiftyTwoWeekHigh', hist['Close'].max())
        drop_pct = (current_price - high_52) / high_52
        
        # 计算 RSI
        rsi_series = ta.rsi(hist['Close'], length=14)
        if rsi_series is None or rsi_series.empty:
            rsi = 50.0 # 默认值
        else:
            rsi = rsi_series.iloc[-1]
        
        # 4. 评分逻辑
        score = 0
        if drop_pct < -0.15: score += 40      # 跌幅深
        if rsi < 45: score += 30              # 超卖
        if 0 < pe < 35: score += 30           # 估值合理 (0意味着亏损，排除)
        
        return {
            "代码": symbol,
            "行业": sector,
            "现价": round(current_price, 2),
            "动态PE": round(pe, 2),
            "跌幅": f"{round(drop_pct*100, 1)}%",
            "RSI": round(rsi, 1),
            "AI评分": score,
            "建议": "✅ 关注" if score >= 70 else "👀 观察"
        }

    except Exception as e:
        # st.error(f"{symbol} 分析失败: {e}") # 调试时可打开
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
        if st.button("运行量化验证 (Yahoo Finance)"): # 按钮名字改一下
            results = []
            progress = st.progress(0)
            
            for i, ticker in enumerate(picks):
                # 注意：这里不需要传 av_api_key 了，传 None 即可
                data = verify_stock_data(ticker, None) 
                if data: results.append(data)
                
                # yfinance 很快，不需要睡 12秒，睡 0.1秒 给 UI 刷新留点时间即可
                time.sleep(0.1) 
                progress.progress((i+1)/len(picks))
            
            if results:
                df = pd.DataFrame(results).sort_values(by="AI评分", ascending=False)
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
