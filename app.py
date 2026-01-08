import streamlit as st
import pandas as pd
import google.generativeai as genai
import yfinance as yf
import pandas_ta as ta
import time
from datetime import datetime
import os

# 尝试加载本地 .env 文件 (用于本地开发)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- 1. 页面配置 ---
st.set_page_config(page_title="AI 美股猎手 (Yahoo版)", layout="wide", page_icon="🚀")
st.title("🚀 AI 美股猎手: 智能筛选 + Yahoo Finance 极速验证")
st.caption("无 API 频率限制 | 极速量化分析 | 自动纠错")

# --- 2. 安全加载 Key ---
def load_key_securely(key_name, display_name):
    """优先从 Secrets/Env 读取，否则显示输入框"""
    val = st.secrets.get(key_name, os.getenv(key_name, ""))
    
    if val:
        st.sidebar.success(f"✅ {display_name} 已激活")
        return val
        
# --- 3. 侧边栏配置 ---
st.sidebar.header("⚙️ 全局配置")

# 3.1 获取 Google API Key (用于 AI 思考)
llm_api_key = load_key_securely("GOOGLE_API_KEY", "Google Gemini Key")

# 3.2 动态时间设置
default_date = datetime.now().strftime("%Y年%m月%d日")
analysis_date = st.sidebar.text_input("分析时间锚点", value=default_date)

st.sidebar.markdown("---")
st.sidebar.info("💡 **提示**: Yahoo Finance 接口完全免费且无硬性限制，但请保持网络通畅（访问国际互联网）。")

# --- 4. AI 策略定义 ---
STRATEGY_PROMPT = f"""
Role: 资深美股量化分析师。
Context: 假设现在的市场时间是 **{analysis_date}**。
Task: 请基于这个时间点的宏观环境，筛选出 5-8只标普500成分股。
Criteria:
1. 错杀型 (Deep Value): 当前动态市盈率（Forward P/E）显著低于过去3年的中位数, 股价较{analysis_date}前的高点下跌超过15%，但基本面（营收/EPS）依然健康。
2. 资金流 (Money Flow): 近期成交量有异动，或处于行业轮动（Sector Rotation）的受益区。
Output Format: 仅输出股票代码(Ticker)，用英文逗号隔开，不要包含任何解释或Markdown格式。
"""

# --- 5. 核心功能函数 ---

def get_ai_picks(api_key, prompt):
    """第一步: 让 AI 生成名单"""
    try:
        if not api_key:
            st.error("❌ 请先配置 Google Gemini Key")
            return []
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        with st.spinner(f'🧠 AI 正在基于 [{analysis_date}] 的市场环境进行深度扫描...'):
            response = model.generate_content(prompt)
            # 极强的数据清洗逻辑
            raw_text = response.text.replace('\n', '').replace('`', '').replace('"', '').replace("'", "")
            tickers = [t.strip().upper() for t in raw_text.split(',') if t.strip()]
            return tickers
    except Exception as e:
        st.error(f"AI 调用失败: {e}")
        return []

def verify_stock_yahoo(symbol):
    """第二步: 使用 Yahoo Finance 验证数据"""
    # 1. 格式清洗 (Yahoo 对格式很敏感)
    symbol = symbol.strip().upper()
    # 修正特殊代码: 比如 BRK.B -> BRK-B
    clean_symbol = symbol.replace('.', '-').replace('NASDAQ:', '').replace('NYSE:', '')
    
    try:
        ticker = yf.Ticker(clean_symbol)
        
        # 2. 获取基础信息 (Info)
        # 注意: yf.Ticker.info 可能会慢，设置超时或容错
        try:
            info = ticker.info
        except:
            info = {} # 降级处理
        
        # 3. 智能获取价格 (双重保障)
        # fast_info 通常比 info 快 10 倍
        try:
            curr_price = ticker.fast_info['last_price']
        except:
            curr_price = info.get('currentPrice', info.get('regularMarketPrice', 0.0))
            
        if curr_price == 0:
            return None # 拿不到价格通常意味着代码无效
            
        # 4. 获取估值与行业
        pe = info.get('forwardPE', info.get('trailingPE', 0.0))
        if pe is None: pe = 0.0
        sector = info.get('sector', 'Unknown')
        name = info.get('shortName', clean_symbol)

        # 5. 获取技术面 (K线数据)
        # 获取 3 个月数据以计算 RSI 和 回撤
        hist = ticker.history(period="3mo")
        
        if hist.empty:
            return None
            
        # 计算 52周高点 (用近期高点近似，或者尝试读取 info)
        high_52 = info.get('fiftyTwoWeekHigh', hist['Close'].max())
        if not high_52: high_52 = curr_price
        
        drop_pct = (curr_price - high_52) / high_52
        
        # 计算 RSI
        rsi_series = ta.rsi(hist['Close'], length=14)
        rsi = rsi_series.iloc[-1] if not rsi_series.empty else 50.0
        
        # 计算成交量异动 (今日量 vs 20日均量)
        vol_today = hist['Volume'].iloc[-1]
        vol_avg = hist['Volume'].mean()
        vol_ratio = vol_today / vol_avg if vol_avg > 0 else 1.0

        # 6. 量化评分模型
        score = 0
        reasons = []
        
        if drop_pct < -0.15: 
            score += 40
            reasons.append("超跌")
        if rsi < 40: 
            score += 30
            reasons.append("RSI超卖")
        elif rsi > 70:
            reasons.append("RSI超买")
            
        if 0 < pe < 25: 
            score += 30
            reasons.append("低估值")
        
        if vol_ratio > 1.5:
            score += 10
            reasons.append("放量")

        if drop_pct > -0.05 and rsi > 50:
            return None 
            
        return {
            "代码": clean_symbol,
            "名称": name,
            "行业": sector,
            "现价": round(curr_price, 2),
            "动态PE": round(pe, 1),
            "距高点跌幅": f"{round(drop_pct*100, 1)}%",
            "RSI(14)": round(rsi, 1),
            "量比": round(vol_ratio, 1),
            "AI评分": min(score, 100), # 封顶100
            "标签": " ".join(reasons) if reasons else "平稳"
        }

    except Exception:
        # st.error(f"{clean_symbol} 验证出错") # 调试时可打开
        return None

# --- 6. 主界面逻辑 ---

col1, col2 = st.columns([1, 2.5])

# === 左侧: AI 策略生成 ===
with col1:
    st.subheader("1️⃣ 策略生成")
    st.info(f"时间锚点: {analysis_date}")
    
    # --- 冷却时间逻辑开始 ---
    COOLDOWN_SEC = 30  # 设置冷却时间 30 秒
    
    # 初始化上次运行时间
    if 'last_run_time' not in st.session_state:
        st.session_state['last_run_time'] = 0
    
    # 计算距离上次运行过了多久
    current_time = time.time()
    time_since_last_run = current_time - st.session_state['last_run_time']
    time_remaining = COOLDOWN_SEC - time_since_last_run
    
    # 判断是否在冷却期
    if time_remaining > 0:
        # 冷却中：显示灰色不可点按钮，并显示倒计时
        st.button(f"⏳ 冷却中... 请等待 {int(time_remaining)} 秒", disabled=True)
    # 按钮 A: 生成名单
    else:
        if st.button("开始 AI 选股", type="primary"):
            picks = get_ai_picks(llm_api_key, STRATEGY_PROMPT)
            if picks:
                st.session_state['ai_picks'] = picks # 存入缓存
                st.success(f"AI 已锁定 {len(picks)} 只目标!")
            else:
                st.warning("AI 未返回结果，请检查 Key 或网络。")

    # 显示当前的 AI 名单
    if 'ai_picks' in st.session_state:
        st.write("📋 **目标清单:**")
        st.code(", ".join(st.session_state['ai_picks']))

# === 右侧: 量化验证结果 ===
with col2:
    st.subheader("2️⃣ 量化数据验证 (Yahoo Finance)")
    
    if 'ai_picks' in st.session_state:
        target_list = st.session_state['ai_picks']
        
        # 按钮 B: 运行验证
        if st.button("🚀 运行极速验证"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, ticker in enumerate(target_list):
                status_text.markdown(f"🔍 正在分析: **{ticker}** ...")
                
                # 调用 Yahoo 验证函数
                data = verify_stock_yahoo(ticker)
                
                if data:
                    results.append(data)
                
                # Yahoo 速度很快，稍微给点延迟让 UI 刷新丝滑一点，也可以设为 0
                time.sleep(0.05) 
                progress_bar.progress((i + 1) / len(target_list))
            
            status_text.success("✅ 所有分析已完成！")
            
            # 将结果存入 Session State
            if results:
                df = pd.DataFrame(results).sort_values(by="AI评分", ascending=False)
                st.session_state['final_result'] = df
            else:
                st.error("未能获取任何数据，请检查网络 (Yahoo需访问国际互联网)。")

    # === 结果展示区 (独立渲染) ===
    if 'final_result' in st.session_state:
        final_df = st.session_state['final_result']
        
        # 1. 样式高亮函数
        def highlight_opportunity(row):
            # 绿色: 评分高 (值得买)
            if row['AI评分'] >= 70:
                return ['background-color: #d4edda; color: black'] * len(row)
            # 红色: 评分低或数据异常
            elif row['AI评分'] < 30:
                return ['background-color: #f8d7da; color: black'] * len(row)
            else:
                return [''] * len(row)

        # 2. 渲染表格
        st.dataframe(
            final_df.style.apply(highlight_opportunity, axis=1),
            use_container_width=True,
            column_config={
                "现价": st.column_config.NumberColumn(format="$%.2f"),
                "动态PE": st.column_config.NumberColumn(format="%.1f倍"),
                "AI评分": st.column_config.ProgressColumn(format="%d", min_value=0, max_value=100)
            }
        )
        
        # 3. 下载按钮
        csv = final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下载分析报告 (Excel/CSV)",
            data=csv,
            file_name=f"market_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    elif 'ai_picks' not in st.session_state:
        st.info("👈 请先点击左侧的【开始 AI 选股】按钮")
