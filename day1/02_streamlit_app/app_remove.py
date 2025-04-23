# app.py
import streamlit as st
import ui                   # UIモジュール
import llm                  # LLMモジュール
import database             # データベースモジュール
import metrics              # 評価指標モジュール
import data                 # データモジュール
import torch
from transformers import pipeline
from config import MODEL_NAME
from huggingface_hub import HfFolder

## 改善方針
# - 時間間に合わず断念 モデルを `AutoModelForCausalLM` + `AutoTokenizer` に切り替えて制御性と速度を向上
# - BLEUスコアを用いた自動評価指標を導入，応答品質を可視化
# - チャット画面で履歴を左サイドバーに表示，会話文脈の把握を強化

# ----- モデル名 -----
# MODEL_NAME = "distilgpt2"

# ----- BLEUスコアを評価指標に導入 From -----
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_bleu(reference, hypothesis):
    reference = [reference.split()]
    hypothesis = hypothesis.split()
    return sentence_bleu(reference, hypothesis, smoothing_function=SmoothingFunction().method1)

# ----- BLEUスコアを評価指標に導入 To -----


# --- アプリケーション設定 ---
st.set_page_config(page_title="Gemma Chatbot", layout="wide")

# --- 初期化処理 ---
# NLTKデータのダウンロード（初回起動時など）
metrics.initialize_nltk()

# データベースの初期化（テーブルが存在しない場合、作成）
database.init_db()

# データベースが空ならサンプルデータを投入
data.ensure_initial_data()

# LLMモデルのロード（キャッシュを利用）
# モデルをキャッシュして再利用
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}") # 使用デバイスを表示
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device
        )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None
pipe = llm.load_model()

# ---- モデル改善 From -----
# モデルを AutoModelForCausalLM + AutoTokenizer に変更
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# def load_model():
#     try:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#         model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)
#         st.success(f"✅ モデル '{MODEL_NAME}' を {device} 上でロードしました")
#         return lambda prompt: tokenizer.decode(
#             model.generate(tokenizer(prompt, return_tensors="pt").input_ids.to(device), max_new_tokens=128)[0],
#             skip_special_tokens=True
#         )
#     except Exception as e:
#         st.error(f"モデルの読み込みに失敗: {e}")
#         return None

# nltk.download("punkt") # NLTK初期化

# pipe = load_model()

# ---- モデル改善 To -----

# ----- セッション初期化（履歴管理） -----
if "history" not in st.session_state:
    st.session_state.history = []

# --- Streamlit アプリケーション ---
# st.title("🤖 Gemma 2 Chatbot with Feedback")
# st.write("Gemmaモデルを使用したチャットボットです。回答に対してフィードバックを行えます。")
#  ----- 変更 From -----
st.title("🤖 Gemma Chatbot with BLEU Feedback")
st.write("Gemmaモデルを用いたチャットボットです。各応答に自動評価（BLEUスコア）を表示します。")
#  ----- 変更 To -----

st.markdown("---")

# ----- 左サイドバー：履歴表示 From -----
st.sidebar.title("💬 チャット履歴")
for idx, (user, bot) in enumerate(st.session_state.history[::-1]):
    st.sidebar.markdown(f"**Q{len(st.session_state.history)-idx}**: {user}")
    st.sidebar.caption(f"→ {bot[:30]}...")

# ----- 左サイドバー：履歴表示 To -----


# # --- サイドバー ---
# st.sidebar.title("ナビゲーション")
# # セッション状態を使用して選択ページを保持
# if 'page' not in st.session_state:
#     st.session_state.page = "チャット" # デフォルトページ

# page = st.sidebar.radio(
#     "ページ選択",
#     ["チャット", "履歴閲覧", "サンプルデータ管理"],
#     key="page_selector",
#     index=["チャット", "履歴閲覧", "サンプルデータ管理"].index(st.session_state.page), # 現在のページを選択状態にする
#     on_change=lambda: setattr(st.session_state, 'page', st.session_state.page_selector) # 選択変更時に状態を更新
# )


# # --- メインコンテンツ ---
# if st.session_state.page == "チャット":
#     if pipe:
#         ui.display_chat_page(pipe)
#     else:
#         st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。")
# elif st.session_state.page == "履歴閲覧":

#     ui.display_history_page()
# elif st.session_state.page == "サンプルデータ管理":
#     ui.display_data_page()


# ----- メイン：チャット入力・応答 From -----
user_input = st.text_input("あなたの質問を入力してください")

if st.button("送信") and user_input:
    # 🔧 応答のテキストだけを抽出
    response = pipe(user_input, max_new_tokens=1000)[0]['generated_text']

    # 応答表示
    st.markdown("### 🤖 モデルの応答")
    st.success(response)

    # BLEUスコア計算
    bleu = compute_bleu(user_input, response)
    st.markdown(f"🧠 BLEUスコア: `{bleu:.2f}`")

    # 履歴保存
    st.session_state.history.append((user_input, response))

# ----- メイン：チャット入力・応答 To -----


# --- フッターなど（任意） ---
st.sidebar.markdown("---")
st.sidebar.info("変更者 : Yasuhiro Seino")






st.sidebar.markdown("---")
st.sidebar.info("開発者: [Your Name]")