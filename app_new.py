"""
ISOM5240 智能零售营销助手（新版本）
流程：商品图片 → BLIP 图像描述 → GPT-2 广告文案生成
运行：streamlit run app_new.py
"""

import streamlit as st
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ISOM5240 Retail AI Assistant",
    page_icon="🛍️",
)

st.title("🛍️ 智能零售营销助手")
st.write("上传一张商品图片，AI 将自动生成商品描述并创作广告词。")

# ---------------------------------------------------------------------------
# 模型配置与加载
# ---------------------------------------------------------------------------
BLIP_MODEL = "SCM1120/blip-fashion-finetuned"  # 可改为 "Salesforce/blip-image-captioning-base"


@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(device)
    text_generator = pipeline(
        "text-generation",
        model="gpt2",
        device=0 if device == "cuda" else -1,
    )
    return processor, blip_model, text_generator, device


processor, blip_model, text_generator, device = load_models()

# ---------------------------------------------------------------------------
# 上传与展示
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "选择商品图片...",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.stop()

image = Image.open(uploaded_file)
st.image(image, caption="上传的商品图片", use_container_width=True)
st.divider()

# ---------------------------------------------------------------------------
# 第一步：BLIP 图像描述
# ---------------------------------------------------------------------------
with st.spinner("正在生成商品描述..."):
    image_rgb = image.convert("RGB")
    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True).strip()
    product_desc = caption if caption else "商品"

st.subheader("第一步：商品描述 (BLIP Image Captioning)")
st.success(f"描述: **{product_desc}**")

# ---------------------------------------------------------------------------
# 第二步：GPT-2 广告文案生成
# ---------------------------------------------------------------------------
with st.spinner("正在创作广告文案..."):
    prompt = (
        "The following is a creative advertisement for a professional retail product.\n"
        f"Product: {product_desc}\n"
        "Ad Copy:"
    )
    result = text_generator(
        prompt,
        max_length=100,
        num_return_sequences=1,
        truncation=True,
        pad_token_id=50256,
    )
    full_text = result[0]["generated_text"]
    ad_copy = full_text.replace(prompt, "").strip()

st.subheader("第二步：广告生成 (Ad Generation)")
st.info(ad_copy if ad_copy else "正在努力构思中...")

# ---------------------------------------------------------------------------
# 技术逻辑说明
# ---------------------------------------------------------------------------
with st.expander("查看技术逻辑 (Technical Logic)"):
    st.write(
        "1. **Vision-Language (BLIP)**: 使用微调后的 BLIP 模型从商品图像生成描述文本。"
    )
    st.write(f"2. **NLP Bridge**: 将描述「{product_desc}」作为上下文输入给 GPT-2。")
    st.write("3. **Generative AI**: 通过自回归预测生成后续营销文本。")
