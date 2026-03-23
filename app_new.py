"""
ISOM5240 智能零售营销助手（独立新文件）
使用 nlpconnect/vit-gpt2-image-captioning（图像描述）+ Step2 微调 GPT-2（广告文案）。
流程：商品图片 → ViT-GPT2 描述 → GPT-2 广告文案
运行：streamlit run app_streamlit.py
"""

import streamlit as st
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
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
st.write("上传商品图片，使用 ViT-GPT2 生成描述，再由微调 GPT-2 生成广告文案。")

# ---------------------------------------------------------------------------
# 模型配置
# ---------------------------------------------------------------------------
VIT_GPT2_REPO = "nlpconnect/vit-gpt2-image-captioning"
GPT2_REPO     = "SCM1120/gpt2-ad-finetuned"


@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 第一步：ViT-GPT2 图像描述模型 ----
    vit_processor = ViTImageProcessor.from_pretrained(VIT_GPT2_REPO)
    vit_tokenizer = AutoTokenizer.from_pretrained(VIT_GPT2_REPO)
    vit_model     = VisionEncoderDecoderModel.from_pretrained(VIT_GPT2_REPO).to(device)

    # ---- 第二步：微调 GPT-2 广告文案模型 ----
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_REPO)
    gpt2_tokenizer.pad_token    = gpt2_tokenizer.eos_token
    gpt2_tokenizer.padding_side = "left"
    gpt2_model = GPT2LMHeadModel.from_pretrained(GPT2_REPO).to(device)

    return vit_processor, vit_tokenizer, vit_model, gpt2_tokenizer, gpt2_model, device


def clean_ad_text(raw: str) -> str:
    """只保留「Ad:」后的内容，去掉无意义的 Price/Color/Size 行，取前几句。"""
    if not raw or not raw.strip():
        return ""
    if "Ad:" in raw:
        raw = raw.split("Ad:")[-1].strip()
    lines = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.endswith(":") and len(line) < 30:
            continue
        lines.append(line)
    merged = " ".join(lines)
    for sep in (". ", "。", "\n"):
        merged = merged.replace(sep, " <<S>> ")
    parts = [p.strip() for p in merged.split(" <<S>> ") if len(p.strip()) > 10][:3]
    result = " ".join(parts).strip()
    if not result:
        result = " ".join(lines[:3]).strip() or raw.strip()[:400]
    return result


# ---------------------------------------------------------------------------
# 加载模型
# ---------------------------------------------------------------------------
vit_processor, vit_tokenizer, vit_model, gpt2_tokenizer, gpt2_model, device = load_models()

# ---------------------------------------------------------------------------
# 上传与展示
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "选择商品图片...",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="上传的商品图片", use_container_width=True)
st.divider()

# ---------------------------------------------------------------------------
# 第一步：ViT-GPT2 生成商品描述
# ---------------------------------------------------------------------------
with st.spinner("正在生成商品描述（ViT-GPT2）..."):
    pixel_values = vit_processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        caption_ids = vit_model.generate(
            pixel_values,
            max_new_tokens=40,
            num_beams=4,
            early_stopping=True,
            repetition_penalty=1.5,
        )
    caption = vit_tokenizer.decode(caption_ids[0], skip_special_tokens=True).strip()


def _is_bad_caption(t: str) -> bool:
    if not t or len(t) < 2:
        return True
    if all(c in ".\u00b7\u2022 \t" for c in t.replace(" ", "")):
        return True
    words = t.lower().split()
    if len(words) >= 3 and len(set(words)) == 1:
        return True
    if len(words) <= 4 and len(set(words)) <= 2:
        return True
    return False


if _is_bad_caption(caption):
    caption = "product"
product_desc = caption

st.subheader("第一步：商品描述 (ViT-GPT2 Image Captioning)")
st.success(f"描述: **{product_desc}**")

# ---------------------------------------------------------------------------
# 第二步：GPT-2 生成广告文案
# ---------------------------------------------------------------------------
with st.spinner("正在创作广告文案（GPT-2 微调模型）..."):
    prompt = f"Product: {product_desc}\nDescription: {product_desc}\nAd:"
    inputs = gpt2_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gpt2_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=gpt2_tokenizer.eos_token_id,
        )
    full_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    ad_copy = clean_ad_text(full_text)

st.subheader("第二步：广告生成 (Ad Generation)")
if ad_copy:
    st.info(ad_copy)
else:
    st.warning("本次未生成有效文案，请换一张商品图或稍后重试。")

# ---------------------------------------------------------------------------
# 技术逻辑说明
# ---------------------------------------------------------------------------
with st.expander("查看技术逻辑 (Technical Logic)"):
    st.write("1. **ViT-GPT2**: 使用 `nlpconnect/vit-gpt2-image-captioning`，从图片生成商品英文描述。")
    st.write(f"2. **Bridge**: 将描述「{product_desc}」按 Step2 格式构造 Product + Description + Ad。")
    st.write("3. **GPT-2（微调）**: Step2 在 Product-Descriptions-and-Ads 上微调，续写广告文案。")
