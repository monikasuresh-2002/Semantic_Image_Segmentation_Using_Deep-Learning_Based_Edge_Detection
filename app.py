import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from models.UNet import UNet
from utils.count_classes import ID_TO_COLOR
from utils.labels import id2label
import torchvision.transforms as T
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt  # NEW

@st.cache_resource
def load_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_classes=34, depth=5, wf=3, batch_norm=True, padding=True, up_mode='upconv').to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict_image(image, model):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((512, 1024)),
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    return pred

def colorize_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in ID_TO_COLOR.items():
        color_mask[mask == cls_id] = color
    return color_mask

def instance_id_to_label_id(mask):
    return np.vectorize(lambda x: x if x < 1000 else x // 1000)(mask)

def compute_metrics(pred, gt, num_classes=34):
    intersect = np.zeros(num_classes)
    union = np.zeros(num_classes)
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        gt_inds = (gt == cls)
        intersect[cls] = np.logical_and(pred_inds, gt_inds).sum()
        union[cls] = np.logical_or(pred_inds, gt_inds).sum()
    correct = (pred == gt).sum()
    total = pred.size
    pixel_acc = correct / total * 100
    iou = (intersect / np.maximum(union, 1e-10)) * 100
    mean_iou = iou.mean()
    return pixel_acc, mean_iou

def compute_per_class_metrics(pred, gt, num_classes=34):
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    for cls in range(num_classes):
        pred_mask = (pred == cls).astype(np.uint8)
        gt_mask = (gt == cls).astype(np.uint8)
        if np.sum(gt_mask) == 0 and np.sum(pred_mask) == 0:
            continue
        precision[cls] = precision_score(gt_mask.flatten(), pred_mask.flatten(), zero_division=0)
        recall[cls] = recall_score(gt_mask.flatten(), pred_mask.flatten(), zero_division=0)
        f1[cls] = f1_score(gt_mask.flatten(), pred_mask.flatten(), zero_division=0)
    return precision, recall, f1

# ---------------------------------------
# Streamlit Interface
# ---------------------------------------
st.set_page_config(page_title="Cityscapes Road Segmentation", layout="wide")
st.title("🚦 Image Segmentation with U-Net (Cityscapes)")
st.markdown("Upload a **Cityscapes-style image** and its matching **gtFine_labelIds.png** mask to evaluate segmentation quality.")

uploaded_image = st.file_uploader("📤 Upload image (leftImg8bit)", type=["png", "jpg", "jpeg"], key="image")
uploaded_gt = st.file_uploader("📥 Upload matching ground truth (gtFine_labelIds)", type=["png"], key="gt")

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    with st.spinner("🚀 Segmenting..."):
        model = load_model('weights/unet-id3-10e-WCE-d5-MS.pt')
        pred_mask = predict_image(image, model)

        # ---------------------------------------
        # NEW: Show raw prediction map (greyscale)
        # ---------------------------------------
        fig, ax = plt.subplots()
        ax.imshow(pred_mask, cmap='gray')
        ax.set_title("🧩 Raw Class Prediction Map")
        ax.axis('off')
        st.pyplot(fig)
        plt.savefig("raw_prediction_map.png")

        # ---------------------------------------
        # Colorize and Show Segmentation
        # ---------------------------------------
        color_mask = colorize_mask(pred_mask)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 Predicted Segmentation")
            st.image(color_mask, use_container_width=True)
        with col2:
            st.subheader("🧐 Overlay with Original Image")
            overlay = cv2.addWeighted(np.array(image.resize((1024, 512))), 0.6, color_mask, 0.4, 0)
            st.image(overlay, use_container_width=True)

        # ---------------------------------------
        # Classes in Prediction
        # ---------------------------------------
        st.subheader("🗂️ Classes Detected in This Image")
        unique_ids, counts = np.unique(pred_mask, return_counts=True)
        total_pixels = pred_mask.size

        for train_id, count in zip(unique_ids, counts):
            label = id2label.get(int(train_id))
            if label is None or label.ignoreInEval or train_id == 255:
                continue
            r, g, b = label.color
            name = label.name
            percent = (count / total_pixels) * 100
            st.markdown(
                f"<div style='display:flex;align-items:center;margin-bottom:4px;'>"
                f"<div style='width:20px;height:20px;background-color:rgb({r},{g},{b});"
                f"margin-right:10px;border:1px solid #000'></div>"
                f"<span style='font-size:15px;font-weight:500'>{name}</span>"
                f"<span style='font-size:14px;color:gray;margin-left:8px'>"
                f"({count} px, {percent:.2f}%)</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        # ---------------------------------------
        # Evaluation
        # ---------------------------------------
        if uploaded_gt is not None:
            st.subheader("📊 Evaluation Metrics")
            gt_image = Image.open(uploaded_gt)
            gt_mask = np.array(gt_image.convert("L").resize((1024, 512)))
            gt_label_mask = instance_id_to_label_id(gt_mask)

            pixel_acc, mean_iou = compute_metrics(pred_mask, gt_label_mask)
            precision, recall, f1 = compute_per_class_metrics(pred_mask, gt_label_mask)

            st.success(f"✅ Pixel Accuracy: {pixel_acc:.2f}%")
            st.info(f"📏 Mean IoU: {mean_iou:.2f}%")

            data = []
            for cls_id, label in id2label.items():
                if label.ignoreInEval or cls_id >= len(precision):
                    continue
                data.append({
                    "Class": label.name,
                    "Precision (%)": f"{precision[cls_id]:.2f}",
                    "Recall (%)": f"{recall[cls_id]:.2f}",
                    "F1-Score (%)": f"{f1[cls_id]:.2f}"
                })

            df_metrics = pd.DataFrame(data)
            df_metrics = df_metrics.sort_values(by="F1-Score (%)", ascending=False)
            st.subheader("📀 Precision, Recall, F1-score per Class")
            st.table(df_metrics)
