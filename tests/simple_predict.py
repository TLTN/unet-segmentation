import sys
sys.path.append('src')

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def predict_image(image_path):
    # Load model
    from model.unet import UNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet()
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.to(device)
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        prediction = model(image_tensor)

    # Convert to numpy
    pred_np = prediction.squeeze().cpu().numpy()
    pred_binary = (pred_np > 0.5).astype(np.uint8) * 255

    # Show results
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(pred_np, cmap='viridis')
    axes[1].set_title('Probability')
    axes[1].axis('off')

    axes[2].imshow(pred_binary, cmap='gray')
    axes[2].set_title('Binary Mask')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return pred_binary

# Sử dụng:
# result = predict_image('path/to/your/image.jpg')
"""

print("\\n" + "=" * 80)
print("🔮 PREDICTION TEST SCRIPT CREATED!")
print("=" * 80)
print("\\n📋 Cách sử dụng:")
print("\\n1️⃣  Test đơn giản:")
print("   python test_predict.py")
print("\\n2️⃣  Test với ảnh cụ thể:")
print("   python test_predict.py --image path/to/image.jpg")
print("\\n3️⃣  Interactive mode:")
print("   python test_predict.py --interactive")
print("\\n4️⃣  Test tất cả ảnh test:")
print("   python test_predict.py --batch")
print("\\n5️⃣  Với threshold tùy chỉnh:")
print("   python test_predict.py --image image.jpg --threshold 0.3")
print("\\n🎯 Script sẽ:")
print("   ✅ Load model đã train")
print("   ✅ Preprocess ảnh input")
print("   ✅ Predict segmentation mask")
print("   ✅ Hiển thị kết quả với visualization")
print("   ✅ Lưu kết quả ra file")
"""