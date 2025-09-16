import numpy as np
import matplotlib.pyplot as plt

file_path = "test.csv"  # đổi thành tên file CSV của bạn
data = np.loadtxt(file_path,delimiter=",")

# Vẽ heatmap
plt.figure(figsize=(8, 6))
plt.imshow(data, cmap='jet', aspect='auto')
plt.colorbar(label='Giá trị')
plt.title('')
plt.xlabel('')
plt.ylabel('')

# Lưu ảnh
output_image = "anh_nhiet.png"
plt.savefig(output_image, dpi=300, bbox_inches='tight')
plt.close()

print(f"Đã lưu ảnh nhiệt vào file: {output_image}")
