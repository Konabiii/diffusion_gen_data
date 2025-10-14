import gdown

# Đường dẫn file Google Drive mới (đã chuyển sang dạng tải trực tiếp)
url = "https://drive.google.com/uc?id=1w8D2AcGo_R2DcJWnHchwHEbq9vSFmrsU"
output = "Data.zip"  # Đặt tên file tải về, thêm .zip nếu file là nén

# Tải file
gdown.download(url, output, quiet=False)
print(f"Đã tải xong: {output}")
