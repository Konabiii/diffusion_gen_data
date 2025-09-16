import gdown

url = f"https://drive.google.com/uc?id=1lxhWkQIZsHkz_QGVtBidErMSWQk7LWzO"
output = "Data_Dipole_Model.zip"
gdown.download(url, output, quiet=False)
print(f"Đã tải xong: {output}")
