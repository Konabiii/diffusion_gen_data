import os
import shutil

root_folder = r"C:/Users/admin/Documents/GPBL/diffusers/Data_Dipole_Model_Normalized"   # đổi lại đúng đường dẫn

prefixes = ["ellipse", "step_r", "step_t", "rectangular", "triangular"]


for prefix in prefixes:
    dest_folder = os.path.join(root_folder, prefix)
    os.makedirs(dest_folder, exist_ok=True)


for filename in os.listdir(root_folder):
    file_path = os.path.join(root_folder, filename)

    if os.path.isfile(file_path) and filename.lower().endswith(".csv"):  
        matched = False
        for prefix in prefixes:
            if filename.lower().startswith(prefix): 
                dest_path = os.path.join(root_folder, prefix, filename)
                shutil.copy2(file_path, dest_path)  
                print(f"✅ Copied {filename} -> {prefix}/")
                matched = True
                break
        if not matched:
            
