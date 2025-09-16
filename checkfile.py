import os
import csv

def remove_empty_csv(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)

            if os.path.getsize(file_path) == 0:
                print(f"[DELETE] {file_path} (empty file)")
                os.remove(file_path)
                continue

            with open(file_path, "r") as f:
                reader = csv.reader(f)
                rows = [row for row in reader if any(cell.strip() for cell in row)]

            if len(rows) == 0:
                print(f"[DELETE] {file_path} (only blank lines)")
                os.remove(file_path)
            else:
                print(f"[KEEP]   {file_path} ({len(rows)} rows)")

folder = "C:/Users/admin/Documents/GPBL/diffusers/Data_Dipole_Model"
remove_empty_csv(folder)
