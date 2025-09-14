import os
import requests
import zipfile
from tqdm import tqdm

def download_kvasir_seg():
    """Download Kvasir-SEG polyp segmentation dataset (easy to download)"""
    url = "https://datasets.simula.no/downloads/kvasir-seg.zip"
    download_dir = "data/raw/"
    zip_path = os.path.join(download_dir, "kvasir-seg.zip")

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(download_dir, exist_ok=True)

    try:
        # Lấy kích thước tệp từ header
        response = requests.head(url)
        total_size = int(response.headers.get("content-length", 0))

        # Tải tệp
        print("Downloading Kvasir-SEG dataset...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Kiểm tra lỗi HTTP

        with open(zip_path, "wb") as f:
            with tqdm(total=total_size, unit="iB", unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        size = f.write(chunk)
                        pbar.update(size)

        # Giải nén
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(download_dir)

        # Xóa tệp zip sau khi giải nén thành công
        os.remove(zip_path)
        print("✅ Kvasir-SEG dataset ready in data/raw/")

    except requests.exceptions.RequestException as e:
        print(f"❌ Lỗi khi tải dữ liệu: {e}")
    except zipfile.BadZipFile:
        print("❌ Lỗi khi giải nén: Tệp zip bị hỏng hoặc không hợp lệ.")
    except Exception as e:
        print(f"❌ Lỗi không xác định: {e}")

if __name__ == "__main__":
    download_kvasir_seg()