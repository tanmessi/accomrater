# 📚 Hướng dẫn cài đặt và triển khai AccomRater Crawler

## 📑 Mục lục
- [Cài đặt Docker](#cài-đặt-docker)
- [Thiết lập dự án](#thiết-lập-dự-án)
- [Build và Run container](#build-và-run-container)
- [Theo dõi quá trình Crawl với VNC Viewer](#theo-dõi-quá-trình-crawl-với-vnc-viewer)
- [Xử lý lỗi thường gặp](#xử-lý-lỗi-thường-gặp)

## 🐳 Cài đặt Docker

### Windows 10
1. **Kiểm tra yêu cầu hệ thống**:
   - Windows 10 64-bit: Pro, Enterprise, hoặc Education (Build 17134 hoặc mới hơn)
   - Bật tính năng Hyper-V và Containers Windows
   - Ít nhất 4GB RAM

2. **Bật WSL 2**:
   - Mở PowerShell với quyền Administrator
   - Chạy lệnh: `dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart`
   - Chạy lệnh: `dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart`
   - Khởi động lại máy tính
   - Tải [WSL2 Linux kernel update package](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi)
   - Cài đặt package và đặt WSL 2 làm mặc định: `wsl --set-default-version 2`

3. **Cài đặt Docker Desktop**:
   - Tải [Docker Desktop Installer](https://desktop.docker.com/win/stable/Docker%20Desktop%20Installer.exe)
   - Chạy installer và làm theo hướng dẫn
   - Chọn "Use WSL 2 instead of Hyper-V" khi được hỏi
   - Hoàn tất cài đặt và khởi động lại máy tính
   - Khởi động Docker Desktop từ Start menu

4. **Kiểm tra cài đặt**:
   - Mở Command Prompt hoặc PowerShell
   - Chạy lệnh: `docker --version`
   - Chạy lệnh: `docker run hello-world`

### macOS
1. **Kiểm tra yêu cầu hệ thống**:
   - macOS 10.14 (Mojave) trở lên
   - Ít nhất 4GB RAM

2. **Cài đặt Docker Desktop**:
   - Tải [Docker Desktop for Mac](https://desktop.docker.com/mac/stable/Docker.dmg)
   - Mở file .dmg và kéo Docker vào thư mục Applications
   - Mở Docker từ thư mục Applications
   - Đăng nhập nếu được yêu cầu và hoàn tất thiết lập

3. **Kiểm tra cài đặt**:
   - Mở Terminal
   - Chạy lệnh: `docker --version`
   - Chạy lệnh: `docker run hello-world`

## 🛠️ Thiết lập dự án

1. **Clone repository**:
   ```bash
   git clone https://github.com/yourusername/accomrater.git
   cd accomrater
   ```

2. **Thiết lập file môi trường**:
   - Tạo file `.env` dựa trên mẫu `.env.docker`:
   ```bash
   cp .env.docker .env
   ```
   
   - Chỉnh sửa file `.env` với thông tin phù hợp:
   ```
   # 💾 Database config
   DB_NAME=accom_rater
   DB_USER=postgres
   DB_PASSWORD=your_secure_password
   DB_HOST=postgres
   DB_PORT=5432

   # 🕸️ Crawler config
   HEADLESS_MODE=true
   CRAWL_DELAY=3
   MAX_HOTELS=5
   MAX_REVIEWS_PER_HOTEL=15
   USE_SELENIUM=true
   ```

## 🚀 Build và Run container

### Chạy toàn bộ hệ thống
```bash
docker-compose up -d
```

Lệnh này sẽ khởi tạo và chạy tất cả các container:
- PostgreSQL database
- Selenium Hub
- Chrome node cho Booking crawler
- Chrome node cho Agoda crawler
- Booking crawler
- Agoda crawler
- Streamlit app

### Chạy từng phần riêng biệt

1. **Khởi chạy database**:
   ```bash
   docker-compose up -d postgres
   ```

2. **Khởi chạy Selenium Hub và Chrome node**:
   ```bash
   docker-compose up -d selenium-hub chrome-booking chrome-agoda
   ```

3. **Chạy crawler Booking**:
   ```bash
   docker-compose up booking-crawler
   ```

4. **Chạy crawler Agoda**:
   ```bash
   docker-compose up agoda-crawler
   ```

5. **Chạy ứng dụng Streamlit**:
   ```bash
   docker-compose up app
   ```

### Theo dõi logs

```bash
# Xem logs của tất cả container
docker-compose logs -f

# Xem logs của container cụ thể
docker-compose logs -f booking-crawler
docker-compose logs -f agoda-crawler
```

### Dừng và xóa container

```bash
# Dừng tất cả container nhưng giữ dữ liệu
docker-compose down

# Dừng và xóa tất cả container kèm dữ liệu
docker-compose down -v
```

## 👁️ Theo dõi quá trình Crawl với VNC Viewer

### Cài đặt RealVNC Viewer

#### Windows
1. Tải [RealVNC Viewer](https://www.realvnc.com/download/file/viewer.files/VNC-Viewer-6.21.1109-Windows.exe)
2. Chạy file .exe đã tải
3. Làm theo hướng dẫn cài đặt

#### macOS
1. Tải [RealVNC Viewer](https://www.realvnc.com/download/file/viewer.files/VNC-Viewer-6.21.1109-MacOSX.dmg)
2. Mở file .dmg và kéo VNC Viewer vào thư mục Applications
3. Mở VNC Viewer từ thư mục Applications

### Kết nối đến Chrome node

1. **Lấy địa chỉ IP máy host**:
   - Windows: Mở Command Prompt và chạy `ipconfig`
   - macOS: Mở Terminal và chạy `ifconfig`

2. **Kết nối đến Chrome node cho Booking crawler**:
   - Mở RealVNC Viewer
   - Nhập địa chỉ: `localhost:5901` hoặc `your_ip_address:5901`
   - Nhấn Connect
   - Không cần mật khẩu (chế độ mặc định)

3. **Kết nối đến Chrome node cho Agoda crawler**:
   - Mở RealVNC Viewer
   - Nhập địa chỉ: `localhost:5902` hoặc `your_ip_address:5902`
   - Nhấn Connect
   - Không cần mật khẩu (chế độ mặc định)

4. **Xem quá trình crawl**:
   - Sau khi kết nối, bạn sẽ thấy màn hình của Chrome đang chạy trong container
   - Có thể theo dõi các thao tác của crawler trên trình duyệt

## ⚠️ Xử lý lỗi thường gặp

### 1. Lỗi kết nối đến database
```
ERROR: Database connection failed: could not connect to server: Connection refused
```

**Giải pháp**:
- Kiểm tra container PostgreSQL: `docker-compose ps postgres`
- Đảm bảo container đang chạy: `docker-compose up -d postgres`
- Kiểm tra log: `docker-compose logs postgres`
- Xác nhận thông tin kết nối trong file `.env`

### 2. Lỗi kết nối đến Selenium Hub
```
ERROR: Connection to http://selenium-hub:4444/wd/hub failed
```

**Giải pháp**:
- Kiểm tra container Selenium Hub: `docker-compose ps selenium-hub`
- Đảm bảo container đang chạy: `docker-compose up -d selenium-hub`
- Kiểm tra log: `docker-compose logs selenium-hub`
- Đảm bảo biến môi trường `SELENIUM_HUB_URL` được cấu hình đúng

### 3. Lỗi không thể xem Chrome node qua VNC
```
Unable to connect to host on port 5901
```

**Giải pháp**:
- Đảm bảo port 5901/5902 đã được forward đúng trong docker-compose.yml
- Kiểm tra firewall: tạm thời tắt firewall hoặc mở port 5901/5902
- Kiểm tra container Chrome node: `docker-compose ps chrome-booking chrome-agoda`
- Khởi động lại container: `docker-compose restart chrome-booking chrome-agoda`

### 4. Lỗi không tìm thấy Chrome driver
```
ERROR: SessionNotCreatedException: Message: session not created: This version of ChromeDriver only supports Chrome version XX
```

**Giải pháp**:
- Kiểm tra version Chrome và ChromeDriver trong Dockerfile
- Cập nhật ChromeDriver theo version Chrome trong container
- Sửa URL ChromeDriver trong Dockerfile và build lại image

## 🔗 Tài liệu tham khảo
- [Docker Documentation](https://docs.docker.com/)
- [Selenium Documentation](https://www.selenium.dev/documentation/en/)
- [RealVNC Documentation](https://www.realvnc.com/en/connect/docs/user/viewer.html)