#!/usr/bin/env python3
"""
Script để chạy Agoda crawler (thông thường hoặc Selenium) dựa vào cấu hình .env
"""

import os
from dotenv import load_dotenv
load_dotenv()
import sys
import logging
import time
import io
from datetime import datetime
import traceback

# Đảm bảo thư mục hiện tại có trong PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Cấu hình logging với UTF-8 encoding để hỗ trợ emoji trên Windows
os.makedirs('logs', exist_ok=True)

# Tạo UTF-8 StreamHandler để hỗ trợ emoji
class Utf8StreamHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__(stream=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8'))

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/agoda_crawler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
        Utf8StreamHandler()
    ]
)

def main():
    
    debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    if debug_mode:
        import debugpy
        logging.info("🐛 Khởi động chế độ debug với debugpy trên port 5678")
        debugpy.listen(("0.0.0.0", 5678))
        logging.info("🔍 Đang đợi kết nối debug từ VS Code...")
        debugpy.wait_for_client()
        logging.info("✅ Đã kết nối với VS Code debugger")
        
    # ⏳ Chờ database khởi động hoàn toàn
    logging.info("🔄 Chờ kết nối database...")
    time.sleep(5)
    
    try:
        # Xác định loại crawler cần sử dụng
        use_selenium = os.getenv('USE_SELENIUM', 'false').lower() == 'true'
        
        # Log các biến môi trường (ẩn mật khẩu)
        logging.info("📊 Cấu hình hiện tại:")
        logging.info(f"  → DB_HOST: {os.getenv('DB_HOST')}")
        logging.info(f"  → DB_NAME: {os.getenv('DB_NAME')}")
        logging.info(f"  → HEADLESS_MODE: {os.getenv('HEADLESS_MODE')}")
        logging.info(f"  → CRAWL_DELAY: {os.getenv('CRAWL_DELAY')}")
        logging.info(f"  → MAX_HOTELS: {os.getenv('MAX_HOTELS')}")
        logging.info(f"  → MAX_REVIEWS_PER_HOTEL: {os.getenv('MAX_REVIEWS_PER_HOTEL', '15')}")
        logging.info(f"  → USE_SELENIUM: {use_selenium}")
        
        # Khởi chạy crawler phù hợp dựa vào cấu hình
        if use_selenium:
            # Sử dụng Selenium crawler
            logging.info("🤖 Sử dụng Selenium crawler")
            from crawlers.agoda_selenium_crawler import AgodaSeleniumCrawler
            
            # Khởi chạy crawler
            logging.info("🚀 Bắt đầu thu thập dữ liệu từ Agoda.com sử dụng Selenium")
            crawler = AgodaSeleniumCrawler()
            crawler.scrape_all_hotels()
        else:
            # Sử dụng crawler thông thường
            logging.info("🤖 Sử dụng crawler thông thường")
            from crawlers.agoda_crawler import AgodaCrawler
            
            # Khởi chạy crawler
            logging.info("🚀 Bắt đầu thu thập dữ liệu từ Agoda.com")
            crawler = AgodaCrawler()
            crawler.craw_data_agoda()
        
        logging.info("✅ Thu thập dữ liệu hoàn tất!")
        
    except Exception as e:
        logging.error(f"❌ Lỗi: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())