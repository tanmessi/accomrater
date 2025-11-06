#!/usr/bin/env python3
"""
Script để chạy crawler trong môi trường Docker
"""

import os
from dotenv import load_dotenv
load_dotenv()
import sys
import logging
import time
from datetime import datetime
import traceback

# Đảm bảo thư mục hiện tại có trong PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Cấu hình logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/booking_crawler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
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
        logging.info("📊 Cấu hình hiện tại:")
        logging.info(f"  → DB_HOST: {os.getenv('DB_HOST')}")
        logging.info(f"  → DB_NAME: {os.getenv('DB_NAME')}")
        logging.info(f"  → HEADLESS_MODE: {os.getenv('HEADLESS_MODE')}")
        logging.info(f"  → CRAWL_DELAY: {os.getenv('CRAWL_DELAY')}")
        logging.info(f"  → MAX_HOTELS: {os.getenv('MAX_HOTELS')}")
        
        # 🔍 Import crawler (import ở đây để đảm bảo đã load dotenv)
        from crawlers.booking_crawler import BookingCrawler
        
        # 🕸️ Khởi chạy crawler
        logging.info("🚀 Bắt đầu thu thập dữ liệu từ Booking.com")
        crawler = BookingCrawler()
        crawler.scrape_all_hotels()
        logging.info("✅ Thu thập dữ liệu hoàn tất!")
        
    except Exception as e:
        logging.error(f"❌ Lỗi: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())