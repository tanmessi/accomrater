-- ================================
-- 0️⃣ Xóa dữ liệu cũ để tránh lỗi trùng lặp
-- ================================
TRUNCATE TABLE processed_reviews RESTART IDENTITY CASCADE;
TRUNCATE TABLE reviews RESTART IDENTITY CASCADE;
TRUNCATE TABLE hotels RESTART IDENTITY CASCADE;

-- ================================
-- 1️⃣ Bổ sung cột hotel_name và name cho bảng hotels (nếu thiếu)
-- ================================
ALTER TABLE hotels ADD COLUMN IF NOT EXISTS hotel_name TEXT;
ALTER TABLE hotels ADD COLUMN IF NOT EXISTS name TEXT;

-- ================================
-- 2️⃣ Insert dữ liệu vào bảng hotels
-- ================================
INSERT INTO hotels (id, hotel_name, name) VALUES
(1, 'Khách sạn Sài Gòn', 'Khách sạn Sài Gòn'),
(2, 'Khách sạn Hà Nội', 'Khách sạn Hà Nội'),
(3, 'Khách sạn Đà Nẵng', 'Khách sạn Đà Nẵng'),
(4, 'Khách sạn Nha Trang', 'Khách sạn Nha Trang'),
(5, 'Khách sạn Huế', 'Khách sạn Huế'),
(6, 'Khách sạn Phú Quốc', 'Khách sạn Phú Quốc'),
(7, 'Khách sạn Vũng Tàu', 'Khách sạn Vũng Tàu'),
(8, 'Khách sạn Hạ Long', 'Khách sạn Hạ Long'),
(9, 'Khách sạn Sapa', 'Khách sạn Sapa'),
(10, 'Khách sạn Đà Lạt', 'Khách sạn Đà Lạt'),
(11, 'Khách sạn Bà Nà Hills', 'Khách sạn Bà Nà Hills'),
(12, 'Khách sạn Côn Đảo', 'Khách sạn Côn Đảo'),
(13, 'Khách sạn Bình Thuận', 'Khách sạn Bình Thuận'),
(14, 'Khách sạn Lý Sơn', 'Khách sạn Lý Sơn'),
(15, 'Khách sạn Mộc Châu', 'Khách sạn Mộc Châu')
ON CONFLICT DO NOTHING;

-- ================================
-- 3️⃣ Insert dữ liệu vào bảng reviews (2000 records)
-- ================================
DO $$
DECLARE 
    i INT;
    hotel_id INT;
    rating NUMERIC(2,1);
    comment TEXT;
    review_date DATE;
    source TEXT;
    helpful_votes INT;
    reviewer_hash TEXT;
BEGIN
    FOR i IN 1..2000 LOOP
        -- Chọn hotel_id ngẫu nhiên
        SELECT id INTO hotel_id FROM hotels ORDER BY random() LIMIT 1;

        -- Tạo rating ngẫu nhiên
        rating := round(CAST(random() * 4 + 1 AS NUMERIC), 1);

        -- Tạo nội dung đánh giá ngẫu nhiên
        SELECT comment_text INTO comment
        FROM (
            VALUES 
                ('Phòng rộng rãi, sạch sẽ, nhân viên tận tâm, giá hợp lý.'),
                ('Khách sạn tuyệt vời, vị trí đẹp, phục vụ chuyên nghiệp.'),
                ('Dịch vụ hoàn hảo, giá cả phù hợp, đồ ăn rất ngon.'),
                ('Phòng có view biển đẹp, wifi nhanh, hồ bơi trong xanh.'),
                ('Nhân viên thân thiện, phòng sạch mới, đồ ăn phong phú.'),
                ('Phòng ẩm thấp, giường cũ, giá cao, không đáng tiền.'),
                ('Wifi yếu, nhân viên phục vụ chậm, đồ ăn kém chất lượng.'),
                ('Giá cả hợp lý, nhưng cách âm kém, hơi ồn ào vào ban đêm.'),
                ('Nhà vệ sinh không sạch, dịch vụ cần cải thiện.'),
                ('Gần trung tâm, dễ di chuyển nhưng bữa sáng không ngon.'),
                ('Khung cảnh thiên nhiên đẹp, không khí trong lành, thích hợp nghỉ dưỡng.'),
                ('Nằm ngay cạnh bờ biển, có dịch vụ lặn san hô rất tuyệt vời.'),
                ('Dịch vụ tốt nhưng hơi xa trung tâm thành phố.'),
                ('Đồ ăn hải sản tươi ngon, view núi đẹp, không gian yên tĩnh.'),
                ('Khách sạn có khu spa tuyệt vời, rất thư giãn.')
        ) AS temp_table(comment_text)
        ORDER BY random()
        LIMIT 1;

        -- Tạo ngày đánh giá ngẫu nhiên
        review_date := CURRENT_DATE - (random() * 30)::INT;

        -- Chọn nguồn đánh giá ngẫu nhiên
        SELECT src INTO source
        FROM (
            VALUES ('Booking.com'), ('Agoda'), ('TripAdvisor'), ('Google Reviews')
        ) AS temp_table(src)
        ORDER BY random()
        LIMIT 1;

        -- Số lượt đánh giá hữu ích ngẫu nhiên
        helpful_votes := (random() * 100)::INT;

        -- Hash người đánh giá để bảo mật danh tính
        reviewer_hash := md5(hotel_id::TEXT || random()::TEXT);

        -- Chèn vào bảng reviews
        INSERT INTO reviews (hotel_id, rating, comment, review_date, source, helpful_votes, reviewer_hash)
        VALUES (hotel_id, rating, comment, review_date, source, helpful_votes, reviewer_hash);
    END LOOP;
END $$;

-- ================================
-- 4️⃣ Insert dữ liệu vào bảng processed_reviews (2000 records)
-- ================================
DO $$
DECLARE 
    i INT;
    review_id INT;
    cleaned TEXT;
    normalized TEXT;
    segmented TEXT;
    final TEXT;
    timestamps TIMESTAMP;
BEGIN
    FOR i IN 1..2000 LOOP
        -- Chọn review_id ngẫu nhiên từ bảng reviews
        SELECT id INTO review_id FROM reviews ORDER BY random() LIMIT 1;

        -- Chọn nội dung đã xử lý ngẫu nhiên
        SELECT cleaned_text INTO cleaned
        FROM (
            VALUES 
                ('phong rong rai sach se nhan vien tan tam gia hop ly'),
                ('khach san tuyet voi vi tri dep phuc vu chuyen nghiep'),
                ('dich vu hoan hao gia ca phu hop do an rat ngon'),
                ('phong co view bien dep wifi nhanh ho boi trong xanh'),
                ('nhan vien than thien phong sach moi do an phong phu'),
                ('phong am thap giuong cu gia cao khong dang tien'),
                ('wifi yeu, nhan vien phuc vu cham, do an kem chat luong'),
                ('khach san co tien nghi tot nhung xa trung tam'),
                ('phong dep nhung am thanh on ao, kho ngu'),
                ('nhan vien tan tinh nhung qua dong khach nen cham'),
                ('khung canh thien nhien dep khong khi trong lanh thich hop nghi duong'),
                ('nam ngay canh bo bien co dich vu lan san ho rat tuyet voi'),
                ('dich vu tot nhung hoi xa trung tam thanh pho'),
                ('do an hai san tuoi ngon view nui dep khong gian yen tinh'),
                ('khach san co khu spa tuyet voi rat thu gian')
        ) AS temp_table(cleaned_text)
        ORDER BY random()
        LIMIT 1;

        -- Chuẩn hóa dữ liệu
        normalized := REPLACE(cleaned, 'phong', 'phòng');
        normalized := REPLACE(normalized, 'nhan vien', 'nhân viên');
        normalized := REPLACE(normalized, 'dich vu', 'dịch vụ');
        normalized := REPLACE(normalized, 'gia', 'giá');
        normalized := REPLACE(normalized, 'sach se', 'sạch sẽ');
        normalized := REPLACE(normalized, 'view', 'cảnh đẹp');

        -- Phân đoạn từ
        segmented := REPLACE(normalized, ' ', '_');

        -- Văn bản cuối cùng
        final := segmented;

        -- Ngày xử lý ngẫu nhiên trong vòng 30 ngày qua
        timestamps := NOW() - INTERVAL '30 days' * random();

        -- Chèn vào bảng processed_reviews
        INSERT INTO processed_reviews (review_id, cleaned_text, normalized_text, segmented_text, final_text, processed_at, version)
        VALUES (review_id, cleaned, normalized, segmented, final, timestamps, '1.0');
    END LOOP;
END $$; 