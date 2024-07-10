# Công cụ Hỗ trợ Chẩn đoán Bệnh Alzheimer từ Hình ảnh MRI

## Giới thiệu

Dự án này nhằm mục đích xây dựng một công cụ hỗ trợ chẩn đoán bệnh Alzheimer từ hình ảnh MRI sử dụng các thuật toán trí tuệ nhân tạo (AI). Công cụ này sẽ giúp các bác sĩ và chuyên gia y tế phân tích hình ảnh MRI não bộ một cách nhanh chóng và chính xác hơn, từ đó đưa ra chẩn đoán sớm và chính xác hơn về bệnh Alzheimer.

## Tiến độ dự án

### Giai đoạn 1: Khám phá và Tiền xử lý Dữ liệu (Hoàn thành)

- Tải và khám phá bộ dữ liệu "Alzheimer MRI Disease Classification Dataset" từ Kaggle.
- Thực hiện tiền xử lý dữ liệu:
  - Chuyển đổi ảnh sang mảng NumPy.
  - Thay đổi kích thước ảnh về 224x224 pixel.
  - Chuẩn hóa giá trị pixel.

### Giai đoạn 2: Xây dựng và Huấn luyện Mô hình (Đang thực hiện)

- Nghiên cứu và lựa chọn mô hình học sâu phù hợp (ví dụ: CNNs, ResNet, EfficientNet).
- Thực hiện transfer learning từ các mô hình đã được huấn luyện trước.
- Thiết kế và huấn luyện mô hình trên dữ liệu đã tiền xử lý.
- Đánh giá hiệu suất mô hình trên tập dữ liệu kiểm tra.

### Giai đoạn 3: Triển khai và Đánh giá Thực tế (Chưa thực hiện)

- Phát triển giao diện người dùng cho công cụ.
- Triển khai và kiểm thử công cụ tại các cơ sở y tế.
- Thu thập phản hồi và đánh giá hiệu quả thực tế của công cụ.

## Công nghệ sử dụng

- Ngôn ngữ lập trình: Python
- Thư viện học sâu: PyTorch hoặc TensorFlow/Keras
- Thư viện xử lý ảnh: OpenCV, Pillow
- Thư viện khác: NumPy, Matplotlib, Hugging Face Datasets

## Dữ liệu

Bộ dữ liệu được sử dụng trong dự án này là:

@dataset{alzheimer_mri_dataset,
author = {Falah.G.Salieh},
title = {Alzheimer MRI Dataset},
year = {2023},
publisher = {Hugging Face},
version = {1.0},
url = {https://huggingface.co/datasets/Falah/Alzheimer_MRI}
}

## Liên hệ

Nếu bạn có bất kỳ câu hỏi hoặc góp ý nào, vui lòng liên hệ với chúng tôi qua email: caoan339@gmail.com.
