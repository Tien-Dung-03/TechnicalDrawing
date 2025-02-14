import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from functools import partial
import json

# Hàm cosine similarity
def cosine_similarity(tensors):
    # Kiểm tra xem tensors có phải là một cấu trúc lồng nhau hay không
    if isinstance(tensors, (list, tuple)):
        tensors = tuple(tensors)
    x, y = tensors
    x_norm = K.sqrt(K.sum(K.square(x), axis=1, keepdims=True)) + K.epsilon()
    y_norm = K.sqrt(K.sum(K.square(y), axis=1, keepdims=True)) + K.epsilon()
    cos_sim = K.sum(x * y, axis=1, keepdims=True) / (x_norm * y_norm)
    return cos_sim
    
def contrastive_loss(y_true, y_pred):
    margin = 1.0  # Độ lệch margin
    y_true = tf.cast(y_true, tf.float32)
    
    # Giới hạn giá trị của y_pred trong khoảng [0, 1]
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    
    # Công thức tính contrastive loss
    loss = (1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0))
    
    return K.mean(loss)

# Đường dẫn tới mô hình đã lưu
# model_path = os.path.abspath('./model/best_siamese_model_new_test.h5')
model_path = os.path.abspath('../model/best_siamese_model_new_test.h5')
 # Sử dụng partial để truyền đúng dạng đầu vào
cosine_similarity_fn = partial(cosine_similarity)

try:
    siamese_model = load_model(
        model_path,
        custom_objects={'cosine_similarity': cosine_similarity_fn, 'contrastive_loss': contrastive_loss}
    )
    print("Model loaded successfully!")
    # siamese_model.summary()
except Exception as e:
    print("Error loading model:", e)

# file_path = os.path.abspath('./data/normal_embeddings.pkl')
file_path = os.path.abspath('../data/normal_embeddings.pkl')
embeddings = np.load(file_path, allow_pickle=True)

# Đọc dữ liệu từ file JSON
file_path = os.path.abspath('../data/labels.json')
with open(file_path, 'r') as f:
    house_details = json.load(f)
    
# Xác định danh sách các loại phòng và tính năng để tạo các vector có chiều cố định
room_types = ["living_room", "bedroom_1", "bedroom_2", "bedroom_3", "bedroom_4", "bedroom_5", "bedroom_6", "kitchen", "wc_1", "wc_2", "wc_3", "wc_4", "wc_5", "wc_6", "wc_7", "wc_8"]
features_list = ["yard", "main_hall", "side_hall", "laundry_area", "worship_room", "frontyard", "backyard",
                 "bar", "garage", "internal_corridor", "balcon", "lobby", "drying_yard", "library", "common_room",
                 "playground", "business_space", "hallway", "stair_hall", "side_hall", "warehouse", "technical_room",
                "miniatures", "dressing_room", "sports_area", "laundry_room", "relaxation_room", "parking_yard",
                 "skylight", "logia", "planting_yard", "yoga_room", "common_area", "sports_area", "karaoke_room",
                 "massage_room", "multi-purpose_room", "working_room", "sports_room", "rooftop"]

def fix_request(user_request):
    user_request = user_request.replace('và', ',')
    user_request = user_request.replace('để', '')
    user_request = user_request.replace('  ', ' ')
    user_request = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), user_request, flags=re.IGNORECASE)
    user_request = user_request.lower().strip()
    return user_request

def extract_features(user_request):
    user_request = fix_request(user_request)

    # Tách số tầng
    floors = re.search(r'(\d+)\s*tầng', user_request)
    num_floors = int(floors.group(1)) if floors else 1  # Mặc định 1 tầng nếu không tìm thấy

    # Tách diện tích tổng
    area = re.search(r'diện tích\s*(\d+\.?\d*)\s', user_request)
    total_area = float(area.group(1)) if area else None

    # Tìm các loại phòng và số lượng
    rooms = {}
    room_matches = re.findall(r'(\d+)?\s*(phòng ngủ|phòng khách|bếp|wc|nhà vệ sinh)', user_request)
    for count, room_type in room_matches:
        count = int(count) if count else 1  # Nếu không có số lượng, mặc định là 1
        normalized_type = {
            "phòng khách": "living_room",
            "bếp": "kitchen",
            "nhà bếp": "kitchen",
            "phòng ngủ": "bedroom",
            "wc": "wc",
            "nhà vệ sinh": "wc"
        }.get(room_type, room_type)

        if normalized_type in ["bedroom", "wc"]:
            for i in range(1, count + 1):
                rooms[f"{normalized_type}_{i}"] = 1
        else:
            rooms[normalized_type] = rooms.get(normalized_type, 0) + count

    # Thêm mặc định nếu không tìm thấy "wc"
    if not any(k.startswith("wc") for k in rooms.keys()):
        rooms["wc_1"] = 1

    # Tìm các đặc điểm khác
    features = set()

    feature_keywords = {
        "sảnh chính": "main_hall",
        "sảnh phụ": "side_hall",
        "khu vực giặt": "laundry_area",
        "phòng thờ": "worship_room",
        "sân trước": "frontyard", 
        "sân sau": "backyard",
        "sân chơi": "playground",
        "sân phơi": "drying_yard", 
        "sân thượng": "rooftop",
        "bar": "bar",
        "gara": "garage",
        "hành lang nội bộ": "internal_corridor", 
        "ban công":  "balcon",
        "sảnh": "lobby",
        "sân": "yard",
        "thư viện": "library", 
        "logia": "logia",
        "giếng trời": "skylight",
        "massage": "massage_room",
        "karaoke": "karaoke_room",
        "đa năng": "multi-purpose_room",
        "trồng cây": "planting_yard",
        "yoga": "yoga_room",
        "thay đồ": "dressing_room",
        "thể thao": "sports_area",
        "thư giãn": "relaxation_room",
        "kho": "warehouse",
        "chơi": "playground",
        "tiểu cảnh": "miniatures",
        "cầu thang": "stair_hall",
        "khu thờ": "worship_area",
        "phòng sinh hoạt chung": "common_room",
        "không gian kinh doanh": "business_space",
        "hành lang": "hallway",
        "phòng kỹ thuật": "technical_room",
        "phòng giặt": "laundry_room",   
        "đậu xe": "parking_yard",
        "khu sinh hoạt chung": "common_area",
        "phòng thể thao": "sports_room",
        "làm việc": "working_room",
        
    }

    for keyword, feature in feature_keywords.items():
        if keyword in user_request:
            features.add(feature)

    return {
        "floors": num_floors,
        "total_area": total_area,
        "rooms": rooms,
        "features": list(features)
    }

def create_user_embedding(user_request):
    # Trích xuất các đặc trưng từ yêu cầu
    features = extract_features(user_request)

    # Tách thông tin cụ thể từ các đặc trưng
    num_floors = features['floors']  # Số tầng
    total_area = features['total_area'] or 0  # Tổng diện tích, mặc định là 0 nếu không có
    rooms = features['rooms']  # Các phòng và diện tích
    user_features = features['features']  # Các tính năng

    # Mã hóa thông tin về phòng (giả sử room_types đã được định nghĩa)
    room_embedding = [rooms.get(room, 0) for room in room_types]  # Sử dụng danh sách `room_types` làm chuẩn

    # Mã hóa các tính năng bổ sung
    feature_embedding = [1 if feature in user_features else 0 for feature in features_list]  # Sử dụng `features_list`

    # Kết hợp tất cả các thông tin thành một vector embedding
    user_embedding = np.array(
        [num_floors, total_area] + room_embedding + feature_embedding,
        dtype=np.float32
    )

    # Kiểm tra và điều chỉnh kích thước embedding
    target_size = 61  # Kích thước cố định (đảm bảo tương thích với các embedding khác)
    if user_embedding.shape[0] < target_size:
        # Pad nếu thiếu
        user_embedding = np.pad(user_embedding, (0, target_size - user_embedding.shape[0]), mode='constant', constant_values=0)
    elif user_embedding.shape[0] > target_size:
        # Cắt bớt nếu quá
        user_embedding = user_embedding[:target_size]

    return user_embedding

def predict_house(user_request, siamese_model, house_embeddings, house_details):
    user_embedding = create_user_embedding(user_request)  # Đảm bảo hàm này trả về embedding đúng định dạng
    similarities = []

    for house_detail, house_embedding in zip(house_details, house_embeddings):
        house_embedding_array = np.array(house_embedding['embedding'])

        # Dự đoán độ tương đồng
        similarity = siamese_model.predict(
            [np.expand_dims(user_embedding, axis=0), np.expand_dims(house_embedding_array, axis=0)],
            verbose=0
        )[0][0]

        similarities.append({
            'similarity': similarity,
            **house_detail
        })

    ranked_houses = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
    return ranked_houses[:2]

# Ứng dụng chính
st.title("Hệ thống gợi ý nhà thông minh")
st.write("Nhập yêu cầu của bạn để tìm nhà phù hợp.")

# Trạng thái quay lại
if "show_results" not in st.session_state:
    st.session_state.show_results = False
        
user_request = st.text_input("Yêu cầu của bạn:", placeholder="Ví dụ: 2 tầng, 3 phòng ngủ, 1 bếp, diện tích 150m²")
if user_request and siamese_model:
    st.write(f"Đang tìm kiếm nhà phù hợp cho yêu cầu: **{user_request}**")
    ranked_houses = predict_house(user_request, siamese_model, embeddings, house_details)
    if ranked_houses:
        for idx, house in enumerate(ranked_houses):
            st.subheader(f"Ngôi nhà #{idx + 1} (Độ tương đồng: {house['similarity']:.4f})")
            st.write(f"Phong cách: {house['architectural_style']}")
            st.write(f"Số tầng: {house['floor']}, Diện tích: {house['total_area']} m²")
            
            # Tìm tất cả các phòng từ tất cả các house_id liên quan
            related_houses = [h for h in house_details if h['house_id'] == house['house_id']]
            all_rooms = []
            all_features = []

            for related_house in related_houses:
                all_rooms.extend(related_house.get('rooms', []))
                all_features.extend(related_house.get('feature', []))
            
            # Loại bỏ trùng lặp trong danh sách phòng và tiện ích
            unique_rooms = list({(room['type'], room['area']): room for room in all_rooms}.values())
            unique_features = list(set(all_features))

            # Thông tin về các phòng
            room_info = ', '.join([f"{room['type']}: {room['area']} m²" for room in unique_rooms])
            st.write(f"Các phòng: {room_info}")
            
            # Thông tin về các tiện ích
            feature_info = ', '.join(unique_features)
            st.write(f"Các tiện ích: {feature_info}")
            
            # Hiển thị tất cả các ảnh liên quan
            if related_houses:
                for img_idx, related_house in enumerate(related_houses):
                    image_path = related_house.get('image_path', '')
                    if image_path:  # Kiểm tra xem đường dẫn ảnh có tồn tại
                        full_path = os.path.join("../data/images/", os.path.basename(image_path))
                        
                        # Hiển thị ảnh
                        try:
                            with Image.open(full_path) as img:
                                st.image(
                                    img,
                                    caption=f"Ảnh {img_idx + 1} - Phong cách: {related_house['architectural_style']}\nTầng: {related_house['floor']}, Diện tích: {related_house['total_area']} m²",
                                    use_container_width=True
                                )
                        except FileNotFoundError:
                            st.warning(f"Không tìm thấy tệp ảnh: {full_path}")
    else:
        st.warning("Không tìm thấy ngôi nhà phù hợp!")