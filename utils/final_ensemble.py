def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = {}
    for line in lines:
        parts = line.strip().split(' ')
        img_name = parts[0]
        predict = parts[1]
        data[img_name] = predict
    return data

def vote(files):
    data_list = [read_file(filename) for filename in files]
    
    # Kết quả cuối cùng
    final_data = {}
    
    for img_name in data_list[0].keys():
        # Đếm số lần xuất hiện của mỗi giá trị predict
        counter = {}
        
        for data in data_list:
            if img_name in data:
                predict = data[img_name]
                counter[predict] = counter.get(predict, 0) + 1
        
        # Lấy predict có số lần xuất hiện nhiều nhất hoặc predict từ file 3
        most_common_predict = max(counter, key=lambda k: (counter[k], k == data_list[2].get(img_name)))
        
        final_data[img_name] = most_common_predict
        
    return final_data

files = [r'Last_CKPT\results\test.txt', r'Last_CKPT\results\ensemble_crop.txt', r'Last_CKPT\results\conner_crop.txt']
result = vote(files)

# Xuất kết quả ra file prediction.txt
with open(r'Last_CKPT\results\prediction_91.txt', 'w', encoding='utf-8') as f:
    for key, value in result.items():
        f.write(f"{key} {value}\n")