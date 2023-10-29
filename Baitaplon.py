
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#  Đây là đường dẫn được tạo từ máy nhóm bọn em. Nếu thầy không chạy được code thì đổi lại đường dẫn ạ . Em xin 
# lỗi vì sự bất tiện này ạ !!!
TRAIN_PATH = "D:\BTL-Nhom8\Train.xlsx"
TEST_PATH = "D:\BTL-Nhom8\Test.xlsx"
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Tải dữ liệu
df = pd.read_excel(TRAIN_PATH)


# Định nghĩa hàm để sao chép khung dữ liệu
def copy(df):
    return df.copy()


# Loại bỏ 1 cột từ Dataframe
def drop(df, col):
    new_df = df.drop(col, axis=1)
    return new_df


# định nghĩa một hàm để loại bỏ các hàng từ một DataFrame dựa trên điều kiện
def drop_row(df, col, find):
    new_df = df[df[col] != find]
    return new_df


def get_dummies(df, col):
    return pd.get_dummies(df, columns=[col])


def replace(df, col):
    replace_list = [",", " minutes"]
    new_df = df.copy()
    for rep in replace_list:
        new_df[col] = new_df[col].str.replace(rep, "")
    return new_df


def to_numeric(df, col):
    new_df = df.copy()
    new_df[col] = pd.to_numeric(new_df[col])
    return new_df


def train_pipeline(df):
    return (df.pipe(replace, "Delivery_Time")
            .pipe(to_numeric, "Delivery_Time"))


# định nghĩa một hàm có tên pipeline để tạo một chuỗi xử lý dữ liệu cho một DataFrame,
# xử lý dữ liệu trên DataFrame làm sạch dữ liệu, biến đổi cột, loại bỏ cột không cần thiết
def pipeline(df):
    return (df.pipe(copy)
            .pipe(get_dummies, "Location")
            .pipe(to_numeric, "Distance")
            .pipe(drop_row, "Cost", "for")  #  drop_row
            .pipe(to_numeric, "Cost"))


train = pipeline(df)
train = train_pipeline(train)
train

train_data = drop(train, "Delivery_Time")
train_data

train_data_numpy = train_data.to_numpy()
train_data_numpy

train_label = train["Delivery_Time"]
train_label

# Tạo và huấn luyện mô hình Hồi quy Tuyến tính
linear_reg = LinearRegression()
linear_reg.fit(train_data_numpy, train_label)

# Đánh giá mô hình Hồi quy Tuyến tính bằng cách sử dụng kỹ thuật cross-validation
linear_reg_scores = cross_val_score(linear_reg, train_data_numpy, train_label, scoring="neg_mean_squared_error", cv=10)
linear_reg_rmse_scores = np.sqrt(-linear_reg_scores)


# Hiển thị các điểm số đánh giá của mô hình
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


#  Hiển thị điểm số đánh giá cho mô hình hồi quy tuyến tính sau khi đã huấn luyện
display_scores(linear_reg_rmse_scores)

# Tải dữ liệu Test data
test_df = pd.read_excel(TEST_PATH)

# Quá trình tiền xử lý dữ liệu cho dữ liệu kiểm tra (test data)
dataset = pd.concat([df, test_df], axis=0)
dataset = pipeline(dataset)
test = dataset[len(train):]
test = test.drop("Delivery_Time", axis=1)

# Dự đoán kết quả cho tập dữ liệu Test bằng hồi quy tuyến tính
prediction = linear_reg.predict(test)

# Tạo 1 bảng dữ liệu
submission = pd.DataFrame({"Delivery_Time": prediction})
submission.to_csv('result.csv', index=False)