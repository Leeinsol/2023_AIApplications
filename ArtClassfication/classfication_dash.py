# import libraries
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as Datasets
import time
import matplotlib.pyplot as plt
import torch, gc
import os
import PIL.Image as Image
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import base64
from collections import OrderedDict

# GPU / CPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 가비지 컬렉터 호출: 메모리 정리
gc.collect()

# 디렉토리 설정
save_dir = '../data'
os.makedirs(save_dir, exist_ok=True)


# VGG 네트워크 레이어 함수: VGG 네트워크 구축
def get_vgg_layers(config, batch_norm):  # VGG 아키텍처 지정 리스트, 배치 정규화 사용 여부
    layers = []
    in_channels = 3  # RGB

    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':  # Maxpooling 레이어를 생성
            layers += [nn.MaxPool2d(kernel_size=2)]
        else:  # Conv2d 레이어 생성
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            if batch_norm:  # 배치 정규화 레이어 생성
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c

    # Sequential 모델 반환
    return nn.Sequential(*layers)


# VGG 모델 정의 함수
class VGG(nn.Module):
    def __init__(self, features, output_dim):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(7)  # 입력 차원 줄이기
        # classifier 생성
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim)
        )

    # forward pass 수행
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)  # 텐서 크기 변경
        x = self.classifier(h)
        return x, h

# 정확도 계산 함수
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


# 학습 함수
def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred, _ = model(x)

        # loss, 정확도 계산
        loss = criterion(y_pred, y)
        acc = calculate_accuracy(y_pred, y)

        # 그래디언트 계산, 가중치 업데이트
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# 모델 펼가 함수
def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            # loss, 정확도 계산
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# loss값 시각화 함수
def plot_loss(train_losses, valid_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# 모델 학습 함수
def train_model(model, train_iterator, valid_iterator, optimizer, criterion, device, epochs):
    train_losses = []
    valid_losses = []

    best_valid_loss = float('inf')  # 가장 낮은 loss값 저장

    for epoch in range(epochs):  # epochs 수 만큼 반복
        start = time.monotonic()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)  # 모델 학습 -> 학습 loss, 학습 정확도 반환
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)  # 모델 검증 -> 검증 loss, 검증 정확도 반환

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < best_valid_loss:  # 현재 loss가 최적 loss 보다 작을 때 모델 저장
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'VGG-model.pt'))

        end = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start, end)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc * 100:.2f}%')

    # loss 그래프 출력
    plot_loss(train_losses, valid_losses)


# epoch 소요 시간 계산
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# 이미지 분류 작업 함수
def classify_image(image_path, model, classes, device):
    image = Image.open(image_path).convert('RGB')  # RGB로 채널 수 맞추기
    # 이미지 리사이즈, 텐서로 변환, 정규화 실행
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0).to(device)
    model = model.to(device)

    with torch.no_grad():
        output, _ = model(image)
        probabilities = F.softmax(output, dim=1)

        # 일치 확률을 계산
        matching_percentages = (probabilities * 100).squeeze().tolist()

    # 클래스 레이블과 일치 확률을 딕서너리에 저장
    class_percentages = {class_label: matching_percentage for class_label, matching_percentage in
                         zip(classes, matching_percentages)}

    outputs, _ = model(image)
    _, predicted = torch.max(outputs, 1)
    predicted_label = classes[predicted.item()]

    return predicted_label, class_percentages


# 업로드된 이미지 처리 함수
def process_image(image_content):
    # 이미지 데이터를 base64 디코딩
    image_decoded = base64.b64decode(image_content.split(",")[1])

    temp_image_path = os.path.join(save_dir, "temp_image.jpg")
    with open(temp_image_path, "wb") as f:
        f.write(image_decoded)

    return temp_image_path


# Dash 앱 생성
app = dash.Dash(__name__)

# 레이아웃 설정
app.layout = html.Div(
    style={
        "textAlign": "center",
        "maxWidth": "600px",
        "margin": "auto",
        "padding": "20px",
        "background-color": "#F9F9F9",  # Custom background color
    },
    children=[
        html.H1("Art Classification: 예술 작품 기법 분류기", style={"marginBottom": "20px"}),
        html.H5(
            "예술 작품은 각기 다른 기법과 스타일로 제작됩니다. 예술 작품 기법 분류기는 기법을 자동으로 분류해주는 인공지능 분류기입니다.",
            style={"marginBottom": "40px"},
        ),
        dcc.Upload(  # 이미지 업로드 컴포넌트 생성
            id="upload-image",
            children=html.Div(["Drag and drop or click to select an image"]),
            style={
                "width": "100%",
                "height": "200px",
                "lineHeight": "200px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
                "color": "#666666",
                "font-size": "18px",
            },
            multiple=False,
        ),
        html.Div(id="output-image"),
        html.Div(id="output-label", style={"marginTop": "20px", "color": "#333333"}),  # Custom text color
    ],
)


# 콜백 함수 정의
@app.callback(
    [Output("output-image", "children"),
     Output("output-label", "children")],
    [Input("upload-image", "contents")],
    [State("upload-image", "filename"),
     State("upload-image", "last_modified")]
)
def classify_uploaded_image(image_content, image_filename, image_last_modified):
    if image_content is not None:
        # 임시 파일 경로 얻음
        image_path = process_image(image_content)

        # 훈련된 모델 로드(VGG16)
        vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        num_classes = 5
        vgg_features = get_vgg_layers(vgg_config, batch_norm=True)
        model = VGG(vgg_features, num_classes).to(device)

        try:
            model.load_state_dict(torch.load(os.path.join(save_dir, "VGG-model.pt"), map_location=device), strict=False)
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')

        model.eval()

        # 이미지 분류 -> 예측 라벨과 클래스 확률 얻음
        classes = ['drawings', 'engraving', 'iconography', 'painting', 'sculpture']
        predicted_label, class_percentages = classify_image(image_path, model, classes, device)

        # 임시 이미지 파일 삭제
        os.remove(image_path)

        # 화면에 표시
        image_html = html.Img(src=image_content, style={"width": "50%"})

        label_html = html.H3(f"이 작품은 {predicted_label}을 사용한 작품입니다. ")

        percentages_html = []
        for class_label, percentage in class_percentages.items():
            percentages_html.append(html.P(f"{class_label}: {percentage:.2f}%"))

        return image_html, [label_html] + percentages_html

    return [None, None]

# Dash 앱 실행
if __name__ == "__main__":

    num_classes = 5
    batch_size = 16
    learning_rate = 0.001
    epochs = 50

    # 모델
    vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    vgg_features = get_vgg_layers(vgg_config, batch_norm=True)
    model = VGG(vgg_features, num_classes).to(device)

    # 옵티마이저: Adam
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_valid = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 데이터 셋 로드
    train_dataset = torchvision.datasets.ImageFolder(root="./data/art", transform=transform_train)
    valid_dataset = torchvision.datasets.ImageFolder(root="./data/art_test", transform=transform_valid)

    train_iterator = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_iterator = data.DataLoader(valid_dataset, batch_size=batch_size)

    # 실행 여부 설정
    run_training = False
    run_testing = False
    run_dash_app =True

    if run_training:
        # 모델 훈련
        train_model(model, train_iterator, valid_iterator, optimizer, criterion, device, epochs)

    if run_testing:
        # 모델 테스트
        folder_path = "./data/art_image"
        image_file = [file for file in os.listdir(folder_path) if file.endswith(".jpg")][0]
        image_path = os.path.join(folder_path, image_file)
        vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M']
        num_classes = 5
        vgg_features = get_vgg_layers(vgg_config, batch_norm=True)
        model = VGG(vgg_features, num_classes).to(device)

        try:
            model.load_state_dict(torch.load(os.path.join(save_dir, "VGG-model.pt"), map_location=device), strict=False)
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')

        model.eval()
        classes = ['drawings', 'engraving', 'iconography', 'painting', 'sculpture']

        predicted_label = classify_image(image_path, model, classes, device)

        print("Predicted label:", predicted_label)

    if run_dash_app:
        # Dash 앱 실행
        app.run_server(debug=True)