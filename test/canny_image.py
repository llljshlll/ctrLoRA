import cv2
import matplotlib.pyplot as plt

# 1. 이미지 읽기 및 리사이징
image = cv2.imread('images/_1Gn_xkw7sa_i9GU4mkxxQ.jpg')  # 이미지 경로
image = cv2.resize(image, (512, 512))  # 512x512로 리사이징
cv2.imwrite('images/_1Gn_xkw7sa_i9GU4mkxxQ.jpg', image)

# 2. 그레이스케일 + Gaussian Blur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 블러로 디테일 감소

# 3. Canny Edge Detection (엣지 간소화)
# ▶ threshold를 높여서 복잡한 엣지 제거
edges = cv2.Canny(blurred, threshold1=100, threshold2=200)

# 4. 결과 저장
cv2.imwrite('images/_1Gn_xkw7sa_i9GU4mkxxQ_canny__.jpg', edges)

# 5. 출력
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original (Resized)')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Simplified Canny Edges')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
