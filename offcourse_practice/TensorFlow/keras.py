import tensorflow as tf

# MNIST 데이터 세트를 가져옵니다.
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 뉴럴 네트워크 모델을 생성합니다.
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')])

# 학습을 위한 optimizer와 loss function을 선택합니다.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 학습 데이터 세트를 사용하여 모델을 학습시킵니다. 5번 반복합니다.
model.fit(x_train, y_train, epochs=5)

# 테스트 데이터 세트를 사용하여 모델을 평가합니다.
model.evaluate(x_test, y_test)
