#include <Adafruit_NeoPixel.h>
#include <Servo.h>

// 왼손 스트립은 D5(5), 오른손 스트립은 D6(6)에 연결합니다.
const int LEFT_NEOPIXEL_PIN = 5;
const int RIGHT_NEOPIXEL_PIN = 6;
const int NEOPIXEL_COUNT = 8;
const int LED_BRIGHTNESS = 50;

Adafruit_NeoPixel leftPixels(NEOPIXEL_COUNT, LEFT_NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel rightPixels(NEOPIXEL_COUNT, RIGHT_NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);

// 서보모터 객체 생성 (MG995)
Servo servo3;
Servo servo4;
Servo servo9;
Servo servo10;
Servo servo11;

void clearPixels() {
  leftPixels.clear();
  rightPixels.clear();
}

void showLeftCount(int count) {
  int safeCount = constrain(count, 0, NEOPIXEL_COUNT);
  for (int i = 0; i < safeCount; i++) {
    leftPixels.setPixelColor(i, leftPixels.Color(0, 0, 255));
  }
  
  // 5번 LED(왼손) 개수에 부합하는 서보모터 동작 제어
  // 초기값 90도, 180도로 전환
  servo3.write(safeCount >= 1 ? 180 : 90);
  servo4.write(safeCount >= 2 ? 180 : 90);
  servo9.write(safeCount >= 3 ? 180 : 90);
  servo10.write(safeCount >= 4 ? 180 : 90);
  servo11.write(safeCount >= 5 ? 180 : 90);
}

void showRightCount(int count) {
  int safeCount = constrain(count, 0, NEOPIXEL_COUNT);
  for (int i = 0; i < safeCount; i++) {
    rightPixels.setPixelColor(i, rightPixels.Color(255, 0, 0));
  }
}

void showCommand(String command) {
  clearPixels();

  if (command == "ON" || command == "L") {
    showLeftCount(1);
    showRightCount(0);
  } else if (command == "R") {
    showRightCount(1);
    showLeftCount(0); // 왼손 및 서보 리셋
  } else if (command == "LR") {
    showLeftCount(1);
    showRightCount(1);
  } else if (command.length() >= 4 && command.charAt(0) == 'L') {
    int rightMarker = command.indexOf('R', 1);
    if (rightMarker > 1) {
      int leftCount = command.substring(1, rightMarker).toInt();
      int rightCount = command.substring(rightMarker + 1).toInt();
      showLeftCount(leftCount);
      showRightCount(rightCount);
    }
  } else if (command.length() >= 2 && (command.charAt(0) == 'L' || command.charAt(0) == 'R')) {
    int ledCount = command.substring(1).toInt();
    if (command.charAt(0) == 'L') {
      showLeftCount(ledCount);
    } else {
      showRightCount(ledCount);
      showLeftCount(0); // 왼손 및 서보 리셋
    }
  } else {
    // OFF 또는 기타 명령 시 모든 LED 꺼짐 및 서보모터 90도 초기화
    showLeftCount(0);
    showRightCount(0);
  }

  leftPixels.show();
  rightPixels.show();
}

void setup() {
  Serial.begin(9600);
  
  // NeoPixel 초기화
  leftPixels.begin();
  rightPixels.begin();
  leftPixels.setBrightness(LED_BRIGHTNESS);
  rightPixels.setBrightness(LED_BRIGHTNESS);
  clearPixels();
  leftPixels.show();
  rightPixels.show();
  
  // 서보모터 핀 연결 및 초기각도(90도) 세팅
  servo3.attach(3);
  servo4.attach(4);
  servo9.attach(9);
  servo10.attach(10);
  servo11.attach(11);
  
  servo3.write(90);
  servo4.write(90);
  servo9.write(90);
  servo10.write(90);
  servo11.write(90);
}

void loop() {
  if (Serial.available() <= 0) {
    return;
  }

  String command = Serial.readStringUntil('\n');
  command.trim();
  showCommand(command);
}
