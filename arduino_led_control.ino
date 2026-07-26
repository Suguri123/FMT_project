#include <Adafruit_NeoPixel.h>

// 왼손 스트립은 D5, 오른손 스트립은 D6에 연결합니다.
const int LEFT_NEOPIXEL_PIN = 5;
const int RIGHT_NEOPIXEL_PIN = 6;
const int NEOPIXEL_COUNT = 8;
const int LED_BRIGHTNESS = 50;

Adafruit_NeoPixel leftPixels(NEOPIXEL_COUNT, LEFT_NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel rightPixels(NEOPIXEL_COUNT, RIGHT_NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);

void clearPixels() {
  leftPixels.clear();
  rightPixels.clear();
}

void showLeftCount(int count) {
  int safeCount = constrain(count, 0, NEOPIXEL_COUNT);
  for (int i = 0; i < safeCount; i++) {
    leftPixels.setPixelColor(i, leftPixels.Color(0, 0, 255));
  }
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
  } else if (command == "R") {
    showRightCount(1);
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
    }
  }

  leftPixels.show();
  rightPixels.show();
}

void setup() {
  Serial.begin(9600);
  leftPixels.begin();
  rightPixels.begin();
  leftPixels.setBrightness(LED_BRIGHTNESS);
  rightPixels.setBrightness(LED_BRIGHTNESS);
  clearPixels();
  leftPixels.show();
  rightPixels.show();
}

void loop() {
  if (Serial.available() <= 0) {
    return;
  }

  String command = Serial.readStringUntil('\n');
  command.trim();
  showCommand(command);
}
