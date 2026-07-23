const int LED_PIN = 13;

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() <= 0) {
    return;
  }

  String command = Serial.readStringUntil('\n');
  command.trim();

  if (command == "ON") {
    digitalWrite(LED_PIN, HIGH);
  } else if (command == "OFF") {
    digitalWrite(LED_PIN, LOW);
  }
}
