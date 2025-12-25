/**！
 * @file basics.ino
 * @brief This is an example of the C1001 mmWave Human Detection Sensor detecting the presence of people and their respiration and heart rates.
 *
 * ---------------------------------------------------------------------------------------------------
 *    board   |             MCU                | Leonardo/Mega2560/M0 | ESP32 |  
 *     VCC    |            3.3V/5V             |        VCC           |  VCC  |  
 *     GND    |              GND               |        GND           |  GND  |  
 *     RX     |              TX                |     Serial1 TX1      |  D2   |  
 *     TX     |              RX                |     Serial1 RX1      |  D3   |  
 * ---------------------------------------------------------------------------------------------------
 *
 * @copyright  Copyright (c) 2010 DFRobot Co.Ltd (http://www.dfrobot.com)
 * @license     The MIT License (MIT)gemini3.0とgpt5.1はどちらが高性能ですかgemini3.0とgpt5.1はどちらが高性能ですか
 * @author [tangjie](jie.tang@dfrobot.com)
 * @version  V1.0
 * @date  2024-06-03
 * @url https://github.com/DFRobot/DFRobot_HumanDetection
 */
 
// https://www.circuitschools.com/c1001-mmwave-human-detection-sensor-detects-life-fall-and-sleep/
 
// Technical Specifications at a Glance
// Parameter  Specification
// Operating Voltage  5V
// Operating Current  ≤100mA
// Detection Range  Up to 11 meters
// Frequency Band 61–61.5GHz
// Transmission Power 6dBm
// Radar Detection Angle  100×100 degrees
// Sleep Detection Distance (Chest) 0.4-2.5m
// Breath and Heart Rate Detection Distance (Chest):  0.4-1.5m
// Respiration Rate Range 10–25 breaths/min
// Heart Rate Range 60–100 BPM
// Operating Temperature  -20°C to 60°C
 
//https://www.circuitschools.com/c1001-mmwave-human-detection-sensor-detects-life-fall-and-sleep/
 
#include "DFRobot_HumanDetection.h"
 
#include <WiFi.h>
#include <WiFiClient.h>
 
// WiFi Configuration
const char *ssid = "atsu";
const char *password = "5318725a";
const char *host = "172.20.10.2";//"192.168.xxx.xx"; // Your PC's IP address
const int port = 7007;             // Choose a port number
WiFiClient client;
bool wifiConnected = false;
 
 
DFRobot_HumanDetection hu(&Serial1);
 
// WiFi Functions
void connectToWiFi()
{
    Serial.println("Connecting to WiFi...");
    WiFi.begin(ssid, password);
 
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20)
    {
        delay(500);
        Serial.print(".");
        attempts++;
    }
 
    if (WiFi.status() == WL_CONNECTED)
    {
        wifiConnected = true;
        Serial.println("\nWiFi connected");
        Serial.print("IP address: ");
        Serial.println(WiFi.localIP());
        
        // ★追加: 接続時にNagleアルゴリズムを無効化（遅延対策）
        client.setNoDelay(true);
    }
    else
    {
        Serial.println("\nFailed to connect to WiFi");
    }
}
 
// ★修正: 引数でデータを受け取り、内部でchar配列を使って高速送信する形に変更（fall/dwell を追加）
void sendDataOverWiFi(int presence, int movement, int range, int breath, int heart/*, int falldata, int dwellstatus*/)
{
    if (!wifiConnected)
    {
        connectToWiFi();
        if (!wifiConnected)
            return;
    }

    if (!client.connected())
    {
        if (!client.connect(host, port))
        {
            Serial.println("Connection to host failed");
            wifiConnected = false;
            return;
        }
        // ★追加: 接続時にTCPの遅延を無効化（即時送信）
        client.setNoDelay(true);
    }

    // ★最適化: Stringクラスを使わず、char配列(バッファ)を使って送信
    // メモリ確保・解放のオーバーヘッドがなくなり高速化
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "%d,%d,%d,%d,%d\n", presence, movement, range, breath, heart/*, falldata, dwellstatus*/);
    
    client.print(buffer);
}
 
void setup() {
  Serial.begin(115200);
 
  #if defined(ESP32)
  // Serial1.begin(115200, SERIAL_8N1, /*rx =*/D3, /*tx =*/D2);
  Serial1.begin(115200, SERIAL_8N1, /*rx =*/16, /*tx =*/17);//Initialization successful
 
  #else
  Serial1.begin(115200);
  #endif
 
  Serial.println("Start initialization");
  while (hu.begin() != 0) {
    Serial.println("init error!!!");
    delay(1000);
  }
  Serial.println("Initialization successful");
 
  Serial.println("Start switching work mode");
  while (hu.configWorkMode(hu.eSleepMode) != 0) {//1=hu.eFallingMode,2=hu.eSleepMode
    Serial.println("error!!!");
    delay(1000);
  }
  Serial.print("Work mode switch successful:");Serial.println(hu.getWorkMode());
 
  Serial.print("Current work mode:");
  switch (hu.getWorkMode()) {
    case 1:
      Serial.println("Fall detection mode");
      break;
    case 2:
      Serial.println("Sleep detection mode");
      break;
    default:
      Serial.println("Read error");
  }
 
  hu.configLEDLight(hu.eHPLed, 1);  // Set HP LED switch, it will not light up even if the sensor detects a person when set to 0.
  hu.sensorRet();                   // Module reset, must perform sensorRet after setting data, otherwise the sensor may not be usable
 
  Serial.print("HP LED status:");
  switch (hu.getLEDLightState(hu.eHPLed)) {
    case 0:
      Serial.println("Off");
      break;
    case 1:
      Serial.println("On");
      break;
    default:
      Serial.println("Read error");
  }
 
  hu.configLEDLight(hu.eFALLLed, 1);         // Set HP LED switch, it will not light up even if the sensor detects a person present when set to 0.
  hu.configLEDLight(hu.eHPLed, 1);           // Set FALL LED switch, it will not light up even if the sensor detects a person falling when set to 0.
  hu.dmInstallHeight(270);                   // Set installation height, it needs to be set according to the actual height of the surface from the sensor, unit: CM.
  hu.dmFallTime(5);                          // Set fall time, the sensor needs to delay the current set time after detecting a person falling before outputting the detected fall, this can avoid false triggering, unit: seconds.
  
  // ★修正: 無人判定時間を 1秒 -> 5秒 に変更（すぐに消灯しないようにする）
  hu.dmUnmannedTime(5);                      
  
  hu.dmFallConfig(hu.eResidenceTime, 200);   // Set dwell time, when a person remains still within the sensor detection range for more than the set time, the sensor outputs a stationary dwell status. Unit: seconds.
  hu.dmFallConfig(hu.eFallSensitivityC, 3);  // Set fall sensitivity, range 0~3, the larger the value, the more sensitive.
  hu.sensorRet();                            // Module reset, must perform sensorRet after setting data, otherwise the sensor may not be usable.
 
 
 
  // Connect to WiFi first
  connectToWiFi();
 
  //Serial.println();
}
 
 
void loop() {
  // ★修正: String型の結合処理を廃止し、直接変数に取得
  int presence = hu.smHumanData(hu.eHumanPresence);           // [0] Presence
  int movement = hu.smHumanData(hu.eHumanMovement);           // [1] Movement
  int range = hu.smHumanData(hu.eHumanMovingRange);           // [2] Range
  int breath = hu.getBreatheValue();                          // [3] Breath
  int heart = hu.getHeartRate();                              // [4] Heart
  //int falldata = hu.getFallData(hu.eFallState);               // [5] Fall status
  //int dwellstatus = hu.getFallData(hu.estaticResidencyState); // [6] Dwell status            


  // ★修正: 最適化した送信関数に数値を直接渡す
  sendDataOverWiFi(presence, movement, range, breath, heart /*falldata, dwellstatus*/);
 
  // 通信安定化のため、ごく短い待機時間を入れる
  delay(5);
}





 
 
 
// void loop() {
//   data=String("");
//   // Serial.print("Existing information:");
//   // Serial.print("(Presence(01),Movement(012),MovingRange,getBreatheValue(10-25),getHeartRate(60-100) :=: ");
//   //Serial.print(hu.smHumanData(hu.eHumanPresence));
 
//   //data += String(hu.configWorkMode(hu.eSleepMode));//1=hu.eFallingMode,2=hu.eSleepMode
//   //data += String(hu.getWorkMode());//1=hu.eFallingMode,2=hu.eSleepMode
//   data += String(hu.smHumanData(hu.eHumanPresence));//0="No one is present",1="Someone is present"
 
//   // switch (hu.smHumanData(hu.eHumanPresence)) {
//   //   case 0:
//   //     Serial.println("No one is present");
//   //     break;
//   //   case 1:
//   //     Serial.println("Someone is present");
//   //     break;
//   //   default:
//   //     Serial.println("Read error");
//   // }
 
//   // Serial.print("Motion information:");
//   //Serial.print(",");
//   //Serial.print(hu.smHumanData(hu.eHumanMovement));
//   data += ","+String(hu.smHumanData(hu.eHumanMovement));//0="None",1="Still",2="Active";
 
//   // switch (hu.smHumanData(hu.eHumanMovement)) {
//   //   case 0:
//   //     Serial.println("None");
//   //     break;
//   //   case 1:
//   //     Serial.println("Still");
//   //     break;
//   //   case 2:
//   //     Serial.println("Active");
//   //     break;
//   //   default:
//   //     Serial.println("Read error");
//   // }
 
//   // Serial.print("Body movement parameters: ");
//   // Serial.println(hu.smHumanData(hu.eHumanMovingRange));
//   //Serial.print(",");
//   //Serial.print(hu.smHumanData(hu.eHumanMovingRange));
//   data += ","+String(hu.smHumanData(hu.eHumanMovingRange));
//   //data += ","+String(hu.getFallData(hu.eFallState));            //0="Not fallen",1="fallen"
//   //data += ","+String(hu.getFallData(hu.estaticResidencyState)); //0="No stationary dwell",1="Stationary dwell present"
 
 
//   // Serial.print("Respiration rate: ");
//   // Serial.println(hu.getBreatheValue());
//   //Serial.print(",");
//   //Serial.print(hu.getBreatheValue());
//   data += ","+String(hu.getBreatheValue());
 
//   // Serial.print("Heart rate: ");
//   // Serial.println(hu.getHeartRate());
//   //Serial.print(",");
//   //Serial.println(hu.getHeartRate());
//   data += ","+String(hu.getHeartRate());
//   data += ","+String(hu.smSleepData(hu.eSleepState));
//   data += ","+String(hu.smSleepData(hu.eWakeDuration));
//   data += ","+String(hu.smSleepData(hu.eDeepSleepDuration));
//   data += ","+String(hu.smSleepData(hu.eSleepQuality));
 
//   sSleepComposite comprehensiveState = hu.getSleepComposite();
//   data += ","+String(comprehensiveState.presence);
//                                       // case 0:"No one"
//                                       // case 1:"Someone is present"
 
//   data += ","+String(comprehensiveState.sleepState);
//                                       // case 0: "Deep sleep"
//                                       // case 1: "Light sleep"
//                                       // case 2: "Awake"
//                                       // case 3: "None"
 
 
//   data += ","+String(hu.smSleepData(hu.eSleepDisturbances));
//                                       // case 0:"Sleep duration less than 4 hours"
//                                       // case 1:"Sleep duration more than 12 hours"
//                                       // case 2:"Long time abnormal absence of person"
//                                       // case 3:"None"
 
//   data += ","+String(hu.smSleepData(hu.eSleepQualityRating));
//                                       // case 0:"None"
//                                       // case 1:"Good sleep quality"
//                                       // case 2:"Average sleep quality"
//                                       // case 3:"Poor sleep quality"
 
//   data += ","+String(hu.smSleepData(hu.eAbnormalStruggle));
//                                       // 0="None"
//                                       // 1="Normal status"
//                                       // 2="Abnormal struggle status"
 
//     //data += "\n";
//   Serial.println(data);
 
//   // // Send data over WiFi if all anchors have valid data
//   // if (allAnchorsHaveValidData())
//   // {
//       sendDataOverWiFi();
//       //client.print(data);
 
//   // }
 
//   delay(100);
// }