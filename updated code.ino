/* ===========================================================================
   TinyGuardian - CLOUD NODE
   ---------------------------------------------------------------------------
   Fall detection on an ESP32 using a TinyML model that runs fully on-device,
   with live telemetry and fall alerts pushed to Supabase over HTTPS.

   ARCHITECTURE
     Core 1 (loop)     : I2C sampling at 50 Hz, feature extraction, inference.
     Core 0 (netTask)  : WiFi, HTTPS, Supabase REST calls.
     The two cores share one snapshot struct (mutex) and one event queue,
     so a slow network request can never stall the sensor pipeline.

   WHAT GOES TO SUPABASE
     devices      : upsert on boot and every 15 s (heartbeat, RSSI, IP)
     telemetry    : one row every 2 s (status, class probabilities, features)
     fall_events  : one row the instant a fall passes the confidence gate

   WIRING
     MPU6050  VCC -> 3V3
              GND -> GND
              SDA -> GPIO21
              SCL -> GPIO22
     LED   -> GPIO2  (on-board LED on most ESP32 boards)
     Buzzer-> GPIO4  (optional active buzzer, set USE_BUZZER to 0 to disable)

   BOARD
     Arduino IDE -> ESP32 Dev Module. No extra libraries needed.
   =========================================================================== */

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <math.h>

// ===========================================================================
//  USER CONFIGURATION
// ===========================================================================
const char *WIFI_SSID = "Momo";
const char *WIFI_PASS = "momo1234";

const char *SUPABASE_URL = "https://zrwapsqqptlodfgnjzjs.supabase.co";
const char *SUPABASE_KEY =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
  "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpyd2Fwc3FxcHRsb2RmZ25qempzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODg0MjAwNzAsImV4cCI6MjEwMzk5NjA3MH0."
  "yNnhD6bSSWplH7IxOha2haafQYwTfCbXnZLsfOb2fCc";

const char *DEVICE_ID   = "TG-01";              // primary key in devices table
const char *DEVICE_NAME = "Guardian Node 1";
const char *DEVICE_ROOM = "Living Room";
const char *FIRMWARE    = "1.2.0";

const uint32_t TELEMETRY_MS = 2000;             // live row every 2 s
const uint32_t HEARTBEAT_MS = 15000;            // device row every 15 s

#define USE_BUZZER  1
#define BUZZER_PIN  4

// ------------------------------ configuration ------------------------------
#define SDA_PIN     21
#define SCL_PIN     22
#define MPU_ADDR    0x68        // use 0x69 if AD0 is wired to 3V3
#define LED_PIN     2

const float    FALL_THRESHOLD = 0.80f;   // minimum confidence to call it a fall
const uint32_t COOLDOWN_MS    = 4000;    // ignore new falls for this long
const uint32_t ALARM_MS       = 6000;    // LED + buzzer alarm length

// ------------------------------ model shape --------------------------------
// NOTE: every model identifier carries a TG_ prefix on purpose. The ESP32 core
// header binary.h already defines B0, B1, B10, B11 ... as binary literals, so
// plain names like B1 or W1 will not compile.
#define TG_NFEAT  13
#define TG_H1      16
#define TG_H2      12
#define TG_NCLS   4

const char *CLASS_NAMES[TG_NCLS] = {"IDLE", "NORMAL", "FALL", "ABNORMAL"};
enum { C_IDLE = 0, C_NORMAL = 1, C_FALL = 2, C_ABNORMAL = 3 };

// ------------------------------ sampling -----------------------------------
const uint16_t TG_WIN        = 50;              // 1 second window at 50 Hz
const uint16_t TG_HOP        = 25;              // new decision every 0.5 s
const uint32_t SAMPLE_US  = 20000;           // 50 Hz
const float ACC_LSB_PER_G   = 4096.0f;       // +-8 g range
const float GYR_LSB_PER_DPS = 65.5f;         // +-500 dps range

// ===========================================================================
//  TRAINED MODEL  -  13 -> 16 (ReLU) -> 12 (ReLU) -> 4 (Softmax), 480 params
//  Test accuracy 99.56 %. Regenerate these numbers with 2_train_model.py
//  after you collect real data.
// ===========================================================================
const float TG_MEAN[TG_NFEAT] = {
1.293223f, 0.398331f, 0.653445f, 2.643163f, 1.989720f, 2.211688f, 1.950605f, 87.137115f,
  63.299999f, 292.057587f, 1.353140f, 43.793449f, 0.257413f,
};

const float TG_STD[TG_NFEAT] = {
0.517114f, 0.336437f, 0.269573f, 1.531435f, 1.706104f, 2.101613f, 0.804381f, 78.848221f,
  57.381790f, 255.880112f, 1.293310f, 46.182785f, 0.329793f,
};

const float TG_W1[TG_NFEAT][TG_H1] = {
{-0.093805f, 0.172126f, -0.115108f, 0.206858f, -0.009537f, -0.228359f, 0.472542f, 0.378951f, -0.187742f, -0.199060f, 0.069021f, -0.466505f, -0.263366f, 0.080224f, -0.130135f, -0.032087f},
  {0.541513f, 0.388234f, 0.213434f, 0.356170f, 0.061361f, -0.478807f, -0.525931f, 0.047007f, 0.332829f, -0.060934f, -0.475410f, -0.506818f, 0.019922f, -0.276733f, 0.080201f, 0.141012f},
  {-0.324216f, 0.095176f, 0.338961f, -0.027489f, -0.085704f, 0.668704f, 0.613647f, -0.115897f, 0.129720f, -0.675692f, 0.675279f, -0.101555f, -0.342076f, 0.001922f, -0.126207f, -0.599474f},
  {0.314226f, -0.431159f, 0.425853f, -0.266979f, 0.164364f, -0.080087f, -0.041125f, 0.072653f, -0.074058f, 0.204299f, -0.331206f, -0.348744f, 0.000902f, 0.395785f, -0.054212f, 0.174075f},
  {0.343236f, 0.157558f, 0.356682f, -0.213635f, 0.484925f, -0.376182f, 0.109944f, -0.098401f, -0.282072f, 0.398041f, -0.119325f, -0.225806f, 0.098075f, -0.101315f, -0.110795f, 0.167431f},
  {-0.694394f, 0.489037f, -0.329163f, -0.238705f, -0.467089f, 0.171777f, 0.298703f, -0.439311f, -0.319229f, -0.641049f, -0.257686f, -0.261135f, -0.030387f, 0.023788f, 0.117024f, 0.244810f},
  {-0.443854f, 0.514705f, 0.272759f, 0.540201f, -0.197153f, 0.024667f, -0.322254f, 0.186526f, -0.229118f, -0.271500f, 0.271296f, -0.040344f, -0.109108f, -0.077235f, 0.410934f, 0.418493f},
  {0.245964f, 0.280995f, -0.444567f, -0.114486f, -0.304466f, -0.397458f, -0.233269f, -0.158777f, -0.086373f, -0.072578f, -0.741149f, 0.005599f, 0.748829f, -0.107841f, 0.537987f, 0.615699f},
  {-0.294388f, 0.355265f, 0.048372f, -0.497781f, 0.676864f, -0.546168f, 0.033510f, -0.343820f, -0.061073f, 0.503811f, -0.344507f, -0.017600f, 0.354795f, -0.529492f, 0.157674f, 0.280642f},
  {0.019737f, -0.244923f, 0.773326f, 0.009945f, 0.384187f, 0.361025f, 0.252912f, -0.140175f, -0.275451f, -0.153852f, 0.055470f, 0.534901f, -0.020014f, -0.437709f, -0.290204f, 0.289969f},
  {-0.474978f, -0.280632f, 0.134318f, -0.006565f, -0.118435f, 0.451064f, 0.058746f, -0.012536f, -0.064236f, 0.191052f, 0.184168f, -0.504591f, 0.233840f, 0.035932f, -0.400926f, -0.130940f},
  {0.002956f, -0.080078f, -0.291823f, 0.596320f, 0.241167f, 0.066611f, -0.576644f, -0.560420f, -0.564340f, 0.236686f, -0.230420f, 0.497523f, 0.090308f, -0.569170f, 0.143501f, -0.050690f},
  {-0.098064f, -0.100431f, -0.536525f, -0.127012f, 0.246084f, -0.152412f, -0.537701f, -0.133987f, 0.422585f, -0.384939f, -0.734234f, -0.215825f, 0.320676f, -0.231623f, 0.376960f, 0.180570f},
};
const float TG_B1[TG_H1] = {
0.535735f, 0.344837f, 0.412130f, 0.490103f, 0.173712f, -0.044878f, -0.533214f, 0.286023f,
  0.255344f, 0.197291f, -0.381509f, 0.342728f, 0.211398f, 0.134750f, 0.005192f, 0.378755f,
};

const float TG_W2[TG_H1][TG_H2] = {
{0.301174f, 0.140662f, 0.147091f, -1.156988f, 0.805218f, 0.668403f, 0.301754f, -0.118450f, -0.011572f, 0.740402f, -1.155687f, 0.582900f},
  {0.780163f, 0.679849f, -0.384447f, 0.588819f, 0.351626f, -0.338232f, -0.294785f, -0.341300f, 0.326611f, -0.649737f, 0.462173f, -0.427692f},
  {-0.455685f, 0.535657f, 0.432701f, 0.135370f, 0.728464f, 0.373101f, -0.233031f, -0.156518f, 0.267901f, 0.597222f, -0.444961f, 0.778356f},
  {0.326065f, 0.561187f, 0.097506f, 0.136162f, 0.806675f, -0.568607f, -0.543115f, 0.316430f, 0.727703f, 0.817512f, -0.463250f, 0.581937f},
  {-0.395864f, 0.267283f, 0.437196f, -0.219823f, 0.261719f, 0.705357f, 0.713296f, -0.352250f, -0.089384f, 0.055073f, 0.162309f, 0.215569f},
  {-0.357117f, -0.435784f, 0.280488f, 0.418164f, 0.142906f, -0.390742f, -0.487719f, -0.167295f, 0.691366f, 0.203832f, 0.633220f, 0.314775f},
  {-0.066837f, -0.838512f, -0.110621f, 0.849047f, -0.214194f, -0.348315f, -0.073602f, 0.012005f, 0.195612f, -0.552006f, 0.951399f, -0.589282f},
  {0.367634f, 0.408597f, 0.415726f, 0.427975f, -0.232262f, -0.400166f, -0.210050f, 0.386235f, 0.009958f, 0.709519f, -0.311677f, 0.026155f},
  {-0.307981f, 0.627254f, 0.507040f, -0.333658f, 0.032608f, 0.055753f, -0.237973f, 0.134681f, 0.399087f, 0.288214f, 0.056207f, -0.005152f},
  {-0.499320f, -0.703124f, -0.189010f, 0.194885f, -0.063098f, 0.290691f, 0.759561f, -0.294863f, 0.297176f, -0.338075f, -0.022006f, 0.377236f},
  {-0.336900f, -0.192002f, 0.544007f, 0.023551f, -0.319499f, -0.260664f, 0.345826f, -0.097706f, 0.262191f, 0.258311f, 0.346696f, -0.284684f},
  {0.019639f, 0.236012f, 0.235012f, -0.257272f, -0.236993f, 0.766616f, -0.114318f, -0.428791f, -0.171082f, -0.168060f, -0.030116f, 0.874123f},
  {0.236480f, 0.139999f, 0.139151f, -0.008012f, -0.028243f, -0.177720f, 0.688487f, 0.261646f, -0.433647f, -0.492659f, 0.366486f, -0.367098f},
  {0.382353f, 0.025770f, 0.621384f, 0.195559f, 0.333239f, -0.532062f, 0.099288f, -0.418530f, 0.162353f, 0.375579f, -0.053206f, -0.508330f},
  {0.203581f, 0.198293f, -0.129553f, 0.721001f, 0.242874f, -0.055203f, -0.037883f, 0.159862f, 0.318838f, -0.651602f, 0.140971f, -0.057399f},
  {0.668305f, 0.537064f, 0.105843f, 0.838521f, 0.280376f, -0.065686f, 0.630082f, -0.248138f, -0.272864f, -0.610290f, 0.542486f, -0.206120f},
};
const float TG_B2[TG_H2] = {
0.239933f, 0.358102f, -0.147722f, -0.345650f, 0.476505f, 0.109441f, 0.178145f, -0.055589f,
  0.047189f, 0.446250f, -0.300514f, 0.426285f,
};

const float TG_W3[TG_H2][TG_NCLS] = {
{-0.203861f, -0.381397f, -0.832567f, 0.450719f},
  {-1.148622f, 0.234912f, -0.663332f, 0.611566f},
  {0.611095f, 0.629134f, 0.444020f, -0.556581f},
  {0.711177f, -0.961761f, -0.262627f, 0.591876f},
  {-1.291736f, 0.252179f, -0.376725f, 0.321081f},
  {-0.159225f, -0.153444f, 0.714527f, -0.913240f},
  {-0.595955f, -0.998633f, 0.892445f, 0.248089f},
  {-0.146332f, -0.201705f, 0.436574f, -0.568534f},
  {0.629297f, 0.478778f, -0.793517f, -0.129803f},
  {0.142550f, 0.783846f, -0.348880f, -0.296214f},
  {1.059768f, -1.033600f, 0.027410f, 0.783426f},
  {-1.239740f, 0.693499f, 0.473651f, -0.704995f},
};
const float TG_B3[TG_NCLS] = {
-0.372442f, 0.318045f, -0.087801f, 0.148805f,
};

// ------------------------------ runtime state ------------------------------
float bufAx[TG_WIN], bufAy[TG_WIN], bufAz[TG_WIN];
float bufGx[TG_WIN], bufGy[TG_WIN], bufGz[TG_WIN];
uint16_t head = 0, filled = 0, sinceInfer = 0;

float gyroBias[3] = {0, 0, 0};
uint32_t lastFallMs = 0;
uint32_t fallCount  = 0;
uint32_t alarmUntil = 0;

// ------------------------- shared between the cores ------------------------
struct Snapshot {
  int      cls;
  float    prob[TG_NCLS];
  float    feat[TG_NFEAT];
  uint32_t inferUs;
  bool     valid;
};

struct FallEvent {
  float confidence;
  float impactG;
  float freefallG;
  float rotationDps;
  float tiltDeg;
  uint32_t index;
};

Snapshot          gSnap = {0};
SemaphoreHandle_t gSnapLock = NULL;
QueueHandle_t     gFallQueue = NULL;
volatile bool     gCloudOk = false;
volatile uint32_t gPosted  = 0;
volatile uint32_t gFailed  = 0;

// ===========================================================================
//  MPU6050  (raw register access, no sensor library needed)
// ===========================================================================
void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg); Wire.write(val);
  Wire.endTransmission();
}

bool readMotion(float *a, float *g) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);                                   // ACCEL_XOUT_H
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14) != 14) return false;

  int16_t r[7];
  for (int i = 0; i < 7; i++) {
    uint8_t hi = Wire.read(), lo = Wire.read();
    r[i] = (int16_t)((hi << 8) | lo);
  }
  a[0] = r[0] / ACC_LSB_PER_G;
  a[1] = r[1] / ACC_LSB_PER_G;
  a[2] = r[2] / ACC_LSB_PER_G;
  g[0] = r[4] / GYR_LSB_PER_DPS - gyroBias[0];
  g[1] = r[5] / GYR_LSB_PER_DPS - gyroBias[1];
  g[2] = r[6] / GYR_LSB_PER_DPS - gyroBias[2];
  return true;
}

bool mpuInit() {
  mpuWrite(0x6B, 0x80); delay(100);                   // reset
  mpuWrite(0x6B, 0x01); delay(50);                    // wake up

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x75);                                   // WHO_AM_I
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)1);
  uint8_t who = Wire.available() ? Wire.read() : 0xFF;
  if (who == 0xFF) return false;

  mpuWrite(0x1A, 0x03);      // low pass filter 44 Hz
  mpuWrite(0x19, 19);        // sample rate 50 Hz
  mpuWrite(0x1B, 0x08);      // gyro  +-500 dps
  mpuWrite(0x1C, 0x10);      // accel +-8 g
  delay(50);

  Serial.print(F("MPU6050 found, WHO_AM_I = 0x"));
  Serial.println(who, HEX);
  return true;
}

void calibrateGyro() {
  Serial.println(F("Calibrating - keep the board STILL for 2 seconds..."));
  double s[3] = {0, 0, 0};
  float a[3], g[3];
  int n = 0;
  gyroBias[0] = gyroBias[1] = gyroBias[2] = 0;
  for (int i = 0; i < 400; i++) {
    if (readMotion(a, g)) { s[0] += g[0]; s[1] += g[1]; s[2] += g[2]; n++; }
    delay(4);
  }
  if (n) { gyroBias[0] = s[0]/n; gyroBias[1] = s[1]/n; gyroBias[2] = s[2]/n; }
  Serial.println(F("Calibration done."));
}

// ===========================================================================
//  FEATURE EXTRACTION  -  13 numbers that describe one second of motion
// ===========================================================================
void extractFeatures(float *f) {
  float aMag[TG_WIN], gMag[TG_WIN];
  float sumA = 0, sumA2 = 0, sumSma = 0, sumG = 0, sumG2 = 0;
  float aMin = 1e9f, aMax = -1e9f, gMax = 0;

  for (uint16_t k = 0; k < TG_WIN; k++) {
    uint16_t i = (head + k) % TG_WIN;                    // oldest sample first
    float ax = bufAx[i], ay = bufAy[i], az = bufAz[i];
    float gx = bufGx[i], gy = bufGy[i], gz = bufGz[i];

    float am = sqrtf(ax*ax + ay*ay + az*az);
    float gm = sqrtf(gx*gx + gy*gy + gz*gz);
    aMag[k] = am; gMag[k] = gm;

    sumA += am;  sumA2 += am*am;  sumG += gm;  sumG2 += gm*gm;
    sumSma += fabsf(ax) + fabsf(ay) + fabsf(az);
    if (am < aMin) aMin = am;
    if (am > aMax) aMax = am;
    if (gm > gMax) gMax = gm;
  }

  float accMean = sumA / TG_WIN;
  float accVar  = sumA2 / TG_WIN - accMean * accMean;   if (accVar < 0) accVar = 0;
  float gyrMean = sumG / TG_WIN;
  float gyrVar  = sumG2 / TG_WIN - gyrMean * gyrMean;   if (gyrVar < 0) gyrVar = 0;

  float jerkMax = 0;                                  // sharpest change in |a|
  for (uint16_t k = 1; k < TG_WIN; k++) {
    float d = fabsf(aMag[k] - aMag[k-1]);
    if (d > jerkMax) jerkMax = d;
  }

  // how much the board rotated between the start and the end of the window
  float v1[3] = {0,0,0}, v2[3] = {0,0,0};
  for (uint16_t k = 0; k < 5; k++) {
    uint16_t i = (head + k) % TG_WIN;
    v1[0] += bufAx[i]; v1[1] += bufAy[i]; v1[2] += bufAz[i];
    uint16_t j = (head + (TG_WIN - 5) + k) % TG_WIN;
    v2[0] += bufAx[j]; v2[1] += bufAy[j]; v2[2] += bufAz[j];
  }
  for (int d = 0; d < 3; d++) { v1[d] /= 5.0f; v2[d] /= 5.0f; }
  float n1 = sqrtf(v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2]);
  float n2 = sqrtf(v2[0]*v2[0]+v2[1]*v2[1]+v2[2]*v2[2]);
  float c  = (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]) / (n1*n2 + 1e-9f);
  if (c >  1.0f) c =  1.0f;
  if (c < -1.0f) c = -1.0f;
  float tilt = acosf(c) * 57.2957795f;

  // how still it went in the last 0.3 s - the giveaway after a real fall
  float sp = 0, sp2 = 0;
  for (uint16_t k = TG_WIN - 15; k < TG_WIN; k++) { sp += aMag[k]; sp2 += aMag[k]*aMag[k]; }
  float pm = sp / 15.0f;
  float pv = sp2 / 15.0f - pm * pm;                  if (pv < 0) pv = 0;

  f[0]  = accMean;          // average acceleration magnitude
  f[1]  = sqrtf(accVar);    // how much it varied
  f[2]  = aMin;             // lowest point - near zero during free fall
  f[3]  = aMax;             // impact peak
  f[4]  = aMax - aMin;
  f[5]  = sumA2 / TG_WIN;      // energy
  f[6]  = sumSma / TG_WIN;     // signal magnitude area
  f[7]  = gyrMean;
  f[8]  = sqrtf(gyrVar);
  f[9]  = gMax;             // peak rotation rate
  f[10] = jerkMax;
  f[11] = tilt;             // orientation change in degrees
  f[12] = sqrtf(pv);        // stillness after the event
}

// ===========================================================================
//  NEURAL NETWORK FORWARD PASS
// ===========================================================================
void runModel(const float *raw, float *probs) {
  float x[TG_NFEAT], h1[TG_H1], h2[TG_H2], o[TG_NCLS];

  for (int i = 0; i < TG_NFEAT; i++)
    x[i] = (raw[i] - TG_MEAN[i]) / TG_STD[i];           // normalise

  for (int j = 0; j < TG_H1; j++) {
    float s = TG_B1[j];
    for (int i = 0; i < TG_NFEAT; i++) s += x[i] * TG_W1[i][j];
    h1[j] = s > 0 ? s : 0;                            // ReLU
  }
  for (int j = 0; j < TG_H2; j++) {
    float s = TG_B2[j];
    for (int i = 0; i < TG_H1; i++) s += h1[i] * TG_W2[i][j];
    h2[j] = s > 0 ? s : 0;                            // ReLU
  }
  for (int j = 0; j < TG_NCLS; j++) {
    float s = TG_B3[j];
    for (int i = 0; i < TG_H2; i++) s += h2[i] * TG_W3[i][j];
    o[j] = s;
  }

  float mx = o[0];                                     // softmax
  for (int j = 1; j < TG_NCLS; j++) if (o[j] > mx) mx = o[j];
  float sum = 0;
  for (int j = 0; j < TG_NCLS; j++) { o[j] = expf(o[j] - mx); sum += o[j]; }
  for (int j = 0; j < TG_NCLS; j++) probs[j] = o[j] / sum;
}

// ===========================================================================
//  SUPABASE REST CLIENT  (runs on core 0 only)
// ===========================================================================
WiFiClientSecure netClient;

bool supabasePost(const char *path, const char *body, const char *prefer) {
  if (WiFi.status() != WL_CONNECTED) return false;

  HTTPClient http;
  String url = String(SUPABASE_URL) + path;
  if (!http.begin(netClient, url)) return false;

  http.setTimeout(8000);
  http.setReuse(true);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("apikey", SUPABASE_KEY);
  http.addHeader("Authorization", String("Bearer ") + SUPABASE_KEY);
  http.addHeader("Prefer", prefer);

  int code = http.POST((uint8_t *)body, strlen(body));
  if (code < 200 || code >= 300) {
    Serial.printf("[cloud] POST %s -> HTTP %d %s\n", path, code,
                  http.errorToString(code).c_str());
    String resp = http.getString();
    if (resp.length()) Serial.println("[cloud] " + resp);
    http.end();
    gFailed++;
    gCloudOk = false;
    return false;
  }
  http.end();
  gPosted++;
  gCloudOk = true;
  return true;
}

void upsertDevice() {
  char body[512];
  snprintf(body, sizeof(body),
           // last_seen is stamped by the touch_device trigger in the database
           "{\"id\":\"%s\",\"name\":\"%s\",\"room\":\"%s\",\"firmware\":\"%s\","
           "\"ip\":\"%s\",\"rssi\":%d,\"uptime_s\":%lu}",
           DEVICE_ID, DEVICE_NAME, DEVICE_ROOM, FIRMWARE,
           WiFi.localIP().toString().c_str(), WiFi.RSSI(),
           (unsigned long)(millis() / 1000UL));

  // merge-duplicates turns the insert into an upsert on the primary key
  supabasePost("/rest/v1/devices", body,
               "return=minimal,resolution=merge-duplicates");
}

void postTelemetry(const Snapshot &s) {
  char body[700];
  snprintf(body, sizeof(body),
           "{\"device_id\":\"%s\",\"status\":\"%s\",\"confidence\":%.4f,"
           "\"p_idle\":%.4f,\"p_normal\":%.4f,\"p_fall\":%.4f,\"p_abnormal\":%.4f,"
           "\"acc_mean\":%.4f,\"acc_std\":%.4f,\"acc_min\":%.4f,\"acc_peak\":%.4f,"
           "\"gyro_mean\":%.2f,\"gyro_peak\":%.2f,\"jerk\":%.4f,\"tilt\":%.2f,"
           "\"stillness\":%.4f,\"infer_us\":%lu,\"rssi\":%d,\"uptime_s\":%lu}",
           DEVICE_ID, CLASS_NAMES[s.cls], s.prob[s.cls],
           s.prob[C_IDLE], s.prob[C_NORMAL], s.prob[C_FALL], s.prob[C_ABNORMAL],
           s.feat[0], s.feat[1], s.feat[2], s.feat[3],
           s.feat[7], s.feat[9], s.feat[10], s.feat[11],
           s.feat[12], (unsigned long)s.inferUs, WiFi.RSSI(),
           (unsigned long)(millis() / 1000UL));

  supabasePost("/rest/v1/telemetry", body, "return=minimal");
}

void postFall(const FallEvent &e) {
  char body[400];
  snprintf(body, sizeof(body),
           "{\"device_id\":\"%s\",\"confidence\":%.4f,\"impact_g\":%.3f,"
           "\"freefall_g\":%.3f,\"rotation_dps\":%.1f,\"tilt_deg\":%.1f,"
           "\"event_index\":%lu}",
           DEVICE_ID, e.confidence, e.impactG, e.freefallG,
           e.rotationDps, e.tiltDeg, (unsigned long)e.index);

  if (supabasePost("/rest/v1/fall_events", body, "return=minimal"))
    Serial.println(F("[cloud] fall event delivered"));
  else
    Serial.println(F("[cloud] fall event FAILED, will retry"));
}

void connectWiFi() {
  Serial.printf("[wifi] connecting to %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) {
    delay(400);
    Serial.print('.');
  }
  if (WiFi.status() == WL_CONNECTED)
    Serial.printf("\n[wifi] connected, IP %s, RSSI %d dBm\n",
                  WiFi.localIP().toString().c_str(), WiFi.RSSI());
  else
    Serial.println(F("\n[wifi] failed, will keep retrying in the background"));
}

// ===========================================================================
//  NETWORK TASK  -  core 0
// ===========================================================================
void netTask(void *arg) {
  netClient.setInsecure();          // skip cert pinning, fine for a demo node
  connectWiFi();
  upsertDevice();

  uint32_t lastTel = 0, lastHb = 0, lastRetry = 0;
  FallEvent ev;

  for (;;) {
    if (WiFi.status() != WL_CONNECTED) {
      gCloudOk = false;
      if (millis() - lastRetry > 10000) {
        lastRetry = millis();
        WiFi.disconnect();
        WiFi.begin(WIFI_SSID, WIFI_PASS);
      }
      vTaskDelay(pdMS_TO_TICKS(500));
      continue;
    }

    // falls jump the queue and go out immediately
    while (xQueueReceive(gFallQueue, &ev, 0) == pdTRUE) postFall(ev);

    if (millis() - lastTel >= TELEMETRY_MS) {
      lastTel = millis();
      Snapshot local;
      local.valid = false;
      if (xSemaphoreTake(gSnapLock, pdMS_TO_TICKS(20)) == pdTRUE) {
        local = gSnap;
        xSemaphoreGive(gSnapLock);
      }
      if (local.valid) postTelemetry(local);
    }

    if (millis() - lastHb >= HEARTBEAT_MS) {
      lastHb = millis();
      upsertDevice();
    }

    vTaskDelay(pdMS_TO_TICKS(40));
  }
}

// ===========================================================================
//  ALARM  -  non blocking so sampling never stops
// ===========================================================================
void serviceAlarm() {
  if (millis() < alarmUntil) {
    bool on = ((millis() / 120) % 2) == 0;
    digitalWrite(LED_PIN, on ? HIGH : LOW);
#if USE_BUZZER
    digitalWrite(BUZZER_PIN, on ? HIGH : LOW);
#endif
  } else {
    // idle heartbeat: short blink when the cloud link is healthy
    digitalWrite(LED_PIN, (gCloudOk && (millis() % 2000) < 60) ? HIGH : LOW);
#if USE_BUZZER
    digitalWrite(BUZZER_PIN, LOW);
#endif
  }
}

// ===========================================================================
void setup() {
  Serial.begin(115200);
  delay(800);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
#if USE_BUZZER
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
#endif
  Wire.begin(SDA_PIN, SCL_PIN, 400000);

  Serial.println(F("\n===================================="));
  Serial.println(F("   TinyGuardian - Cloud Fall Node"));
  Serial.println(F("===================================="));

  if (!mpuInit()) {
    Serial.println(F("ERROR: MPU6050 not responding. Check SDA=21, SCL=22, 3V3, GND."));
    while (true) { digitalWrite(LED_PIN, !digitalRead(LED_PIN)); delay(150); }
  }
  calibrateGyro();

  gSnapLock  = xSemaphoreCreateMutex();
  gFallQueue = xQueueCreate(8, sizeof(FallEvent));
  xTaskCreatePinnedToCore(netTask, "netTask", 10240, NULL, 1, NULL, 0);

  Serial.println(F("\nReady. Drop the board on something soft to test.\n"));
}

void loop() {
  static uint32_t nextTick = 0;
  if (nextTick == 0) nextTick = micros();

  serviceAlarm();

  // ---- read one sample every 20 ms ----
  if ((int32_t)(micros() - nextTick) >= 0) {
    nextTick += SAMPLE_US;
    float a[3], g[3];
    if (readMotion(a, g)) {
      bufAx[head] = a[0]; bufAy[head] = a[1]; bufAz[head] = a[2];
      bufGx[head] = g[0]; bufGy[head] = g[1]; bufGz[head] = g[2];
      head = (head + 1) % TG_WIN;
      if (filled < TG_WIN) filled++;
      sinceInfer++;
    }
  }

  // ---- classify every half second ----
  if (filled >= TG_WIN && sinceInfer >= TG_HOP) {
    sinceInfer = 0;

    float feat[TG_NFEAT], probs[TG_NCLS];
    uint32_t t0 = micros();
    extractFeatures(feat);
    runModel(feat, probs);
    uint32_t us = micros() - t0;

    int best = 0;
    for (int i = 1; i < TG_NCLS; i++) if (probs[i] > probs[best]) best = i;

    // hand the result to the network core
    if (xSemaphoreTake(gSnapLock, pdMS_TO_TICKS(5)) == pdTRUE) {
      gSnap.cls = best;
      gSnap.inferUs = us;
      gSnap.valid = true;
      for (int i = 0; i < TG_NCLS; i++)  gSnap.prob[i] = probs[i];
      for (int i = 0; i < TG_NFEAT; i++) gSnap.feat[i] = feat[i];
      xSemaphoreGive(gSnapLock);
    }

    Serial.printf("Status: %-9s  confidence %3.0f%%   peak %.2f g   (%lu us)  cloud %s\n",
                  CLASS_NAMES[best], probs[best] * 100.0f, feat[3],
                  (unsigned long)us, gCloudOk ? "up" : "down");

    if (best == C_FALL && probs[C_FALL] >= FALL_THRESHOLD &&
        millis() - lastFallMs > COOLDOWN_MS) {

      lastFallMs = millis();
      fallCount++;
      alarmUntil = millis() + ALARM_MS;

      FallEvent ev;
      ev.confidence  = probs[C_FALL];
      ev.impactG     = feat[3];
      ev.freefallG   = feat[2];
      ev.rotationDps = feat[9];
      ev.tiltDeg     = feat[11];
      ev.index       = fallCount;
      xQueueSend(gFallQueue, &ev, 0);

      Serial.println();
      Serial.println(F("############################################"));
      Serial.println(F("#            F A L L   D E T E C T E D     #"));
      Serial.println(F("############################################"));
      Serial.printf ("  Confidence   : %.1f %%\n", probs[C_FALL] * 100.0f);
      Serial.printf ("  Impact peak  : %.2f g\n", feat[3]);
      Serial.printf ("  Free fall min: %.2f g\n", feat[2]);
      Serial.printf ("  Peak rotation: %.0f dps\n", feat[9]);
      Serial.printf ("  Tilt change  : %.0f degrees\n", feat[11]);
      Serial.printf ("  Total falls  : %lu\n", (unsigned long)fallCount);
      Serial.printf ("  Rows posted  : %lu ok / %lu failed\n",
                     (unsigned long)gPosted, (unsigned long)gFailed);
      Serial.println();
    }
  }
}
