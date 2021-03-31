#include<Wire.h>
/*Set for IMU Index*/
const int MPU_addr=0x68;  // I2C address of the MPU-6050
const int IMU_Num=4;
const int IMU1=7;
const int IMU2=8;
const int IMU3=9;
const int IMU4=10;


/*Set for IMU Variables*/
double AcX,AcY,AcZ,Tmp,GyX,GyY,GyZ;
const float accelerationThreshold = 2.3; // threshold of significant in Accelarate
const float gyrothreshold=230;           // threshold of significant in Gyroscope
const float pi=3.1416;

float delta_pitch[IMU_Num]={0,0,0,0};
float delta_roll[IMU_Num]={0,0,0,0};
float delta_yaw[IMU_Num]={0,0,0,0};

float start_pitch[IMU_Num]={0,0,0,0};
float start_roll[IMU_Num]={0,0,0,0};
float start_yaw[IMU_Num]={0,0,0,0};

float end_pitch[IMU_Num]={0,0,0,0};
float end_roll[IMU_Num]={0,0,0,0};
float end_yaw[IMU_Num]={0,0,0,0};

float Lastroll[IMU_Num] = {0.0f,0.0f,0.0f,0.0f}; 
float Lastpitch[IMU_Num] ={0.0f,0.0f,0.0f,0.0f}; 
float Lastyaw[IMU_Num] ={0.0f,0.0f,0.0f,0.0f};

float start_acc_pitch[IMU_Num]={0,0,0,0};
float start_acc_roll[IMU_Num]={0,0,0,0};
float end_acc_pitch[IMU_Num]={0,0,0,0};
float end_acc_roll[IMU_Num]={0,0,0,0};
float delta_acc_roll[IMU_Num]={0,0,0,0};
float delta_acc_pitch[IMU_Num]={0,0,0,0};
float Last_acc_roll[IMU_Num] = {0.0f,0.0f,0.0f,0.0f}; 
float Last_acc_pitch[IMU_Num] ={0.0f,0.0f,0.0f,0.0f}; 

unsigned long nLastTime[IMU_Num]={0,0,0,0}; 


/*Set for sample freqence*/
int numSamples = 30*IMU_Num; //Once trigger the threshold, continously record 30 times
int samplesRead = 0;
int sample_step_delay=25;  //delay time(ms) for each step



void setup(){
  pinMode(IMU1,OUTPUT);
  pinMode(IMU2,OUTPUT);
  pinMode(IMU3,OUTPUT);
  pinMode(IMU4,OUTPUT);
  Wire.begin();
  Wire.setClock(400000); // I2C clock rate
  delay(100);
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x6B);  // PWR_MGMT_1 register
  Wire.write(byte(0));     // set to zero (wakes up the MPU-6050)
  Wire.endTransmission(true);
  delay(100);

  // Configure Gyro
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x1B); // Gyro Config
  Wire.write(byte(0x0)); // (0: 250 dps (131 LSB/dps); 1: 500 dps (65.5 LSB/dps) ; 2: 1000 dps ( 32.8 LSB/dps) ; 3: 2000 dps (16.4 LSB/dps)
  Wire.endTransmission();
  //
  delay(100);

  // Configure Accelerometer
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x1C); // Accelerometer  Config
  Wire.write(byte(0x0)); // (0: 2g (16384 LSB/g); 1: 4g (8192 LSB/g) ; 2: 8g ( 4096 LSB/g) ; 3: 16g (2048 LSB/g)
  Wire.endTransmission();
  //
  delay(100);

  Serial.begin(115200);
  for(int j=0;j<IMU_Num;j++)
  {
  nLastTime[j]=millis();}
}
void loop(){
  for(int i=1;i<IMU_Num+1;i++)
  { switch(i)
  {case 1: 
      digitalWrite(IMU2,HIGH);
      digitalWrite(IMU3,HIGH);
      digitalWrite(IMU4,HIGH);
      digitalWrite(IMU1,LOW);
      break;
   case 2:
      digitalWrite(IMU1,HIGH);
      digitalWrite(IMU3,HIGH);
      digitalWrite(IMU4,HIGH);
      digitalWrite(IMU2,LOW);
      break; 
   case 3:
      digitalWrite(IMU1,HIGH);
      digitalWrite(IMU2,HIGH);
      digitalWrite(IMU4,HIGH);
      digitalWrite(IMU3,LOW);
      break;
   case 4:
      digitalWrite(IMU1,HIGH);
      digitalWrite(IMU2,HIGH);
      digitalWrite(IMU3,HIGH);
      digitalWrite(IMU4,LOW);
      break;}
  delay(1);
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x3B);  // starting with register 0x3B (ACCEL_XOUT_H)
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_addr,14,true);  // request a total of 14 registers
  delay(1);
  AcX=(int16_t)(Wire.read()<<8|Wire.read())/16384.00; // g  // 0x3B (ACCEL_XOUT_H) & 0x3C (ACCEL_XOUT_L)    
  AcY=(int16_t)(Wire.read()<<8|Wire.read())/16384.00; // g // 0x3D (ACCEL_YOUT_H) & 0x3E (ACCEL_YOUT_L)
  AcZ=(int16_t)(Wire.read()<<8|Wire.read())/16384.00; // g // 0x3F (ACCEL_ZOUT_H) & 0x40 (ACCEL_ZOUT_L)
  Tmp=(int16_t)(Wire.read()<<8|Wire.read())/340.00 + 36.53;  // 0x41 (TEMP_OUT_H) & 0x42 (TEMP_OUT_L)
  GyX=(int16_t)(Wire.read()<<8|Wire.read())/131.00;  // 0x43 (GYRO_XOUT_H) & 0x44 (GYRO_XOUT_L)
  GyY=(int16_t)(Wire.read()<<8|Wire.read())/131.00; // 0x45 (GYRO_YOUT_H) & 0x46 (GYRO_YOUT_L)
  GyZ=(int16_t)(Wire.read()<<8|Wire.read())/131.00;  // 0x47 (GYRO_ZOUT_H) & 0x48 (GYRO_ZOUT_L)

/*PLEASE CHANGE YOUR OFFSET VALUE HERE for each IMU */
  if (i==3)
{GyX=GyX-22.4;} 

  unsigned long nCurTime = millis();
  double dt = double(nCurTime - nLastTime[i-1]) / 1000.0; 
  nLastTime[i-1]=nCurTime;
  //Serial.print(dt);
  /*Cauculate Eular angle with Acc*/
  float pitch=atan(AcX/sqrt(AcY*AcY+AcZ*AcZ))* 180 / pi ;
  float roll=atan(AcY/sqrt(AcX*AcX+AcZ*AcZ))* 180 / pi ;
  /*Cauculate Eular angle with Gyro*/
  float Gyroll = Lastroll[i-1]+dt*GyX;
  float Gypitch =Lastpitch[i-1]+ dt*GyY;
  float yaw=Lastyaw[i-1]+dt*GyZ;

  /*Save*/
  Lastroll[i-1]=Gyroll;
  Lastpitch[i-1]=Gypitch;
  Lastyaw[i-1]=yaw;
  Last_acc_roll[i-1]=roll;
  Last_acc_pitch[i-1]=pitch;
  
  while (samplesRead == 0 && i == 1){
    float aSum = fabs(AcX) + fabs(AcY) + fabs(AcZ);
    float gSum=fabs(GyX) + fabs(GyY) + fabs(GyZ);
    if (gSum >= gyrothreshold||aSum>=accelerationThreshold){
      samplesRead = 1;
      break;
    }
    break;
  }
  
  
  if(samplesRead>0&&samplesRead < numSamples+1){
       if(samplesRead==1){
       for (int k=0;k<IMU_Num;k++)
       {start_pitch[k]= Lastpitch[k];
       start_roll[k]=Lastroll[k];
       start_yaw[k]=Lastyaw[k];
       start_acc_pitch[k]=Last_acc_pitch[k];
       start_acc_roll[k]=Last_acc_roll[k];
       }
       }
       if(samplesRead==numSamples)
        {for (int p=0;p<IMU_Num;p++)
       {end_pitch[p]=Lastpitch[p];
        end_roll[p]=Lastroll[p];
        end_yaw[p]=Lastyaw[p];
        end_acc_pitch[p]=Last_acc_pitch[p];
        end_acc_roll[p]=Last_acc_roll[p];
       }
       }
       //Serial.print("IMU:");Serial.print(i);
       //Serial.print("\n");
       /*
       Serial.print("AcX = "); Serial.print(AcX);
       Serial.print(" | AcY = "); Serial.print(AcY);
       Serial.print(" | AcZ = "); Serial.print(AcZ);
       //Serial.print(" | Tmp = "); Serial.print(Tmp/340.00+36.53);  //equation for temperature in degrees C from datasheet
       Serial.print(" | GyX = "); Serial.print(GyX);
       Serial.print(" | GyY = "); Serial.print(GyY);
       Serial.print(" | GyZ = "); Serial.println(GyZ);
       */
      
   //Serial.print(" | pitch = "); Serial.print(pitch);
   //Serial.print(" | roll = "); Serial.print(roll);
   //Serial.print(" | Gypitch = "); Serial.print(Gypitch+start_acc_pitch[i-1]);
   //Serial.print(" | Gyroll = "); Serial.print(Gyroll+start_acc_roll[i-1]);
   //Serial.print(" | Gyyaw = "); Serial.println(yaw);
   samplesRead++;
  }


   if(samplesRead == numSamples+1 )
  {
    for (int m=0;m<IMU_Num;m++)
    {delta_roll[m]=end_roll[m]-start_roll[m];
    delta_pitch[m]=end_pitch[m]-start_pitch[m];
    delta_yaw[m]=end_yaw[m]-start_yaw[m];
    delta_acc_pitch[m]=end_acc_pitch[m]-start_acc_pitch[m];
    delta_acc_roll[m]=end_acc_roll[m]-start_acc_roll[m];
    //Serial.print(" | delta_pitch = "); 
    Serial.print(delta_pitch[m]);
    Serial.print(",");
    //Serial.print(" | delta_roll = ");
    Serial.print(delta_roll[m]);
    Serial.print(",");
    //Serial.print(" | delta_acc_pitch = "); 
    Serial.print(delta_acc_pitch[m]);
    Serial.print(",");
    //Serial.print(" | delta_acc_roll = "); 
    Serial.print(delta_acc_roll[m]);
    Serial.print(",");
    //Serial.print(" | delta_yaw = ");
    Serial.print(delta_yaw[m]);
    Serial.print(",");
    }
    samplesRead=0;
    /*Clear all variables after samplesRead>numSamples steps*/
    for(int n=0;n<IMU_Num;n++)
    {
    delta_pitch[n]=0;
    delta_roll[n]=0;
    delta_yaw[n]=0;
    start_pitch[n]=0;
    start_roll[n]=0;
    start_yaw[n]=0;
    end_pitch[n]=0;
    end_roll[n]=0;
    end_yaw[n]=0;
    Lastroll[n] = 0; 
    Lastpitch[n] =0; 
    Lastyaw[n] =0;
    start_acc_pitch[n]=0;
    start_roll[n]=0;
    end_acc_pitch[n]=0;
    end_acc_roll[n]=0;
    delta_acc_pitch[n]=0;
    delta_acc_roll[n]=0;  
    Last_acc_pitch[n]=0;
    Last_acc_roll[n]=0;
    }
    //Serial.println("Sample Finished, waiting for next gesture...");
    Serial.print('\n');
    }
 }

  delay(sample_step_delay);
}
