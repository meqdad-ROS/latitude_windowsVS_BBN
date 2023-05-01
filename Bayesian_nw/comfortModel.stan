data{
    int N; //just a numeber
    real Temp; // measured temperature from sensor
    real Tmu; // Standard comfortable temperature 
    int Tstd; // the maximum (difference of comfortable temperature between summer and winter)/2 
    real Humad;
    real Hmu; // The Standard comfortable relative humadity 
    int Hstd; // The relative humadity standard daviation discribed as (max acciptable-mimi acceptable)/2
    real Rtemp; // measured radient Temperature at the space
}

parameters {
   vector[N] therm_comfort;  // Find the propability of the thermal comfort to be good.
//    real comfort;
//    real std;
}

model{
    Temp ~ normal(Tmu,Tstd); // Temperature propability 
    Humad ~ normal(Hmu, Hstd); // Relative humadity temperature
    Rtemp ~ normal(Tmu,Tstd); // radient temperature propability normally it should be equal to the dry pulb temperature if isolation is good.
    target += normal_lpdf(therm_comfort[1] | [0, Temp, Humad, Rtemp], 1); // propapility of thermal comfort
}