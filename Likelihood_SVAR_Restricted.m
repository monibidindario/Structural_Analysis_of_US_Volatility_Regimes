function [logLik]=Likelihood_SVAR_Restricted(teta)

global Sigma_1Regime
global Sigma_2Regime
global Sigma_3Regime

global T1
global T2
global T3

C=[teta(1) 0       0;
   teta(2) teta(3) 0;
   0       0       teta(4)];

Q2=[teta(5) 0       teta(8);
    teta(6) teta(7) 0;
    0       0       teta(9)];

Q3=[0        0        teta(12);
    teta(10) teta(11) teta(13);
    0        0        teta(14)];

    K1 = (C);
    K1 = K1^(-1);
    K2 = (C+Q2);
    K2 = K2^(-1);
    K3 = (C+Q2+Q3);
    K3 = K3^(-1);
    
    T=T1+T2+T3;
    M=size(C,1);

    logLik=-(-0.5*T*M*(log(2*pi))...
        +0.5*T1*log((det(K1))^2)-0.5*T1*trace(K1'*K1*Sigma_1Regime)...
        +0.5*T2*log((det(K2))^2)-0.5*T2*trace(K2'*K2*Sigma_2Regime)...
        +0.5*T3*log((det(K3))^2)-0.5*T3*trace(K3'*K3*Sigma_3Regime));    


end