%% Angelini G., Bacchiocchi E., Caggiano G. and Fanelli L. (2018) "Uncertainty Across Volatility Regimes"
% This code replicates Table 2 of the paper.

clear
clc


global NLags
global VAR_Variables_X
global VAR_Variables_Y
global T1
global T2
global T3
global Sigma_1Regime
global Sigma_2Regime
global Sigma_3Regime
global StandardErrorSigma_1Regime
global StandardErrorSigma_2Regime
global StandardErrorSigma_3Regime
global CommonPI

%% General data
NLags = 4; % Number of lags of the reduced form VARs
options = optimset('MaxFunEvals',200000,'TolFun',1e-1000,'MaxIter',200000,'TolX',1e-1000);   

LimitTEST = 1.64; %
LimitTEST_Apha = 0.1;

% Graphs settings
LineWidth_IRF=2;
LineWidth_IRF_BOUNDS=1;
FontSizeIRFGraph=12;
FontSizeTimeSerieGraph=14;
HorizonIRF = 60;

% Data set
DataSet = readmatrix('DataSet.txt','Delimiter','\t');

% Break dates
TB1=284; 
TB2=569;

%% Data Set
DataSet = DataSet(2:end,[3 7 1]);
AllDataSet=DataSet;
M=size(DataSet,2);

UM1=DataSet(:,1); % Macro Uncertainty variable
Y=DataSet(:,2); % Industrial Production variable
UMF=DataSet(:,3); % Financial Uncertainty variable

% Creates the data for the three regimes
DataSet_1Regime=DataSet(1:TB1,:); % First regime
DataSet_2Regime=DataSet(TB1+1-NLags:TB2,:); % Second regime
DataSet_3Regime=DataSet(TB2+1-NLags:end,:); % Third regime

% Size three regimes
T1 = size(DataSet_1Regime,1)-NLags;
T2 = size(DataSet_2Regime,1)-NLags;
T3 = size(DataSet_3Regime,1)-NLags;
TAll = size(DataSet,1)-NLags;

%% Reduced form estimation
% Whole sample (Not necessary for replicating Table 2)

T=TAll;
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
VAR_Variables_Y=DataSet(NLags+1:end,:);

DuplicationMatrix = zeros(M^2,0.5*M*(M+1));
DuplicationMatrix(1,1)=1;
DuplicationMatrix(2,2)=1;
DuplicationMatrix(3,3)=1;
DuplicationMatrix(4,2)=1;
DuplicationMatrix(5,4)=1;
DuplicationMatrix(6,5)=1;
DuplicationMatrix(7,3)=1;
DuplicationMatrix(8,5)=1;
DuplicationMatrix(9,6)=1;
mDD=(DuplicationMatrix'*DuplicationMatrix)^(-1)*DuplicationMatrix';
mNN=DuplicationMatrix*mDD;

KommutationMatrix = zeros(M^2,M^2);
KommutationMatrix(1,1)=1;
KommutationMatrix(2,4)=1;
KommutationMatrix(3,7)=1;
KommutationMatrix(4,2)=1;
KommutationMatrix(5,5)=1;
KommutationMatrix(6,8)=1;
KommutationMatrix(7,3)=1;
KommutationMatrix(8,6)=1;
KommutationMatrix(9,9)=1;

NMatrix = 0.5*(eye(M^2)+KommutationMatrix);

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;

x0 = Beta_OLS';              % 3x13
x0 = x0(:);                  % vettore colonna

[beta_vec,Log_LK,exitflag,~,grad,HESSIAN_LK] = fminunc(@(b) Likelihood_UNRESTRICTED(reshape(b,size(Beta_OLS'))), x0, options);

Beta_LK = reshape(beta_vec,size(Beta_OLS'));   % torna 3x13 come prima
% catch
% Beta_LK=Beta_OLS';
% Log_LK=Likelihood_UNRESTRICTED_Error(Beta_OLS');    
% end

CommonPI=Beta_LK;
Errors=VAR_Variables_Y-VAR_Variables_X*Beta_LK';
Omega_LK=1/(T)*Errors'*Errors;
Sigma_AllSample = Omega_LK;
% C = cov(Errors)
LK_AllSample = -Log_LK;
% Standard errors of the reduced form parameters (autoregressive parameters)
StandardErrors_BETA=reshape(sqrt(diag(kron(Omega_LK,(VAR_Variables_X'*VAR_Variables_X)^(-1)))),M*NLags+1,M);

% Standard errors of the reduced form parameters (covariance matrix)
StandardErrors_Omega=sqrt(diag(2/T*((mDD*kron(Omega_LK,Omega_LK)*(mDD)'))));
StandardErrors_Omega_M=[StandardErrors_Omega(1) StandardErrors_Omega(2) StandardErrors_Omega(3);
                        StandardErrors_Omega(2) StandardErrors_Omega(4) StandardErrors_Omega(5);
                        StandardErrors_Omega(3) StandardErrors_Omega(5) StandardErrors_Omega(6)];

% ******************************************************************************
% First Regime
% ******************************************************************************
T=T1;
DataSet=DataSet_1Regime;
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
VAR_Variables_Y=DataSet(NLags+1:end,:);

DuplicationMatrix = zeros(M^2,0.5*M*(M+1));
DuplicationMatrix(1,1)=1;
DuplicationMatrix(2,2)=1;
DuplicationMatrix(3,3)=1;
DuplicationMatrix(4,2)=1;
DuplicationMatrix(5,4)=1;
DuplicationMatrix(6,5)=1;
DuplicationMatrix(7,3)=1;
DuplicationMatrix(8,5)=1;
DuplicationMatrix(9,6)=1;
mDD=(DuplicationMatrix'*DuplicationMatrix)^(-1)*DuplicationMatrix';
mNN=DuplicationMatrix*mDD;

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;
% try
[Beta_LK,Log_LK,exitflag,output,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options);
% catch
% Beta_LK=Beta_OLS';
% Log_LK=Likelihood_UNRESTRICTED_Error(Beta_OLS');    
% end

LK_1Regime_Sampe = Log_LK;

Errors=VAR_Variables_Y-VAR_Variables_X*Beta_LK';
Omega_LK=1/(T)*Errors'*Errors;
% Standard errors of the reduced form parameters (autoregressive parameters)
StandardErrors_BETA=reshape(sqrt(diag(kron(Omega_LK,(VAR_Variables_X'*VAR_Variables_X)^(-1)))),M*NLags+1,M);
% Standard errors of the reduced form parameters (covariance matrix)
StandardErrors_Omega=sqrt(diag(2/T*((mDD*kron(Omega_LK,Omega_LK)*(mDD)'))));
StandardErrors_Omega_M=[StandardErrors_Omega(1) StandardErrors_Omega(2) StandardErrors_Omega(3);
                                StandardErrors_Omega(2) StandardErrors_Omega(4) StandardErrors_Omega(5);
                                StandardErrors_Omega(3) StandardErrors_Omega(5) StandardErrors_Omega(6)];

SE_Sigma_1Regime=StandardErrors_Omega;
StandardErrorSigma_1Regime=(2/T*((mDD*kron(Omega_LK,Omega_LK)*(mDD)')));

CompanionMatrix_1Regime=[Beta_LK(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)];                            
                            
Omega_1Regime=[Omega_LK;StandardErrors_Omega_M];
Beta_1Regime=[Beta_LK'; StandardErrors_BETA];

Sigma_1Regime=Omega_LK;

% Likelihood of the VAR in the first regime
LK_1Regime=[-Log_LK];

% ******************************************************************************
% Second Regime
% ******************************************************************************

T=T2;
DataSet=DataSet_2Regime;
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
VAR_Variables_Y=DataSet(NLags+1:end,:);

DuplicationMatrix = zeros(M^2,0.5*M*(M+1));
DuplicationMatrix(1,1)=1;
DuplicationMatrix(2,2)=1;
DuplicationMatrix(3,3)=1;
DuplicationMatrix(4,2)=1;
DuplicationMatrix(5,4)=1;
DuplicationMatrix(6,5)=1;
DuplicationMatrix(7,3)=1;
DuplicationMatrix(8,5)=1;
DuplicationMatrix(9,6)=1;
mDD=(DuplicationMatrix'*DuplicationMatrix)^(-1)*DuplicationMatrix';
mNN=DuplicationMatrix*mDD;

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;
% try
[Beta_LK,Log_LK,exitflag,output,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options);
% catch
% Beta_LK=Beta_OLS';
% Log_LK=Likelihood_UNRESTRICTED_Error(Beta_OLS');    
% end
Errors=VAR_Variables_Y-VAR_Variables_X*Beta_LK';
Omega_LK=1/(T)*Errors'*Errors;
% Standard errors of the reduced form parameters (autoregressive parameters)
StandardErrors_BETA=reshape(sqrt(diag(kron(Omega_LK,(VAR_Variables_X'*VAR_Variables_X)^(-1)))),M*NLags+1,M);
% Standard errors of the reduced form parameters (Covariance matrix)
StandardErrors_Omega=sqrt(diag(2/T*((mDD*kron(Omega_LK,Omega_LK)*(mDD)'))));
StandardErrors_Omega_M=[StandardErrors_Omega(1) StandardErrors_Omega(2) StandardErrors_Omega(3);
                                StandardErrors_Omega(2) StandardErrors_Omega(4) StandardErrors_Omega(5);
                                StandardErrors_Omega(3) StandardErrors_Omega(5) StandardErrors_Omega(6)];
LK_2Regime_Sampe = Log_LK;
                            
SE_Sigma_2Regime=StandardErrors_Omega;
StandardErrorSigma_2Regime=(2/T*((mDD*kron(Omega_LK,Omega_LK)*(mDD)')));

CompanionMatrix_2Regime=[Beta_LK(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)];                            
                             
Omega_2Regime=[Omega_LK;StandardErrors_Omega_M];
Beta_2Regime=[Beta_LK'; StandardErrors_BETA];

Sigma_2Regime=Omega_LK;

% Likelihood of the VAR in the second regime
LK_2Regime=[-Log_LK];
          
% ******************************************************************************
% Third Regime
% ******************************************************************************

T=T3;
DataSet=DataSet_3Regime;
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
VAR_Variables_Y=DataSet(NLags+1:end,:);

DuplicationMatrix = zeros(M^2,0.5*M*(M+1));
DuplicationMatrix(1,1)=1;
DuplicationMatrix(2,2)=1;
DuplicationMatrix(3,3)=1;
DuplicationMatrix(4,2)=1;
DuplicationMatrix(5,4)=1;
DuplicationMatrix(6,5)=1;
DuplicationMatrix(7,3)=1;
DuplicationMatrix(8,5)=1;
DuplicationMatrix(9,6)=1;
mDD=(DuplicationMatrix'*DuplicationMatrix)^(-1)*DuplicationMatrix';
mNN=DuplicationMatrix*mDD;

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;

[Beta_LK,Log_LK,exitflag,output,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options);

LK_3Regime_Sampe = Log_LK;

Errors=VAR_Variables_Y-VAR_Variables_X*Beta_LK';
Omega_LK=1/(T)*Errors'*Errors;

% Standard errors of the reduced form parameters (autoregressive parameters)
StandardErrors_BETA=reshape(sqrt(diag(kron(Omega_LK,(VAR_Variables_X'*VAR_Variables_X)^(-1)))),M*NLags+1,M);

% Standard errors of the reduced form parameters (covariance matrix)
StandardErrors_Omega=sqrt(diag(2/T*((mDD*kron(Omega_LK,Omega_LK)*(mDD)'))));
StandardErrors_Omega_M=[StandardErrors_Omega(1) StandardErrors_Omega(2) StandardErrors_Omega(3);
                                StandardErrors_Omega(2) StandardErrors_Omega(4) StandardErrors_Omega(5);
                                StandardErrors_Omega(3) StandardErrors_Omega(5) StandardErrors_Omega(6)];

SE_Sigma_3Regime=StandardErrors_Omega;
StandardErrorSigma_3Regime=(2/T*((mDD*kron(Omega_LK,Omega_LK)*(mDD)')));
                            
CompanionMatrix_3Regime=[Beta_LK(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)];                            
                                       
Omega_3Regime=[Omega_LK;StandardErrors_Omega_M];
Beta_3Regime=[Beta_LK'; StandardErrors_BETA];

Sigma_3Regime=Omega_LK;

% Likelihood of the VAR in the third regime  
LK_3Regime=[-Log_LK];

%% Estimation of the structural parameters
% 
% Upper Panel of Table 2
StructuralParam=16; 
InitialValue_SVAR_Initial=[
0.5;
-0.5;
0;
0.5;
0.5;
0;
0.5;
0;
0;
0.5;
0.5;
-0.5;
0.7;
0.5;
-0.5;
-0.5]';


% InitialValue_SVAR_Initial = [0.0421063462998413;-0.0117405094668808;-0.0448104973829488;0.0368304377782917;-0.0157488924466030;0.00580239808317136;-0.0218826768384612;0.0446235881977053;0.0253441293699309;0.0590524902571742;0.0251946247347187;0.0113867188814942;-0.0446119691076574;0.0448659977279010;0.0428022339672646;0.00907094402960949];
% ML function
[StructuralParam_Estiamtion_MATRIX,Likelihood_MATRIX,exitflag,output,grad,Hessian_MATRIX] = fminunc('Likelihood_SVAR_Restricted_Upper',InitialValue_SVAR_Initial',options);

StructuralParam_Estiamtion=StructuralParam_Estiamtion_MATRIX;
LK_Estimation=Likelihood_MATRIX;
Hessian_Estimation=Hessian_MATRIX;
SE_Estimation=diag(Hessian_Estimation^(-1)).^0.5;

% Overidentification LR test
PVarl_LRTest = 1-chi2cdf(2 * ((LK_1Regime(1)+LK_2Regime(1)+LK_3Regime(1))+LK_Estimation),18-StructuralParam);
 
Parameters = [ [1:StructuralParam]' StructuralParam_Estiamtion SE_Estimation]; 

% Here below we define the matrices of the structural parameters with restrictions on the coefficients as described in eq. (18) of the paper. SVAR_C corresponds to the B matrix in the paper, SVAR_Q2 corresponds to
% the Q2 matrix of the paper and SVAR_Q3 corresponds to the Q3 matrix of the paper.

SVAR_C=[StructuralParam_Estiamtion(1) StructuralParam_Estiamtion(3) 0;
        StructuralParam_Estiamtion(2) StructuralParam_Estiamtion(4) 0;
        0                             0                             StructuralParam_Estiamtion(5)];

SVAR_Q2=[StructuralParam_Estiamtion(6)  0                             StructuralParam_Estiamtion(10);
         StructuralParam_Estiamtion(7)  StructuralParam_Estiamtion(9) 0;
         StructuralParam_Estiamtion(8)  0                             StructuralParam_Estiamtion(11)];

SVAR_Q3=[0                              0                              StructuralParam_Estiamtion(14);
         StructuralParam_Estiamtion(12) StructuralParam_Estiamtion(13) StructuralParam_Estiamtion(15);
         0                              0                              StructuralParam_Estiamtion(16)];
              
SVAR_1Regime=SVAR_C; % B
SVAR_2Regime=SVAR_C+SVAR_Q2;   % B+Q2
SVAR_3Regime=SVAR_C+SVAR_Q2+SVAR_Q3;  % B+Q2+Q3

% Flip the sign if the paraeter on the main diagonal is negative

	if SVAR_1Regime(1,1)<0
    SVAR_1Regime(:,1)=-SVAR_1Regime(:,1);
    end
    if SVAR_1Regime(2,2)<0
    SVAR_1Regime(:,2)=-SVAR_1Regime(:,2); 
    end
    if SVAR_1Regime(3,3)<0
    SVAR_1Regime(:,3)=-SVAR_1Regime(:,3);
    end
    
	if SVAR_2Regime(1,1)<0
    SVAR_2Regime(:,1)=-SVAR_2Regime(:,1);
    end
    if SVAR_2Regime(2,2)<0
    SVAR_2Regime(:,2)=-SVAR_2Regime(:,2); 
    end
    if SVAR_2Regime(3,3)<0
    SVAR_2Regime(:,3)=-SVAR_2Regime(:,3);
    end
    
    if SVAR_3Regime(1,1)<0
    SVAR_3Regime(:,1)=-SVAR_3Regime(:,1);
    end
    if SVAR_3Regime(2,2)<0
    SVAR_3Regime(:,2)=-SVAR_3Regime(:,2); 
    end
    if SVAR_3Regime(3,3)<0
    SVAR_3Regime(:,3)=-SVAR_3Regime(:,3);
    end
     
MATRICES=[SVAR_1Regime;
          SVAR_2Regime;
          SVAR_3Regime]
   
% Calculates the analytical derivatives organized in block matrices      
V11=2*NMatrix*kron(SVAR_C,eye(M));
V21=2*NMatrix*kron(SVAR_C,eye(M))+kron(SVAR_Q2,eye(M))+kron(eye(M),SVAR_Q2)*KommutationMatrix;
V22=kron(eye(M),SVAR_C)*KommutationMatrix+kron(SVAR_C,eye(M))+2*NMatrix*kron(SVAR_Q2,eye(M));
V31=2*NMatrix*kron(SVAR_C,eye(M))+kron(SVAR_Q2,eye(M))+kron(SVAR_Q3,eye(M))+kron(eye(M),SVAR_Q2)*KommutationMatrix+kron(eye(M),SVAR_Q3)*KommutationMatrix;
V32=kron(eye(M),SVAR_C)*KommutationMatrix+kron(SVAR_C,eye(M))+2*NMatrix*kron(SVAR_Q2,eye(M))+kron(SVAR_Q3,eye(M))+kron(eye(M),SVAR_Q3)*KommutationMatrix;
V33=kron(eye(M),SVAR_C)*KommutationMatrix+kron(eye(M),SVAR_Q2)*KommutationMatrix+2*NMatrix*kron(SVAR_Q3,eye(M))+kron(SVAR_C,eye(M))+kron(SVAR_Q2,eye(M));

% Calculates the matrix for checking the rank condition (full column rank)
RankMatrix=kron(eye(3),mDD)*[V11 zeros(M^2,M^2) zeros(M^2,M^2);
                             V21 V22            zeros(M^2,M^2);
                             V31 V32            V33]; 
 
% Selection matrix for extracting the structural parameters                         
HSelection=zeros(M*M*3,StructuralParam);
HSelection(1,1)=1;
HSelection(2,2)=1;
HSelection(4,3)=1;
HSelection(5,4)=1;
HSelection(9,5)=1;
HSelection(10,6)=1;
HSelection(11,7)=1;
HSelection(12,8)=1;
HSelection(14,9)=1;
HSelection(16,10)=1;
HSelection(18,11)=1;
HSelection(20,12)=1;
HSelection(23,13)=1;
HSelection(25,14)=1;
HSelection(26,15)=1;
HSelection(27,16)=1;

Jacobian= RankMatrix*HSelection;

% Report the rank of the matrix for checking the identification
rank(Jacobian)

MSigma=size(StandardErrorSigma_1Regime,1);

TetaMatrix=[StandardErrorSigma_1Regime zeros(MSigma, MSigma)      zeros(MSigma, MSigma);
            zeros(MSigma, MSigma)      StandardErrorSigma_2Regime zeros(MSigma, MSigma);
            zeros(MSigma, MSigma)      zeros(MSigma, MSigma)      StandardErrorSigma_3Regime];
%      
% % Calculates the variance and the standard errors of the estimated coefficients        
VarTeta=  (Jacobian'* TetaMatrix^(-1)*Jacobian)^(-1);
SETetaJacobian= diag(VarTeta).^0.5;
% 
%% SISTEMARE
StructuralEstimationCorrected=[
        MATRICES(1,1);
        MATRICES(2,1);
        MATRICES(2,2);
        MATRICES(2,3);
        MATRICES(3,3);
        MATRICES(4,1)-MATRICES(1,1);
        MATRICES(5,1)-MATRICES(2,1);
        MATRICES(6,1);
        MATRICES(5,2)-MATRICES(2,2);
        MATRICES(4,3);
        MATRICES(6,3)-MATRICES(3,3);
        MATRICES(8,1)-MATRICES(5,1);         
        MATRICES(8,2)-MATRICES(5,2);
        MATRICES(7,3)-MATRICES(4,3);
        MATRICES(8,3);
        MATRICES(9,3)-MATRICES(6,3);
        ];
% 
OUTPUT_Table2_StructuralEstimation=[StructuralEstimationCorrected SE_Estimation SETetaJacobian];
% % 
%% Estimation of the standard errors of the parameters in B, B+Q2, B+Q2+Q3 (delta method)

VAR_Est = Hessian_Estimation^(-1); 
 
% Here below we calculate the standard errors for B+Q2 and B+Q2+Q3 where we
% sum two ore three structural parameters. The indeces i,j,k refer to B,
% Q2, and Q3 respectively.

i=1; 
j=6;
index=1;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y z
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);
    
i=2;
j=7;
index=2;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y z
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);

i=4;
j=9;
index=3;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y z
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);

i=5;
j=11;
index=4;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y z
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);

i=2;
j=7;
k=12;
index=5;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
syms x y z
f = x + y + z;
gradient_sigma=gradient(f, [x, y z]);
gradient_sigma_est=subs(gradient_sigma,[x y z],[first_par second_par third_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k],[i j k])*gradient_sigma_Matrix').^0.5);
 
i=4;
j=9;
k=13;
index=6;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
syms x y z
f = x + y + z;
gradient_sigma=gradient(f, [x, y z]);
gradient_sigma_est=subs(gradient_sigma,[x y z],[first_par second_par third_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k],[i j k])*gradient_sigma_Matrix').^0.5);
 
i=10;
j=14;
index=7;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y z
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);

i=5;
j=11;
k=16;
index=8;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
syms x y z
f = x + y + z;
gradient_sigma=gradient(f, [x, y z]);
gradient_sigma_est=subs(gradient_sigma,[x y z],[first_par second_par third_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k],[i j k])*gradient_sigma_Matrix').^0.5);
 
% standard error matrices

SE_ANALYTIC = [SE_Estimation; SETetaDelta];         
                         
OUTPUT_Table2_SE_Analytic = [SE_ANALYTIC(1)  SE_ANALYTIC(3)  0;
                             SE_ANALYTIC(2)  SE_ANALYTIC(4)  0;
                             0               0               SE_ANALYTIC(5);
                             SE_ANALYTIC(17) SE_ANALYTIC(3)  SE_ANALYTIC(10);
                             SE_ANALYTIC(18) SE_ANALYTIC(19) 0;
                             SE_ANALYTIC(8)  0               SE_ANALYTIC(20);                         
                             SE_ANALYTIC(17) SE_ANALYTIC(3)  SE_ANALYTIC(23);
                             SE_ANALYTIC(21) SE_ANALYTIC(22) SE_ANALYTIC(15);
                             SE_ANALYTIC(8)  0               SE_ANALYTIC(24);
                             ];

SVAR_1Regime_SE_UP = OUTPUT_Table2_SE_Analytic(1:3,:);
SVAR_2Regime_SE_UP = OUTPUT_Table2_SE_Analytic(4:6,:);
SVAR_3Regime_SE_UP = OUTPUT_Table2_SE_Analytic(7:9,:);

SVAR_1Regime_UP = SVAR_1Regime;
SVAR_2Regime_UP = SVAR_2Regime;
SVAR_3Regime_UP = SVAR_3Regime;

LR_Test_UP = 2 * ((LK_1Regime(1)+LK_2Regime(1)+LK_3Regime(1))+LK_Estimation);
PVarl_LRTest_UP = PVarl_LRTest;

%%  Lower Panel of Table 2
StructuralParam=14; 
InitialValue_SVAR_Initial=0.5*ones(StructuralParam,1);

% ML function
[StructuralParam_Estiamtion_MATRIX,Likelihood_MATRIX,exitflag,output,grad,Hessian_MATRIX] = fminunc('Likelihood_SVAR_Restricted',InitialValue_SVAR_Initial',options);

StructuralParam_Estiamtion=StructuralParam_Estiamtion_MATRIX;
LK_Estimation=Likelihood_MATRIX;
Hessian_Estimation=Hessian_MATRIX;
SE_Estimation=diag(Hessian_Estimation^(-1)).^0.5;

% Overidentification LR test
PVarl_LRTest = 1-chi2cdf(2 * ((LK_1Regime(1)+LK_2Regime(1)+LK_3Regime(1))+LK_Estimation),18-StructuralParam);
 
Parameters = [ [1:StructuralParam]' StructuralParam_Estiamtion' SE_Estimation]; 

% Here below we define the matrices of the structural parameters with restrictions on the coefficients as described in eq. (18) of the paper. SVAR_C corresponds to the B matrix in the paper, SVAR_Q2 corresponds to
% the Q2 matrix of the paper and SVAR_Q3 corresponds to the Q3 matrix of the paper.

SVAR_C=[StructuralParam_Estiamtion(1) 0                             0;
        StructuralParam_Estiamtion(2) StructuralParam_Estiamtion(3) 0;
        0                             0                             StructuralParam_Estiamtion(4)];

SVAR_Q2=[StructuralParam_Estiamtion(5)  0                             StructuralParam_Estiamtion(8);
         StructuralParam_Estiamtion(6)  StructuralParam_Estiamtion(7) 0;
         0                              0                             StructuralParam_Estiamtion(9)];

SVAR_Q3=[0                              0                              StructuralParam_Estiamtion(12);
         StructuralParam_Estiamtion(10) StructuralParam_Estiamtion(11) StructuralParam_Estiamtion(13);
         0                              0                              StructuralParam_Estiamtion(14)];
              
SVAR_1Regime=SVAR_C; % B
SVAR_2Regime=SVAR_C+SVAR_Q2;   % B+Q2
SVAR_3Regime=SVAR_C+SVAR_Q2+SVAR_Q3;  % B+Q2+Q3

% Flip the sign if the paraeter on the main diagonal is negative

	if SVAR_1Regime(1,1)<0
    SVAR_1Regime(:,1)=-SVAR_1Regime(:,1);
    end
    if SVAR_1Regime(2,2)<0
    SVAR_1Regime(:,2)=-SVAR_1Regime(:,2); 
    end
    if SVAR_1Regime(3,3)<0
    SVAR_1Regime(:,3)=-SVAR_1Regime(:,3);
    end
    
	if SVAR_2Regime(1,1)<0
    SVAR_2Regime(:,1)=-SVAR_2Regime(:,1);
    end
    if SVAR_2Regime(2,2)<0
    SVAR_2Regime(:,2)=-SVAR_2Regime(:,2); 
    end
    if SVAR_2Regime(3,3)<0
    SVAR_2Regime(:,3)=-SVAR_2Regime(:,3);
    end
    
    if SVAR_3Regime(1,1)<0
    SVAR_3Regime(:,1)=-SVAR_3Regime(:,1);
    end
    if SVAR_3Regime(2,2)<0
    SVAR_3Regime(:,2)=-SVAR_3Regime(:,2); 
    end
    if SVAR_3Regime(3,3)<0
    SVAR_3Regime(:,3)=-SVAR_3Regime(:,3);
    end
     
MATRICES=[SVAR_1Regime;
          SVAR_2Regime;
          SVAR_3Regime];

% Calculates the analytical derivatives organized in block matrices      
V11=2*NMatrix*kron(SVAR_C,eye(M));
V21=2*NMatrix*kron(SVAR_C,eye(M))+kron(SVAR_Q2,eye(M))+kron(eye(M),SVAR_Q2)*KommutationMatrix;
V22=kron(eye(M),SVAR_C)*KommutationMatrix+kron(SVAR_C,eye(M))+2*NMatrix*kron(SVAR_Q2,eye(M));
V31=2*NMatrix*kron(SVAR_C,eye(M))+kron(SVAR_Q2,eye(M))+kron(SVAR_Q3,eye(M))+kron(eye(M),SVAR_Q2)*KommutationMatrix+kron(eye(M),SVAR_Q3)*KommutationMatrix;
V32=kron(eye(M),SVAR_C)*KommutationMatrix+kron(SVAR_C,eye(M))+2*NMatrix*kron(SVAR_Q2,eye(M))+kron(SVAR_Q3,eye(M))+kron(eye(M),SVAR_Q3)*KommutationMatrix;
V33=kron(eye(M),SVAR_C)*KommutationMatrix+kron(eye(M),SVAR_Q2)*KommutationMatrix+2*NMatrix*kron(SVAR_Q3,eye(M))+kron(SVAR_C,eye(M))+kron(SVAR_Q2,eye(M));

% Calculates the matrix for checking the rank condition (full column rank)
RankMatrix=kron(eye(3),mDD)*[V11 zeros(M^2,M^2) zeros(M^2,M^2);
                             V21 V22            zeros(M^2,M^2);
                             V31 V32            V33]; 
 
% Selection matrix for extracting the structural parameters                         
HSelection=zeros(M*M*3,StructuralParam);
HSelection(1,1)=1;
HSelection(2,2)=1;
HSelection(5,3)=1;
HSelection(9,4)=1;
HSelection(10,5)=1;
HSelection(11,6)=1;
HSelection(14,7)=1;
HSelection(16,8)=1;
HSelection(18,9)=1;
HSelection(20,10)=1;
HSelection(23,11)=1;
HSelection(25,12)=1;
HSelection(26,13)=1;
HSelection(27,14)=1;

Jacobian= RankMatrix*HSelection;

% Report the rank of the matrix for checking the identification
rank(Jacobian);

MSigma=size(StandardErrorSigma_1Regime,1);

TetaMatrix=[StandardErrorSigma_1Regime zeros(MSigma, MSigma)      zeros(MSigma, MSigma);
            zeros(MSigma, MSigma)      StandardErrorSigma_2Regime zeros(MSigma, MSigma);
            zeros(MSigma, MSigma)      zeros(MSigma, MSigma)      StandardErrorSigma_3Regime];
     
% Calculates the variance and the standard errors of the estimated coefficients        
VarTeta=  (Jacobian'* TetaMatrix^(-1)*Jacobian)^(-1);
SETetaJacobian= diag(VarTeta).^0.5;

StructuralEstimationCorrected=[
        MATRICES(1,1);
        MATRICES(2,1);
        MATRICES(2,2);
        MATRICES(3,3);
        MATRICES(4,1)-MATRICES(1,1);
        MATRICES(5,1)-MATRICES(2,1);
        MATRICES(5,2)-MATRICES(2,2);
        MATRICES(4,3);
        MATRICES(6,3)-MATRICES(3,3);
        MATRICES(8,1)-MATRICES(5,1);         
        MATRICES(8,2)-MATRICES(5,2);
        MATRICES(7,3)-MATRICES(4,3);
        MATRICES(8,3);
        MATRICES(9,3)-MATRICES(6,3);
        ];

OUTPUT_Table2_StructuralEstimation=[StructuralEstimationCorrected SE_Estimation SETetaJacobian];
% 
%% Estimation of the standard errors of the parameters in B, B+Q2, B+Q2+Q3 (delta method)

VAR_Est = Hessian_Estimation^(-1); 
 
% Here below we calculate the standard errors for B+Q2 and B+Q2+Q3 where we
% sum two ore three structural parameters. The indeces i,j,k refer to B,
% Q2, and Q3 respectively.

i=1; 
j=5;
index=1;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y z
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);
    
i=2;
j=6;
index=2;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y z
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);

i=3;
j=7;
index=3;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y z
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);

i=4;
j=9;
index=4;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y z
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);

i=2;
j=6;
k=10;
index=5;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
syms x y z
f = x + y + z;
gradient_sigma=gradient(f, [x, y z]);
gradient_sigma_est=subs(gradient_sigma,[x y z],[first_par second_par third_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k],[i j k])*gradient_sigma_Matrix').^0.5);
 
i=3;
j=7;
k=11;
index=6;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
syms x y z
f = x + y + z;
gradient_sigma=gradient(f, [x, y z]);
gradient_sigma_est=subs(gradient_sigma,[x y z],[first_par second_par third_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k],[i j k])*gradient_sigma_Matrix').^0.5);
 
i=8;
j=12;
index=7;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
syms x y z
f = x + y;
gradient_sigma=gradient(f, [x, y]);
gradient_sigma_est=subs(gradient_sigma,[x y],[first_par second_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j],[i j])*gradient_sigma_Matrix').^0.5);

i=4;
j=9;
k=14;
index=8;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
syms x y z
f = x + y + z;
gradient_sigma=gradient(f, [x, y z]);
gradient_sigma_est=subs(gradient_sigma,[x y z],[first_par second_par third_par]);
gradient_sigma_Matrix=(double(gradient_sigma_est))';    
SETetaDelta(index,:)=(diag(gradient_sigma_Matrix*VAR_Est([i j k],[i j k])*gradient_sigma_Matrix').^0.5);
 
% standard error matrices

SE_ANALYTIC = [SE_Estimation;  SETetaDelta];         
                         
OUTPUT_Table2_SE_Analytic = [SE_ANALYTIC(1)  0               0;
                             SE_ANALYTIC(2)  SE_ANALYTIC(3)  0;
                             0               0               SE_ANALYTIC(4);
                             SE_ANALYTIC(15) 0               SE_ANALYTIC(8);
                             SE_ANALYTIC(16) SE_ANALYTIC(17) 0;
                             0               0               SE_ANALYTIC(18);                         
                             SE_ANALYTIC(15) 0               SE_ANALYTIC(21);
                             SE_ANALYTIC(19) SE_ANALYTIC(20) SE_ANALYTIC(13);
                             0               0               SE_ANALYTIC(22);
                             ];

SVAR_1Regime_SE = OUTPUT_Table2_SE_Analytic(1:3,:);
SVAR_2Regime_SE = OUTPUT_Table2_SE_Analytic(4:6,:);
SVAR_3Regime_SE = OUTPUT_Table2_SE_Analytic(7:9,:);

clc
%% Output Upper Panel
disp('----------------------------------------------------------------')
disp('----------------------- UPPER PANEL ----------------------------')
disp('----------------------------------------------------------------')
disp('----------------------- Coefficients ---------------------------')

disp('B=')
disp(SVAR_1Regime_UP)
disp('B+Q2=')
disp(SVAR_2Regime_UP)
disp('B+Q2+Q3=')
disp(SVAR_3Regime_UP)

disp('---------------------- Standard Errors -------------------------')

disp('B=')
disp(SVAR_1Regime_SE_UP)
disp('B+Q2=')
disp(SVAR_2Regime_SE_UP)
disp('B+Q2+Q3=')
disp(SVAR_3Regime_SE_UP)

disp('-------------- 2 overidentification restrictions: --------------')
disp('Test statistics:')
disp(LR_Test_UP)
disp('P-value:')
disp(PVarl_LRTest_UP)


%% Output Lower Panel
disp('----------------------------------------------------------------')
disp('----------------------- LOWER PANEL ----------------------------')
disp('----------------------------------------------------------------')
disp('----------------------- Coefficients ---------------------------')

disp('B=')
disp(SVAR_1Regime)
disp('B+Q2=')
disp(SVAR_2Regime)
disp('B+Q2+Q3=')
disp(SVAR_3Regime)

disp('---------------------- Standard Errors -------------------------')

disp('B=')
disp(SVAR_1Regime_SE)
disp('B+Q2=')
disp(SVAR_2Regime_SE)
disp('B+Q2+Q3=')
disp(SVAR_3Regime_SE)

disp('---- 4 overidentification restrictions specified in eq. 19: ----')
disp('Test statistics:')
disp(2 * ((LK_1Regime(1)+LK_2Regime(1)+LK_3Regime(1))+LK_Estimation))
disp('P-value:')
disp(PVarl_LRTest)

D = sqrt(diag(Sigma_AllSample));
Rho_AllSample = Sigma_AllSample ./ (D*D');
