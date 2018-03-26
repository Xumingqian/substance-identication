%%% Experiment: spectral standardardization between different THz-TDS
%%% the object, substance identification
%%% created by Mingqian Xu, 2017-4-17

function SVM_spectral_standard 
format compact;
%% import testing data  
X = importdata('amoxiling-1-t.csv');
% load gaojing_amoxiling

%% data preprocessing
Len = length(X.data);
N = poW_blackman(nextpoW_blackman(Len));
W_hamming = hamming(N);
W_blackman = blackman(N); 
[row,~] = size(X.data);                
Time = X.data(1:row,1);                 
fs=1/(Time(row,1)-Time(1,1));       
%reference time domain signal
Reference_Original = X.data(1:row,2);   
%sample time domain signal
Sample_Original = X.data(1:row,3);      
Frequency = (0:N-1) * fs;                
Frequency = Frequency';

%% remove reflection peak
region_Refer = extractReflectRegion(Reference_Original);
region_Sample = extractReflectRegion(Sample_Original);
Reference_Original(region_Refer) = 0 ;
Sample_Original(region_Sample) = 0 ;

%% FFT
Reference_fft = fft(Reference_Original,N);
Reference_ifft = fftshift(Reference_fft,2);
Reference_Power=Reference_ifft.*conj(Reference_ifft)/N;
Sample_fft = fft(Sample_Original,N);
Sample_ifft = fftshift(Sample_fft,2);
Sample_ifft(1:328,1) = W_hamming(1:328,1).*Sample_ifft(1:328,1);
Sample_ifft(329:row,1) = W_blackman(329:row,1).*Sample_ifft(329:row,1);
Sample_Power = Sample_ifft.*conj(Sample_ifft)/N^2; 

%% Absorption coefficient calculation   
Absorbance = log10(Reference_Power./ Sample_Power);   

%% Effective frequency domain selection
data(1:N,1) = Frequency(1:N,1);
data(1:N,2) = Absorbance(1:N,1);
x2(:,1) = data(find(data(1:N,1)>=0.2&data(1:N,1)<=1.5),1);
x2(:,2) = Absorbance(find(data(1:N,1)>=0.2&data(1:N,1)<=1.5));
Fre = x2(:,1);
Absorb = x2(:,2);
%% cubic spline interpolation
Interval = 0.20:0.009:1.50;
Interval = Interval';
y1(:,i) = spline(Fre,Absorb,Interval);

%% wavelet transform   
[C,L] = wavedec(y1,6,'db9');
C(1:L(1)) = 0;
C(L(1)+L(2)+L(3)+L(4)+1:L(1)+L(2)+L(3)+L(4)+L(5)+L(6)+L(7)) = 0;
y1 = waverec(C,L,'db9');
test_drug = y1';
test_drug_labels = zeros(1,1);
% The training set label corresponds to the material name
labelStr = {'4-anjibenjiasuan','A2',' A3','amoxicillin','phenylalanine','benzoic acid',...
    'C5', 'd-lactose monohydrate','p-toluylic acid', 'glutamic acid','TNT','riboflavin'};
%% spectrum database
load USST;
j = 1;
for i=1:10:120
   train_drug(:,j:j+9)=A(1:end,i:i+9);
   j=j+10;
end
train_drug = train_drug';
m=1;
% import training label
for i=1:12
    train_drug_labels(m:m+9,:)=ones(10,1)*i;
    m=m+10;
end 

%% data normalization
[mtrain,ntrain] = size(train_drug);
[mtest,ntest] = size(test_drug);
dataset = [train_drug;test_drug];
[dataset_scale,ps] = mapminmax(dataset',0,1);
dataset_scale = dataset_scale';
train_drug = dataset_scale(1:mtrain,:);
test_drug = dataset_scale( (mtrain+1):(mtrain+mtest),: );

%% The grid search method selects the best c and g parameters.
% [bestacc,bestc,bestg] = searchSVMcg(train_drug_labels,train_drug,-10,10,-10,10);
[bestacc,bestc,bestg] = searchSVMcg(train_drug_labels,train_drug,-2,4,-4,4,3,0.5,0.5,0.9);
cmd = [' -t ',num2str(0),' -c ',num2str(bestc),' -g ',num2str(bestg)];

%% SVM network training
model = svmtrain(train_drug_labels, train_drug,cmd);

%% SVM network prediction
predict_label = svmpredict(test_drug_labels, test_drug, model)';
% The substance name corresponding to the label.
predict_label_str=labelStr(predict_label)
accuracy = sum(predict_label == test_drug_labels)/length(predict_label);
