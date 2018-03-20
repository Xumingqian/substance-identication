clc;
clear;
close all;
format compact;
%% import testing data  
% X1=importdata('amoxiling-1-t.csv');
load amoxiling
%% data preprocessing
m=length(X1.data);
N=pow2(nextpow2(m));
W1=hamming(N);
W2=blackman(N);
[row,~] = size(X1.data);                
Time = X1.data(1:row,1);                 
fs=1/(Time(row,1)-Time(1,1));           
Reference_Original = X1.data(1:row,2);   %reference time domain signal
Sample_Original = X1.data(1:row,3);      %sample time domain signal
Frequency=(0:N-1)*fs;                
Frequency=Frequency';
%% remove reflection peak
[ai,i]=max(Reference_Original);
[bi,j]=max( Sample_Original );
Reference_Original(i+129:end)=0 ;
Sample_Original (j+170:end)=0 ;
%% FFT
Reference1=fft(Reference_Original,N);
Reference=fftshift(Reference1,2);
Reference(1:328,1)=W1(1:328,1).*Reference(1:328,1);
Reference(329:row,1)=W2(329:row,1).*Reference(329:row,1);
Reference_Power=Reference.*conj(Reference)/N;%Power
Sample11=fft(Sample_Original,N);
Sample1=fftshift(Sample11,2);
Sample1(1:328,1)=W1(1:328,1).*Sample1(1:328,1);
Sample1(329:row,1)=W2(329:row,1).*Sample1(329:row,1);
Sample_Power=Sample1.*conj(Sample1)/N^2;
%% Calculation absorption coefficient     
a=log10(Reference_Power./ Sample_Power);   
a(1:328,1)=smooth(a(1:328,1),15,'sgolay');
a(329:row,1)=smooth(a(329:row,1),40,'sgolay');  
Absorbance=a;
%% Effective frequency domain selection
data(1:N,1)=Frequency(1:N,1);
data(1:N,2)=Absorbance(1:N,1);
x2(:,1)=data(find(data(1:N,1)>0.2&data(1:N,1)<1.51),1);
x2(:,2)=a(find(data(1:N,1)>0.2&data(1:N,1)<1.51));
xx=x2(:,1);
yy=x2(:,2);
%% cubic spline interpolation
ff=0.201514411:0.0091597459697118:1.502198339;
ff=ff';
y1(:,i)=spline(xx,yy,ff);
%% wavelet transform   
[C,L]=wavedec(x2(:,2),6,'db9');
C(1:L(1))=0;
C(L(1)+L(2)+L(3)+L(4)+1:L(1)+L(2)+L(3)+L(4)+L(5)+L(6)+L(7))=0;
x2(:,2)=waverec(C,L,'db9');
test_drug=x2(:,2)';
test_drug_labels=zeros(1,1);
labelstr={'4-anjibenjiasuan','A2',' A3','amoxicillin','phenylalanine','benzoic acid', 'C5', 'd-lactose monohydrate','p-toluylic acid', 'glutamic acid','TNT','riboflavin'
};% The training set label corresponds to the material name
%% spectrum database
load USST;
j=1;
h=1;
for i=1:10:120
   train_drug(:,j:j+9)=A(1:end,i:i+9);
   j=j+10;
end
train_drug=train_drug';
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
% [bestacc,bestc,bestg] = SVMcgForClass(train_drug_labels,train_drug,-10,10,-10,10);
[bestacc,bestc,bestg] = SVMcgForClass(train_drug_labels,train_drug,-2,4,-4,4,3,0.5,0.5,0.9);
cmd = [' -t ',num2str(0),' -c ',num2str(bestc),' -g ',num2str(bestg)];
%% SVM network training
model = svmtrain(train_drug_labels, train_drug,cmd);
%% SVM network prediction
predict_label = svmpredict(test_drug_labels, test_drug, model)';
% The substance name corresponding to the label.
predict_label_str=labelstr(predict_label)
