%
% main routine
%

source ("clm.m");

load noise1.dat
load noise2.dat

X1 = DATA1(:,1);
Y1 = DATA1(:,2);
X2 = DATA2(:,1);
Y2 = DATA2(:,2);

u0 = LS (X1, Y1, X2, Y2, 600.0);
u = CLM (X1, Y1, X2, Y2, 600.0, u0);

format long;
F = reshape(u, 3, 3);
%F = F.'
