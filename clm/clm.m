%
% Normalization function
%
function u = normalize (u_in)
	 u = u_in / norm (u_in, 2);
endfunction

%
% Create rotation matrix
%
function A = createRotationMatrix (omega)
	 theta = norm (omega, 2);
	 l = normalize (omega);
	 c = cos (theta);
	 s = sin (theta);
	 A = [c + l(1) * l(1) * (1.0 - c) ...
	      l(1) * l(2) * (1.0 - c) - l(3) * s ...
	      l(1) * l(3) * (1.0 - c) + l(2) * s;
	      l(2) * l(1) * (1.0 - c) + l(3) * s ...
	      c + l(2) * l(2) * (1.0 - c) ...
	      l(2) * l(3) * (1.0 - c) - l(1) * s;
	      l(3) * l(1) * (1.0 - c) - l(2) * s ...
	      l(3) * l(2) * (1.0 - c) + l(1) * s ...
	      c + l(3) * l(3) * (1.0 - c)];	   	   
endfunction

%
% Compute Fu
%
function Fu = calcFu (F)
	 Fu = [ 0.0      F(3, 1) -F(2, 1);
	        0.0      F(3, 2) -F(2, 2);
		0.0      F(3, 3) -F(2, 3);
	       -F(3, 1)  0.0      F(1, 1);
	       -F(3, 2)  0.0      F(1, 2);
	       -F(3, 3)  0.0      F(1, 3);
	        F(2, 1) -F(1, 1)  0.0;
		F(2, 2) -F(1, 2)  0.0;
		F(2, 3) -F(1, 3)  0.0];	 	 
endfunction

%
% Compute Fv
%
function Fv = calcFv (F)
	 Fv = [ 0.0      F(1, 3) -F(1, 2);
	       -F(1, 3)  0.0      F(1, 1);
	        F(1, 2) -F(1, 1)  0.0;
		0.0      F(2, 3) -F(2, 2);
	       -F(2, 3)  0.0      F(2, 1);
	        F(2, 2) -F(2, 1)  0.0;
		0.0      F(3, 3) -F(3, 2);
	       -F(3, 3)  0.0      F(3, 1);
	        F(3, 2) -F(3, 1)  0.0];
endfunction

%
% Compute ut
%
function ut = calcUt (U, S, V)
	 ut = [S(1, 1) * U(1, 2) * V(1, 2) - S(2, 2) * U(1, 1) * V(1, 1);
	       S(1, 1) * U(1, 2) * V(2, 2) - S(2, 2) * U(1, 1) * V(2, 1);
	       S(1, 1) * U(1, 2) * V(3, 2) - S(2, 2) * U(1, 1) * V(3, 1);
	       S(1, 1) * U(2, 2) * V(1, 2) - S(2, 2) * U(2, 1) * V(1, 1);
	       S(1, 1) * U(2, 2) * V(2, 2) - S(2, 2) * U(2, 1) * V(2, 1);
	       S(1, 1) * U(2, 2) * V(3, 2) - S(2, 2) * U(2, 1) * V(3, 1);
	       S(1, 1) * U(3, 2) * V(1, 2) - S(2, 2) * U(3, 1) * V(1, 1);
	       S(1, 1) * U(3, 2) * V(2, 2) - S(2, 2) * U(3, 1) * V(2, 1);
	       S(1, 1) * U(3, 2) * V(3, 2) - S(2, 2) * U(3, 1) * V(3, 1)];
endfunction

%
% Set H and g
%
function [H, g] = setHandG (X, Fu, Fv, u, ut)
	 uJ = Fu' * X * u;
	 vJ = Fv' * X * u;
	 dtheta = ut' * X * u;
	 u2J = Fu' * X * Fu;
	 v2J = Fv' * X * Fv;
	 uvJ = Fu' * X * Fv;
	 uJtheta = Fu' * X * ut;
	 vJtheta = Fv' * X * ut;
	 d2theta = ut' * X * ut;

	 H = [u2J(1,1) u2J(1,2) u2J(1,3) uvJ(1,1) uvJ(1,2) uvJ(1,3) uJtheta(1);
              u2J(2,1) u2J(2,2) u2J(2,3) uvJ(2,1) uvJ(2,2) uvJ(2,3) uJtheta(2);
	      u2J(3,1) u2J(3,2) u2J(3,3) uvJ(3,1) uvJ(3,2) uvJ(3,3) uJtheta(3);
              uvJ(1,1) uvJ(2,1) uvJ(3,1) v2J(1,1) v2J(1,2) v2J(1,3) vJtheta(1);
	      uvJ(1,2) uvJ(2,2) uvJ(3,2) v2J(2,1) v2J(2,2) v2J(2,3) vJtheta(2);
	      uvJ(1,3) uvJ(2,3) uvJ(3,3) v2J(3,1) v2J(3,2) v2J(3,3) vJtheta(3);
              uJtheta(1) uJtheta(2) uJtheta(3) ...
              vJtheta(1) vJtheta(2) vJtheta(3) ...
              d2theta];

         g = [uJ(1); uJ(2); uJ(3); vJ(1); vJ(2); vJ(3); dtheta];
	 
endfunction

%
% Check whether the iteration converges or not
%
function dist = calcDistance (F1, F2)
	 u1 = reshape (F1', 3, 3);
	 u2 = reshape (F2', 3, 3);	 
	 sign = u1' * u2;
	 if (sign < 0)
	    sign = -sign;
	    u2 = -u2;
	 end
	 dist = 1.0 - sign;
endfunction

%
% Compute Xi
%
function Xi = calcXi (X1, Y1, X2, Y2, F0)
	 Xi = zeros (9, length(X1));

	 for i = 1 : length(X1)
	     Xi(1, i) = X1(i) * X2(i);
	     Xi(2, i) = X1(i) * Y2(i);
	     Xi(3, i) = X1(i) * F0;
	     Xi(4, i) = Y1(i) * X2(i);
	     Xi(5, i) = Y1(i) * Y2(i);
	     Xi(6, i) = Y1(i) * F0;
	     Xi(7, i) = F0 * X2(i);
	     Xi(8, i) = F0 * Y2(i);
	     Xi(9, i) = F0 * F0;
	 end	   	 
endfunction

%
% Compute V0
%
function V0 = calcV0 (x1, y1, x2, y2, F0)
	 du = zeros(9, 4);
	 du(7, 1) = F0;
	 du(8, 2) = F0;
	 du(3, 3) = F0;
	 du(6, 4) = F0;
	 du(1, 1) = x1;
	 du(4, 1) = y1;
	 du(2, 2) = x1;
	 du(5, 2) = y1;
	 du(1, 3) = x2;
	 du(2, 3) = y2;
	 du(4, 4) = x2;
	 du(5, 4) = y2;
	 V0 = du * du';
endfunction

%
% Compute matrix X
%	 
function X = calcX (u, Xi, X1, Y1, X2, Y2, F0)

	 M  = zeros (9, 9);
	 N  = zeros (9, 9);

	 for i = 1 : length(X1)
	     xi = Xi(:, i);
	     A = xi * xi';
	     V0 = calcV0 (X1(i), Y1(i), X2(i), Y2(i), F0);
	     uV0u = u' * V0 * u;

	     M = M + (A / uV0u);
	     N = N + V0 * ((u' * A * u) / (uV0u * uV0u));
	 end
	 X = M - N;
endfunction

%
% Compute J
%
function J = calcJ (F, Xi, X1, Y1, X2, Y2, F0)
	 J = 0.0;
	 u = reshape (F', 9, 1);
	 
	 for i = 1 : length(X1)
	     V0 = calcV0 (X1(i), Y1(i), X2(i), Y2(i), F0);
	     J = J + ((u' * Xi(:, i)) * (u' * Xi(:, i)) / (u' * V0 * u));
	 end
endfunction

%
% CLM function
% FS: Startpoint for FM
% X1, X2, Y1, Y2: x and y values of point corresp.
% F0: Normalization factor
% maxIter: max iterations
% stopdist: min error
function u = CLM (F, X1, Y1, X2, Y2, F0, maxIter, stopDist)

	format long;

	%F
	%F0
	%maxIter	
	%stopDist

	warning('off','all');

	 iter = 0;
	 iter_max = maxIter;
	 c = 0.0001;
         eps = 1.0e-15;
	stopDist = stopDist;
	 
	 Xi = calcXi (X1, Y1, X2, Y2, F0);
	 
	 F = reshape (F, 3, 3);
	 %F = F';
	 
	%F

	 [U, S, V] = svd (F);
	 theta = asin (S(2,2) / sqrt (S(1,1) * S(1,1) + S(2,2) * S(2,2)));
	 S(1, 1) = cos (theta);
	 S(2, 2) = sin (theta);
	 S(3, 3) = 0.0;
	 F  = U * S * V';
	 F_ = eye (3, 3);
	 J = calcJ (F, Xi, X1, Y1, X2, Y2, F0); 	 
	 
	 while (iter < iter_max)
	   
	       iter = iter + 1;

	       Fu = calcFu (F);
	       Fv = calcFv (F);
	       ut = calcUt (U, S, V);
	       u  = reshape (F', 9, 1);
	       
	       X = calcX (u, Xi, X1, Y1, X2, Y2, F0);

	       [H, g] = setHandG (X, Fu, Fv, u, ut);
	       DH = diag (diag(H), 0);

	       while (1)
		     param = (H + c * DH) \ (-g);
		     omega = param(1:3, 1);
		     U_ = createRotationMatrix (omega) * U;
		     omega = param(4:6, 1);
		     V_ = createRotationMatrix (omega) * V;
		     theta_ = theta + param(7);
		     S_ = zeros(3, 3);
		     S_(1, 1) = cos (theta_);
     		     S_(2, 2) = sin (theta_);
		     F_ = U_ * S_ * V_';
		     J_ = calcJ (F_, Xi, X1, Y1, X2, Y2, F0);

		     if (J_ < J | abs (J_ - J) < eps)
			J = J_;
			c = c * 0.1;
			break;
		     else
			c = c * 10.0;
		     end
	       end
	       
	       if (calcDistance (F, F_) < stopDist)
			u = reshape (F_', 9, 1);
			break;
	       else
			F = F_;
			U = U_;
			S = S_;
			V = V_;
			theta = theta_;	       
	       end       
	 end

endfunction

%
% Least Squares
%	 
function u = LS (X1, Y1, X2, Y2, F0)

	 M  = zeros (9, 9);
	 xi = zeros (9, 1);

	 for i = 1 : length(X1),
	     xi = [X1(i) * X2(i), X1(i) * Y2(i), X1(i) * F0, ...
		   Y1(i) * X2(i), Y1(i) * Y2(i), Y1(i) * F0, ...
		   F0 * X2(i), F0 * Y2(i), F0 * F0]';

	     M = M + xi * xi';
	 end
         [v, l] = eig (M);
         u = v(:, 1);
endfunction
