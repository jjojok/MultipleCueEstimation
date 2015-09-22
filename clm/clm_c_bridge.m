%
% main routine
%

source ("clm.m");

function u = SPLM (X1, Y1, X2, Y2)

	format long;

	u0 = LS (X1, Y1, X2, Y2, 600.0);
	u = CLM (X1, Y1, X2, Y2, 600.0, u0);

	%u
	%
	%F = reshape(u, 3, 3);
	%F /= F(2,2)
	%F = F.'

endfunction
