% initialize symbolic coefficients
syms EA EI EAI EAS EASI

% define A matrix with parametric entries
syms f1 r1 f2 r2 f3 f4 r4  % rate constants
syms S P I At Et % constants
A = [-(f1*At + f2*S + f4*I), -f1*At, (r4 - f1*At), (f3 + r2 - f1*At), -f1*At;
    -f4*I, -(f4*I + r4 + f1*At), (r1 - f4*I), -f4*I, -f4*I;
    f2*S, 0, 0, -(r2 + f4*I), r4;
    f4*I, f1*At, -(r1 + r4 + f2*S), 0, r2;
    0, 0, f2*S, f4*I, 0]

% RHS vector
b = [-f1*At*Et; -f4*I*Et; 0; 0; 0];

% SOLVE Ax=b
x = linsolve(A,b)
% Notes:
%   20230105: This does provide a solution! However the expressions are
%   extremely complex. 
%   TODO: write expressions on paper and try to simplify