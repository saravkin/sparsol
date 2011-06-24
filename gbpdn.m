function [x,info] = gbpdn(A,b,tau,sigma,x,options)
%GPBDN  Solve generalize basis pursuit, basis pursuit denoise
%
% [x,info] = gbpdn(A, b, tau, sigma, x0, options)
%
% ----------------------------------------------------------------------
% Solve the generalized basis pursuit denoise (GBPDN) problem
% or the kappa-regularized least-squares (KLS) problem:
%
% (GBPDN)   minimize  kappa(x)    subject to ||Ax-b||_2 <= sigma,
%
% (KLS)     minimize  ||Ax-b||_2  subject to  kappa(x) <= tau.
% ----------------------------------------------------------------------
%
% INPUTS
% ======
% A        is an m-by-n matrix, explicit or an operator.
%          If A is a function, then it must have the signature
%
%          y = A(x,mode)   if mode == 1 then y = A x  (y is m-by-1);
%                          if mode == 2 then y = A'x  (y is n-by-1).
%
% b        is an m-vector.
% tau      is a nonnegative scalar; see (KLS).
% sigma    if sigma != inf or != [], then GBPDN will launch into a
%          root-finding mode to find the tau above that solves (GBPDN).
%          In this case, it's STRONGLY recommended that tau = 0.
% x0       is an n-vector estimate of the solution (possibly all
%          zeros). If x0 = [], then GBPDN determines the length n via
%          n = length( A'b ) and sets  x0 = zeros(n,1).
% options  is a structure of options. Any unset options are set to
%          their default value; set options=[] to use all default
%          values. (see options section below)
%
% OUTPUTS
% =======
% x        is a solution of the problem
% info     ???TODO???
%
%
% OPTIONS
% =======
% Use the options structure to control various aspects of the algorithm:
%
% options.fid          File ID to direct log output
%        .verbosity    0   No output
%                      1   Root finding iterations
%                      1+  Root finding iterations, Lasso(verbosity-1)
%        .iterations   Maximum number of Newton root finding iterations
%        .tolerance    Maximum deviation from sigma. In case a
%                      vector of two values is given, these are the
%                      maximum deviation above and below.
%        .maxMatvec    Maximum matrix-vector multiplies allowed
%        .maxRuntime   Maximum runtime allows (in seconds)
%        .kappa        Kappa gauge function (one-norm by default)
%        .kappa_polar  Polar gauge of kappa (inf-norm by default)
%        .project      Projection onto kappa-balls
%        .solver       1   Spectral projected gradient (SPG) = default
%                      2   Projected quadratic Newton (PQN)
%        .lassoOpts
%        .rootFinder   newton -> use newton's method (default)
%                      secant -> use exact secant method (fixed precision solving)
%                      isecant -> use inexact secant method (variable precision)
%
% AUTHORS
% =======
%  Ewout van den Berg (ewout78@cs.ubc.ca)
%  Michael P. Friedlander (mpf@cs.ubc.ca)
%    Scientific Computing Laboratory (SCL)
%    University of British Columbia, Canada.
%  Aleksandr Aravkin (saravkin@eos.ubc.ca)

%   gbpdn.m
%   $Id$
%
%   ----------------------------------------------------------------------
%   This file is part of GBPDN (Generalize Basit Pursuit Denoise).
%
%   Copyright (C) 2009-2011 Ewout van den Berg, Michael P. Friedlander,
%   Department of Computer Science, and Aleksandr Aravkin, Department of
%   Earth and Ocean Sciences, University of British Columbia, Canada.
%   All rights reserved. E-mail: <{ewout78,mpf}@cs.ubc.ca>,
%   <saravkin@eos.ubc.ca>.
%
%   GBPDN is free software; you can redistribute it and/or modify it
%   under the terms of the GNU Lesser General Public License as
%   published by the Free Software Foundation; either version 2.1 of the
%   License, or (at your option) any later version.
%
%   GBPDN is distributed in the hope that it will be useful, but WITHOUT
%   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
%   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General
%   Public License for more details.
%
%   You should have received a copy of the GNU Lesser General Public
%   License along with GBPDN; if not, write to the Free Software
%   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301
%   USA
%   ----------------------------------------------------------------------

REVISION = '$Revision: 1017 $';
DATE     = '$Date: 2008-06-16 22:43:07 -0700 (Mon, 16 Jun 2008) $';
REVISION = REVISION(11:end-1);
DATE     = DATE(35:50);

% Tic-safe toc; start watches!
try, t0 = toc; catch, tic, t0 = toc; end;


%----------------------------------------------------------------------
% Check arguments.
%----------------------------------------------------------------------
if ~exist('options','var') || isempty(options), options = struct(); end;
if ~exist('sigma'  ,'var'), sigma = []; end;
if ~exist('tau'    ,'var'), tau = []; end;
if ~exist('x'      ,'var'), x = []; end;

if nargin < 2 || isempty(b) || isempty(A)
    error('At least two arguments are required');
end

defTol  = 1e-5 * norm(b,2);
options = setOptions(options, ...
    'fid'         ,        1 , ...
    'verbosity'   ,        1 , ...
    'prefix'      ,       '' , ...  % Prefix for formatting output
    'iterations'  ,       100 , ...
    'tolerance'   ,   defTol , ...
    'maxMatvec'   ,      Inf , ...
    'maxRuntime'  ,      Inf , ...
    'solver'      ,        1 , ...
    'project'     ,       [] , ...
    'kappa'       ,       [] , ...
    'kappa_polar' ,       [] , ...
    'lassoOpts'   , struct() , ...
    'rootFinder'  ,  'newton' ...
    );

% Make tolerance two sided if needed.
if isscalar(options.tolerance)
    options.tolerance = [1 1] * options.tolerance;
end

% Check kappa related options.
flags = [isempty(options.project), ...
    isempty(options.kappa)  , ...
    isempty(options.kappa_polar)];
if any(flags) && ~all(flags)
    error(['Either all kappa related fields (kappa, kappa_polar, ' ...
        'project) should be given or none.']);
end
if all(flags)
    options.project     = @(x,tau) NormL1_project(x,1,tau);
    options.kappa       = @(x)     NormL1_primal(x,1);
    options.kappa_polar = @(x)     NormL1_dual(x,1);
end

% Match tau and x0.
if ~isempty(x)
    if isempty(tau)
        tau = options.kappa(x);
    else
        x = project(x,tau);
    end
end

% Determine solver mode.
if isempty(tau) && isempty(sigma)
    tau       = 0;
    sigma     = 0;
    singleTau = false;
elseif isempty(sigma)
    singleTau = true;
else % Empty tau
    tau       = 0;
    singleTau = false;
end

%----------------------------------------------------------------------
% Initialize local variables.
%----------------------------------------------------------------------
iter         =  1; % Number of Pareto curve evaluations
stat         = []; % Solver status
statMsg      = '';
nProdA       =  0;
nProdAt      =  0;
nProjections =  0;
timeMatProd  =  0;
timeProject  =  0;
timeTotal    =  0;
bNorm        = norm(b,2);
maxMatvec    = options.maxMatvec;
maxRuntime   = options.maxRuntime;
f            = -1; % Objective (needed when dealing with matvec error)
tauHist      = [tau];
lambdaHist   = [];
fHist        = [bNorm]; % needed for secant method
dHist        = []; % needed for secant method

% Determine problem size. (May need nProdA and nProdAt)
explicit = ~(isa(A,'function_handle'));
if isempty(x)
    if explicit
        n = size(A,2);
    else
        x = Aprod(b,2);
        n = length(x);
    end
    x = zeros(n,1);
else
    n = length(x);
end
m = length(b);

% Exit conditions (constants).
EXIT_OPTIMAL       = 1; % See options.tolerance
EXIT_ITERATIONS    = 2; % See options.iterations
EXIT_MATVEC_LIMIT  = 3; % See options.maxMatvec
EXIT_ERROR         = 4; % Lasso solver error
EXIT_RUNTIME_LIMIT = 5; % See options.maxRuntime

% ----------------------------------------------------------------------
% Prepare Lasso solver
% ----------------------------------------------------------------------

% Configure user data
data = struct();
data.b           = b;
data.r           = []; % For dealing with matvec error in first call
data.Aprod       = @Aprod;
data.kappa_polar = options.kappa_polar;
data.iter        = 0; % needed for secant callback
data.dualSol     = 0; % needed for secant callback


usingNewton = strcmp(options.rootFinder, 'newton'); % for newton-only options

% Choose solver and set options





switch options.solver
    case 1,
        lassoOpts = setOptions(options.lassoOpts, ...
            'verbosity' , max(0,options.verbosity - 1), ...
            'iterations',   1e5, ...
            'optTol'    ,  1e-6);
        lassoOpts.prefix     = [options.prefix, '  |'];
        
        funLasso = @spgLasso;
        solver   = 'SPG Solver';
    case 2,
        lassoOpts = setOptions(options.lassoOpts, ...
            'verbosity' , max(0,options.verbosity - 1), ...
            'iterations',   1e5, ...
            'optTol'    ,  1e-6);
        lassoOpts.prefix     = [options.prefix, '  >'];
               
        funLasso = @pqnLasso;
        solver   = 'PQN Solver';
        
    otherwise
        error('Unknown solver in options.');
end

switch options.rootFinder
    case {'newton'}
        lassoOpts.callback   = @funCallbackNewton;
    case{'secant'}
        lassoOpts.callback   = @funCallbackSecant;
        
    case{'isecant'}
        lassoOpts.callback   = @funCallbackInexactSecant;
    otherwise
        error('Unknown root finder');
end


%----------------------------------------------------------------------
% Log header.
%----------------------------------------------------------------------
printf('\n');
printf(' %s\n',repmat('=',1,80));
printf(' GBPDN  v.%s (%s)\n', REVISION, DATE);
printf(' %s\n',repmat('=',1,80));
printf(' %-22s: %8i %4s %-22s: %8i\n','No. rows',m,'','No. columns',n);
printf(' %-22s: %8.2e      %-22s: %s\n'   ,'Two-norm of b',bNorm,'Solver',solver);


% ----------------------------------------------------------------------
% Configure output
% ----------------------------------------------------------------------
% Set log format and output header
if options.verbosity ~= 0
    logB = ' %4d  %13.7e %13.7e   %13d  %-30s\n';
    logH = '%4s  %-13s  %-15s   %-30s';
    logS = sprintf(logH,'Iter','Objective', 'Parameter','Subsolver its','Subsolver message');
    printf('\n');
    
    if options.verbosity == 1
        printf(' %s\n',logS);
    else
        bar  = repmat('=',length(logS),1);
        pref = options.prefix;
        logB = sprintf(' %s\n%s %s\n%s%s%s %s\n',...
            bar,pref,logS,pref,logB,pref,bar);
    end
else
    logB = '';
end


% ----------------------------------------------------------------------
% Quick exit if sigma >= ||b||.  Set tau = 0 to short-circuit the loop.
% ----------------------------------------------------------------------
if bNorm <= sigma
    tau = 0;  sigma = [];
end




% ----------------------------------------------------------------------
% Main loop
% ----------------------------------------------------------------------
while 1
    % Evaluate Pareto function and compute gradient
    try
        data.iter = iter;
        data.tauOld = tauHist(end);
        data.fOld   = fHist(end);
        data.tau    = tau;
        data.sigma  = sigma;
        funProject  = @(z) project(z,tau);
        lassoOpts.maxRuntime = maxRuntime - (toc - t0);
        [x,info,data] = funLasso(@funObjective,funProject,x,lassoOpts,data);
        f = data.rNorm;
        dualSol = sqrt(norm(data.b)^2 - norm(data.r - data.b)^2 - 2*tau*norm(data.Atr, inf)); %sqrt of 2*dual solution
        data.dualSol = dualSol; % needed by secant method
        
        if(usingNewton || iter == 1)
            g = -options.kappa_polar(data.Atr / data.rNorm);
        end
        %        dualSol = 0.5*norm(data.r - data.b,2)^2 + tau*norm(data.Atr, inf);
        
    catch
        err = lasterror;
        if strcmp(err.identifier,'GBPDN:MaximumMatvec')
            stat = EXIT_MATVEC_LIMIT;
            iter = iter - 1;
            info.stat    = 0;
            info.statMsg = '---ABORTED BY GPBDN---';
        else
            rethrow(err);
        end
    end
    
    % Output log
    printf(logB,iter,f,tau,info.iter,info.statMsg);
    
    % Check exit conditions
    if (toc - t0) >= maxRuntime
        stat = EXIT_RUNTIME_LIMIT;
    end
    if iter >= options.iterations
        stat = EXIT_ITERATIONS;
    end
    if ischar(info.stat)
        stat    = EXIT_ERROR;
        statMsg = info.stat;
    else
        if isempty(sigma) || ...
                (((f - sigma) <= options.tolerance(1)) && ...
                ((sigma - f) <= options.tolerance(2)))
            stat = EXIT_OPTIMAL;
        end
    end
    if ~isempty(stat), break; end;
    
    % Take Newton step.
    tauHist    = [tauHist, tau];
    
    if(usingNewton)
        lambdaHist = [lambdaHist, -g * data.rNorm];
    else
        fHist = [fHist, f]; % needed for secant method
        dHist = [dHist, dualSol];
    end
    
    
    switch options.rootFinder
        case{'newton'}
            tau  = tau - (f - sigma) / g;
        case{'secant'}
            if(iter == 1)
                tau  = tau - (f - sigma) / g;
            else
                dtau = tauHist(end) - tauHist(end - 1);
                slope = (fHist(end) - fHist(end - 1))/dtau;
                tau = tau - (fHist(end) - sigma)/(slope);
            end
            
        case{'isecant'}
            if(iter == 1)
                tau  = tau - (f - sigma) / g;
            else
                dtau = tauHist(end) - tauHist(end - 1);
                slope = (dHist(end) - fHist(end - 1))/dtau; % using dual solution
                tau = tau - (dHist(end) - sigma)/(slope); %using dual solution
            end
        otherwise
            err('unknown root finding method')
    end
    iter = iter + 1;
    
end
% ----------------------------------------------------------------------
% End of main loop
% ----------------------------------------------------------------------


% Create info structure
info = struct();
info.tau          = tau;
info.iter         = iter;
info.f            = f;
info.r            = data.r;
info.stat         = stat;
info.statMsg      = statMsg;
info.timeTotal    = toc - t0;
info.timeProject  = timeProject;
info.timeMatProd  = timeMatProd;
info.nProdA       = nProdA;
info.nProdAt      = nProdAt;
info.nProjections = nProjections;
info.tauHist      = [tauHist, tau];
info.lambdaHist   = [lambdaHist, -g * data.rNorm];

% Print final output.
switch (stat)
    case EXIT_OPTIMAL
        info.statMsg = 'Optimal solution found';
    case EXIT_ITERATIONS
        info.statMsg = 'Too many iterations';
    case EXIT_MATVEC_LIMIT
        info.statMsg = 'Maximum matrix-vector operations reached';
    case EXIT_RUNTIME_LIMIT
        info.statMsg = 'Maximum runtime reached';
    case EXIT_ERROR
        info.statMsg = sprintf('Lasso error: %s', statMsg);
    otherwise
        info.statMsg = 'Unknown termination condition';
end

printf('\n');
printf(' EXIT -- %s\n', info.statMsg)
printf('\n');
printf(' %-20s:  %6i %6s %-20s:  %6.1f\n',...
    'Products with A',nProdA,'','Total time   (secs)',info.timeTotal);
printf(' %-20s:  %6i %6s %-20s:  %6.1f\n',...
    'Products with A''',nProdAt,'','Project time (secs)',timeProject);
printf(' %-20s:  %6i %6s %-20s:  %6.1f\n',...
    'Newton iterations',info.iter,'','Mat-vec time (secs)',timeMatProd);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NESTED FUNCTIONS.  These share some vars with workspace above.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ----------------------------------------------------------------------
    function z = Aprod(x,mode)
        % ----------------------------------------------------------------------
        if (nProdA + nProdAt >= maxMatvec)
            error('GBPDN:MaximumMatvec','');
        end
        
        tStart = toc;
        if mode == 1
            nProdA = nProdA + 1;
            if   explicit, z = A*x;
            else           z = A(x,1);
            end
        elseif mode == 2
            nProdAt = nProdAt + 1;
            if   explicit, z = A'*x;
            else           z = A(x,2);
            end
        else
            error('Wrong mode!');
        end
        timeMatProd = timeMatProd + (toc - tStart);
    end % function Aprod


% ----------------------------------------------------------------------
    function printf(varargin)
        % ----------------------------------------------------------------------
        if options.verbosity > 0
            fprintf(options.fid,options.prefix);
            fprintf(options.fid,varargin{:});
        end
    end % function printf


% ----------------------------------------------------------------------
    function x = project(x, tau)
        % ----------------------------------------------------------------------
        tStart      = toc;
        
        x = options.project(x,tau);
        
        timeProject  = timeProject + (toc - tStart);
        nProjections = nProjections + 1;
    end % function project


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of nested functions.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end % function gbpdn



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PRIVATE FUNCTIONS.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ----------------------------------------------------------------------
function [f,varargout] = funObjective(x,varargin)
% ----------------------------------------------------------------------
% [f]        = funObjective(x,...)
% [f,data]   = funObjective(x,...);
% [f,g,data] = funObjective(x,...);

% Initialize data
if nargin > 1
    data = varargin{1};
else
    data = struct();
end;

% Compute f
data.r     = data.b - data.Aprod(x,1);
data.rNorm = norm(data.r,2);
data.f     = data.r' * data.r / 2;
f          = data.f;

% Compute gradient (optional)
if nargout == 3
    data.Atr     = data.Aprod(data.r,2);
    varargout{1} = -data.Atr; % gradient
else
    data.Atr = [];
end

% Output data
if nargout > 1
    varargout{nargout-1} = data;
end

end % function funObjective


% ----------------------------------------------------------------------
function [stat,data] = funCallbackInexactSecant(x,data)
% ----------------------------------------------------------------------

% Default output
stat = 0;

% Compute primal dual gap (in quadratic formulation)
if ~isempty(data.Atr)
    gNorm = data.kappa_polar(data.Atr);
    gap   = data.r'*(data.r - data.b) + data.tau*gNorm;
    rGap  = abs(gap) / max(1,data.f);
    
    tau    = data.tau;
    tauOld = data.tauOld;
    fOld   = data.fOld;
    dualSol = data.dualSol;
    
    
    
    if (rGap <= 1e-5*(tau - tauOld)) && (dualSol <= fOld)
        stat = 1;
    end
end


end % function funCallbackInexactSecant


% ----------------------------------------------------------------------
function [stat,data] = funCallbackSecant(x,data)
% ----------------------------------------------------------------------

% Default output
stat = 0;

% Compute primal dual gap (in quadratic formulation)
if ~isempty(data.Atr)
    gNorm = data.kappa_polar(data.Atr);
    gap   = data.r'*(data.r - data.b) + data.tau*gNorm;
    rGap  = abs(gap) / max(1,data.f);
    
    tau    = data.tau;
    tauOld = data.tauOld;
    
    
    if rGap <= 1e-5*(tau - tauOld)
        stat = 1;
    end
end


end % function funCallbackSecant

% ----------------------------------------------------------------------
function [stat,data] = funCallbackNewton(x,data)
% ----------------------------------------------------------------------

% Default output
stat = 0;

% Compute primal dual gap (in quadratic formulation)
if ~isempty(data.Atr)
    gNorm = data.kappa_polar(data.Atr);
    gap   = data.r'*(data.r - data.b) + data.tau*gNorm;
    rGap  = abs(gap) / max(1,data.f);
    
    if rGap <= 1e-10 % TODO
        stat = 1;
    end
end


end % function funCallbackNewton
