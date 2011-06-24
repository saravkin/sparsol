function [x,info,data] = spgLasso(funObj, funProj, x0, options, data)
%SPGLASSO
%
% [x,info,data] = spgLasso(funObj, funProj, x0, options, data)
%
% solves the convex constrained problem
%
%   minimize  funObj(x)  subj to  x in C
%
% where C is a convex set implicitly defined by funProj.

%
% options
% .verbosity    0 = No output
%               1 = Major output
%               2 = All output

% TODO: check handling of complex numbers; search `real'

REVISION = '$Revision: xxxx $';
DATE     = '$Date: xxxx-xx-xx xx:xx:xx -xxxx (Today) $';
REVISION = REVISION(11:end-1);
DATE     = DATE(35:end-3);

% Ouch! Tic-safe toc
try t0 = toc; catch, tic, t0 = toc; end;

%----------------------------------------------------------------------
% Check arguments. 
%----------------------------------------------------------------------
if nargin < 3, error('Too few input arguments'); end
if nargin < 4 || ~exist('options','var'), options = struct(); end;
if nargin < 5 || ~exist('data'   ,'var'), data    = [];       end;

n = length(x0);

% Process options
options = setOptions(options, ...
  'fid'       ,     1 , ... % File ID for output
  'prefix'    ,    '' , ... % Output prefix
  'verbosity' ,     2 , ... % Verbosity level
  'iterations',  10*n , ... % Max number of iterations
  'nPrevVals' ,     3 , ... % Number previous func values for linesearch
  'optTol'    ,  1e-5 , ... % Optimality tolerance in x
  'dxTol'     ,     0 , ... % Optimality tolerance in dx
  'maxFunEval',   Inf , ... % Maximum number of function evaluations
  'maxRuntime',   Inf , ... % Maximum runtime allowed (in seconds)
  'callback'  ,    [] , ... % Callback function for optimality test
  'stepMin'   , 1e-16 , ... % Minimum spectral step
  'stepMax'   , 1e+05   ... % Maximum spectral step
);
fid           = options.fid;
logLevel      = options.verbosity;
maxIts        = options.iterations;
maxRuntime    = options.maxRuntime;
nPrevVals     = options.nPrevVals;
optTol        = options.optTol;
dxTol         = options.dxTol;
stepMin       = options.stepMin;
stepMax       = options.stepMax;
prefix        = options.prefix;

maxLineErrors = 10;   % Maximum number of line-search failures.

%----------------------------------------------------------------------
% Initialize local variables.
%----------------------------------------------------------------------
iter          = 0; % Total iterations.
stat          = 0;
nProjections  = 0; % Number of projections
nFunctionEval = 0; % Number of function evaluations
nGradientEval = 0; % Number of gradient evaluations
nLineTot      = 0; % Total number of linesearch steps
timeProject   = 0; % Projection time
timeFunEval   = 0; % Function/gradient evaluation time
lastFv        = -inf(nPrevVals,1);  % Last m function values
usecallback   = ~isempty(options.callback);

% Exit conditions (constants).
EXIT_OPTIMAL          = 1;
EXIT_ITERATIONS       = 2;
EXIT_LINE_ERROR       = 3;
EXIT_FUNEVAL_LIMIT    = 4;
EXIT_CALLBACK         = 5;
EXIT_RUNTIME_LIMIT    = 6;

%----------------------------------------------------------------------
% Log header.
%----------------------------------------------------------------------
logB = '%s %5i  %15.9e  %9.2e  %6.1f\n';
logH = '%s %5s  %13s  %9s  %6s\n';
if logLevel > 0
   fprintf(fid,'%s\n', prefix);
   fprintf(fid,'%s %s\n', prefix,repmat('=',1,80));
   fprintf(fid,'%s SPG-Lasso  v.%s (%s)\n', prefix, REVISION, DATE);
   fprintf(fid,'%s %s\n', prefix, repmat('=',1,80));
   fprintf(fid,'%s %-22s: %8.2e %4s %-22s: %8i\n', prefix, ...
               'Optimality tol',optTol,'','Maximum iterations',maxIts);
end

if logLevel > 1
   fprintf(fid,'%s \n', prefix);
   fprintf(fid, logH, prefix,'Iter','Objective','gNorm','stepG');
end

%----------------------------------------------------------------------
% Final preparations.
%----------------------------------------------------------------------

% Project the starting point and evaluate function and gradient.
x          = project(x0);
[f,g,data] = objective(x,data);

% Required for nonmonotone strategy.
lastFv(1) = f;
fBest     = f;
xBest     = x;
fOld      = f;

% Compute projected gradient direction and initial steplength.
dx     = project(x - g) - x;
dxNorm = norm(dx,inf);
dx0Norm = dxNorm;
gStep   = 1;


%----------------------------------------------------------------------
% MAIN LOOP.
%----------------------------------------------------------------------
while 1

    %------------------------------------------------------------------
    % Test exit conditions.
    %------------------------------------------------------------------
    if usecallback
       [cbstat,data] = options.callback(x,data);
       if cbstat ~= 0
          stat = EXIT_CALLBACK;
       end
    end

    if (toc - t0) > maxRuntime
       stat = EXIT_RUNTIME_LIMIT;
    end

    if iter >= maxIts
       stat = EXIT_ITERATIONS;
    end
    
    if dxNorm < optTol * max(1,norm(x,2));
       stat = EXIT_OPTIMAL;
    end

    if dxNorm < dxTol * dx0Norm
       stat = EXIT_OPTIMAL;
    end
    
    %------------------------------------------------------------------
    % Print log, update history and act on exit conditions.
    %------------------------------------------------------------------
    if logLevel > 1 || (stat ~= 0 && logLevel > 0)
       fprintf(fid, logB, prefix,iter,f,dxNorm,log10(gStep));
    end

    if stat ~= 0, break; end % Act on exit conditions.


    %==================================================================
    % Iterations begin here.
    %==================================================================
    iter = iter + 1;
    xOld = x;  fOld = f;  gOld = g;

    try
       %---------------------------------------------------------------
       % Projected gradient step and linesearch.
       %---------------------------------------------------------------
       [f,x,nLine,stepG,lnErr,data] = ...
          spgLineCurvy(x,gStep*g,max(lastFv),@objective,@project,data);
       nLineTot = nLineTot + nLine;
       if lnErr
          %  Projected backtrack failed. Retry with feasible dir'n linesearch.
          x    = xOld;
          dx   = project(x - gStep*g) - x;
          gtd  = real(g'*dx);
          [f,x,nLine,lnErr,data] = spgLine(x,dx,gtd,f,max(lastFv),@objective,data);
          nLineTot = nLineTot + nLine;
       end       
       if lnErr
          stat = EXIT_LINE_ERROR;
       end
       
       %---------------------------------------------------------------
       % Update gradient and compute new Barzilai-Borwein scaling.
       %---------------------------------------------------------------
       [f,g,data] = objective(x,data);
       s    = x - xOld;
       y    = g - gOld;
       sts  = s'*s;
       sty  = s'*y;
       if   sty <= 0,  gStep = stepMax;
       else            gStep = min( stepMax, max(stepMin, sts/sty) );
       end

       
    catch % Detect function evaluation limit error
       err = lasterror;
       if strcmp(err.identifier,'SPG:MaximumFunEval')
         stat = EXIT_FUNEVAL_LIMIT;
         iter = iter - 1;
         
         % Restore previous iterate
         x = xOld;  f = fOld;  g = gOld;
         break;
       else
         rethrow(err);
       end
    end

    %------------------------------------------------------------------
    % Compute projected gradient
    %------------------------------------------------------------------
    dx     = project(x - g) - x;
    dxNorm = norm(dx,2);
    
    %------------------------------------------------------------------
    % Update function history.
    %------------------------------------------------------------------
    lastFv(mod(iter,nPrevVals)+1) = f;
    if fBest > f
       fBest = f;
       xBest = x;
    end
    
end % while 1

% Restore best iterate
if f > fBest
   if logLevel > 0
      fprintf(fid,'%s\n', prefix);
      fprintf(fid,'%s Restoring best iterate to objective %13.7e\n', ...
              prefix,fBest);
   end   
   x = xBest;
   f = fBest;
end

info = struct();
info.iter          = iter;
info.stat          = stat;
info.timeTotal     = toc - t0;
info.timeProject   = timeProject;
info.timeFunEval   = timeFunEval;
info.nFunctionEval = nFunctionEval;
info.nGradientEval = nGradientEval;
info.nProjections  = nProjections;

% Print final output.
switch (stat)
   case EXIT_OPTIMAL
      info.statMsg = 'Optimal solution found';
   case EXIT_ITERATIONS
      info.statMsg = 'Too many iterations';
   case EXIT_FUNEVAL_LIMIT
      info.statMsg = 'Maximum function evaluations reached';
   case EXIT_LINE_ERROR
      info.statMsg = 'Line search error';
   case EXIT_CALLBACK
      info.statMsg = 'Callback function exit';
   case EXIT_RUNTIME_LIMIT
      info.statMsg = 'Maximum runtime reached';
   otherwise
      info.statMsg = 'Unknown termination condition';
end

if logLevel > 0
   fprintf(fid,'%s\n', prefix);
   fprintf(fid,'%s EXIT -- %s\n', prefix, info.statMsg)
   fprintf(fid,'%s\n', prefix);
   fprintf(fid,'%s %-20s:  %6i %6s %-20s:  %6.1f\n', prefix, ...
           'Function evals.',nFunctionEval,'', ...
           'Total time   (secs)',info.timeTotal);
   fprintf(fid,'%s %-20s:  %6i %6s %-20s:  %6.1f\n', prefix, ...
           'Gradient evals.',nGradientEval,'', ...
           'Project time (secs)', timeProject);
   fprintf(fid,'%s %-20s:  %6i %6s %-20s:  %6.1f\n', prefix, ...
           'Iterations',info.iter,'', ...
           'Fun eval time (secs)',timeFunEval);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NESTED FUNCTIONS.  These share some vars with workspace above.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% ----------------------------------------------------------------------
function varargout = objective(varargin)
% ----------------------------------------------------------------------
   if (nFunctionEval >= options.maxFunEval)
     error('SPG:MaximumFunEval','');
   end
   tStart = toc;

   if nargout == 1
      varargout{1} = funObj(varargin{:});
   elseif nargout == 2
      [varargout{1},varargout{2}] = funObj(varargin{:});
   else
      [varargout{1},varargout{2},varargout{3}] = funObj(varargin{:});
      nGradientEval = nGradientEval + 1;
   end
   
   nFunctionEval = nFunctionEval + 1;
   
   timeFunEval = timeFunEval + (toc - tStart);
end % function objective

% ----------------------------------------------------------------------
function x = project(x)
% ----------------------------------------------------------------------
   tStart       = toc;
   x            = funProj(x);
   timeProject  = timeProject + (toc - tStart);
   nProjections = nProjections + 1;
end % function project


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of nested functions.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end % spgGeneral


% ----------------------------------------------------------------------
function [fNew,xNew,iter,stat,data] = spgLine(x,d,gtd,f,fMax,funObj,data)
% ----------------------------------------------------------------------
% Nonmonotone linesearch.

EXIT_CONVERGED  = 0;
EXIT_ITERATIONS = 1;

maxIts = 20;
step   =  1;
iter   =  0;
gamma  = 1e-4;

while 1
    % Evaluate trial point and function value.
    xNew = x + step*d;
    [fNew,data] = funObj(xNew,data);

    % Check exit conditions.
    if fNew < fMax + gamma*step*gtd  % Sufficient descent condition.
       stat = EXIT_CONVERGED;
       break
    elseif iter >= maxIts           % Too many linesearch iterations.
       stat = EXIT_ITERATIONS;
       break
    end

    % New linesearch iteration.
    iter = iter + 1;
    
    % Safeguarded quadratic interpolation.
    if step <= 0.1
       step  = step / 2;
    else
       tmp = (-gtd*step^2) / (2*(fNew-f-step*gtd));
       if tmp < 0.1 || tmp > 0.9*step || isnan(tmp)
          tmp = step / 2;
       end
       step = tmp;
    end
end % while 1

end % function spgLine


% ----------------------------------------------------------------------
function [fNew,xNew,iter,step,stat,data] = spgLineCurvy(x,g,f,funObj,funProj,data)
% ----------------------------------------------------------------------
% Projected backtracking linesearch.
% On entry d is the (possibly scaled) steepest descent direction.

EXIT_CONVERGED  = 0;
EXIT_ITERATIONS = 1;

maxIts = 20;
step   =  1;
sNorm  =  1;
scale  =  1; % Safeguard scaling (see below).
iter   =  0;
gamma  =  1e-4;

while 1
    % Evaluate trial point and function value.
    xNew        = funProj(x - step*scale*g);
    [fNew,data] = funObj(xNew,data);
    s           = xNew - x;
    gts         = scale * real(g' * s); % TODO: COMPLEX???

    if gts < 0
       if fNew < f - gamma*step*gts
          stat = EXIT_CONVERGED;
          break;
       end
    end

    if iter >= maxIts
       stat = EXIT_ITERATIONS;
       break;
    end      

    % New linesearch iteration.
    iter = iter + 1;
    step = step / 2;

    % Safeguard: If stepMax is huge, then even damped search directions
    % can give exactly the same point after projection.  If we observe
    % this in adjacent iterations, we drastically damp the next search
    % direction.
    % TODO: MODIFIED!!!!
    sNorm = norm(s,Inf);
    if sNorm <= 1e-6 * max(norm(x,2),1);
       scale = scale / 10;
    end
end % while 1

end % spgLineCurvy
