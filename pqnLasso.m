function [x,info,data] = pqnLasso(funObj,funProj,x0,options,data)
%PQNLASSO
%
% [x,info,data] = pqnLasso(funObj, funProj, x0, options, data)
%
% ---------------------------------------------------------------------
% Solve
%
%         minimize  funObj(x)  subj to  x in C
%
% Where feasible set C is implicitly defined by funProj.
% ---------------------------------------------------------------------
%

%
% options
% .verbosity    0 = No output
%               1 = Major output
%               2 = All output
%               3+= All output, verbosity-2 subproblem output

% TODO: replace constants by parameters

REVISION = '$Revision: xxxx $';
DATE     = '$Date: xxxx-xx-xx xx:xx:xx -xxxx (Today) $';
REVISION = REVISION(11:end-1);
DATE     = DATE(35:end-3);

% Tic-safe toc; start watches!
try, t0 = toc; catch, tic, t0 = toc; end;

%----------------------------------------------------------------------
% Check arguments. 
%----------------------------------------------------------------------
if ~exist('options','var'), options = struct(); end;
if ~exist('data'   ,'var'), data    = [];       end;

% Process options
n = length(x0);
options = setOptions(options, ...
  'fid'       ,    1 , ... % File ID for output
  'prefix'    ,   '' , ... % Output prefix
  'verbosity' ,    2 , ... % Verbosity level
  'iterations', 10*n , ... % Max number of iterations
  'optTol'    , 1e-5 , ... % Optimality tolerance
  'lbfgsHist' ,   10 , ... % Number of vectors used in L-BFGS
  'maxFunEval',  Inf , ... % Maximum number of function evaluations
  'maxRuntime',  Inf , ... % Maximum runtime allowed (in seconds)
  'callback'  ,   []   ... % Callback function for optimality test
);

fid         = options.fid;
logLevel    = options.verbosity;
maxIts      = options.iterations;
maxRuntime  = options.maxRuntime;
lbfgsHist   = options.lbfgsHist;
optTol      = options.optTol;
prefix      = options.prefix;


%----------------------------------------------------------------------
% Initialize local variables.
%----------------------------------------------------------------------
iter          = 0; % Total PQN iterations.
stat          = 0;
nProjections  = 0; % Number of projections
nFunctionEval = 0; % Number of function evaluations
nGradientEval = 0; % Number of gradient evaluations
timeProject   = 0; % Projection time
timeFunEval   = 0; % Function/gradient evaluation time

timeSubTotal = 0; % Total time for subproblem
timeSubEval  = 0; % Function evaluation time for subproblem
timeSubProj  = 0; % Projection time for subproblem

% Prepare Hessian for warm start (if available)
H = getDefaultField(options,'H',lbfgsinit(n,lbfgsHist));

% Exit conditions (constants).
EXIT_OPTIMAL          = 1;
EXIT_ITERATIONS       = 2;
EXIT_SEARCH_DIRECTION = 3;
EXIT_FUNEVAL_LIMIT    = 4;
EXIT_CALLBACK         = 5;
EXIT_LINE_SEARCH      = 6;
EXIT_RUNTIME_LIMIT    = 7;


%----------------------------------------------------------------------
% Log header.
%----------------------------------------------------------------------
logB = '%s %5i  %13.7e  %13.7e  %1s %2d %3d  %-30s\n';
logH = '%s %5s  %13s  %13s  %1s %2s %3s  %-30s\n';

if logLevel > 0
   fprintf(fid,'%s\n', prefix);
   fprintf(fid,'%s %s\n', prefix, repmat('=',1,80));
   fprintf(fid,'%s PQN-Lasso  v.%s (%s)\n', prefix, REVISION, DATE);
   fprintf(fid,'%s %s\n', prefix, repmat('=',1,80));
   fprintf(fid,'%s %-22s: %8.2e %4s %-22s: %8i\n', prefix, ...
           'Optimality tol',optTol,'','Maximum iterations',maxIts);
end
   
if logLevel > 1
   fprintf(fid,'%s\n', prefix);
   fprintf(fid,logH, prefix, ...
           'Iter','Objective','gNorm','H','PG','SPG','Lasso message');
end

%----------------------------------------------------------------------
% Final preparations.
%----------------------------------------------------------------------

% Project the starting point and evaluate function and gradient.
alpha      = 1;
x          = project(x0);
[f,g,data] = objective(x,data);

% TODO: clean up
spgInfo.statMsg = '------';
spgInfo.iter    = 0;
useSearchDir = 0;
HStatus = {'-','m','x'}; % Updated, corrected, not updated

%----------------------------------------------------------------------
% MAIN LOOP.
%----------------------------------------------------------------------
while 1

   %------------------------------------------------------------------
   % Test exit conditions.
   %------------------------------------------------------------------
   pg     = project(x - g) - x;
   pgNorm = norm(pg,Inf);

   if ~isempty(options.callback)
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

   if pgNorm < optTol * max(1,norm(x,2));
      stat = EXIT_OPTIMAL;
   end

   % Increment iteration number
   iter = iter + 1;
    
   %------------------------------------------------------------------
   % Print log, update history and act on exit conditions.
   %------------------------------------------------------------------
   if logLevel > 1 || (stat ~= 0 && logLevel > 0)
      fprintf(fid,logB, prefix, iter,f,pgNorm,HStatus{H.status+1}, ...
              ~useSearchDir, spgInfo.iter, spgInfo.statMsg);
   end   
    
   if stat ~= 0, break; end % Act on exit conditions.
   
   % Compute search direction
   if iter == 1
      d = pg;
      useSearchDir = true;
   else
      % Update Hessian approximation
      if useSearchDir
         H = lbfgsupdate(H, 1, x - xOld, gOld, g);
      end
       
      % Solve subproblem
      funProjSub = funProj;
      funObjSub  = @pqnLassoQuadFun;

      % Set up subproblem data and options
      spgData = struct();
      spgData.x       = x;
      spgData.f       = f;
      spgData.g       = g;       
      spgData.funH    = @(x) lbfgsbprod(H,x);
      spgData.funProj = @project;
       
      spgOptions = struct();
      spgOptions.optTol     = 0; 
      spgOptions.dxTol      = 0.5;
      spgOptions.iterations = 80;
      spgOptions.maxRuntime = maxRuntime - (toc - t0);
      spgOptions.verbosity  = max(0,options.verbosity-2);
      spgOptions.prefix     = [options.prefix,'     |'];

      % Call subproblem solver
      [spgx,spgInfo,spgData] = spgLasso(funObjSub,funProjSub,x,spgOptions,spgData);
      d = spgx - x;
      
      % Accumulate time information
      timeSubTotal = timeSubTotal + spgInfo.timeTotal;
      timeSubEval  = timeSubEval  + spgInfo.timeFunEval;
      timeSubProj  = timeSubProj  + spgInfo.timeProject;
       
      % Check result
      useSearchDir = true;

      if (spgInfo.stat == 3) && spgInfo.iter < 10
          % If SPG_LINE_ERROR and small number of iterations we do
          % not use the resulting search direction. In case the
          % number of iterations is large enough the check below on
          % descent direction suffices.
          useSearchDir = false;
      end
       
      if spgInfo.iter == 0 ||  norm(d) == 0
         % In this case the projected norm is immediately small
         % enough in terms of the first gradient. This condition
         % arises mainly towards the end of the solve. Force use
         % of projected gradient search direction.
         useSearchDir = false;
      end

      % Note: Because the gradient of the quadratic model
      % coincides with the current gradient, any point lower than
      % the current one (on the model) should be a valid descent
      % direction.
   end

   % Check if search direction d is descent direction
   % gtd = real(g'*d);
   % if -gtd < 1e-4 * (norm(g,2)*norm(d,2)) % Approx. 5.7e-3 degrees
   %    useSearchDir = false;
   % end

   % Fall back to projected gradient search direction in case of
   % failure of the subproblem or failure to find a direction of
   % descent. We only allow a limited number of projected gradient
   % iterations, so backtracking line search suffices for now.
   if ~useSearchDir 
     d = pg; % Projected gradient
   end
   
   % -------------------------------------------------------------------
   % Line search 
   % -------------------------------------------------------------------

   % Store previous iteration
   xOld = x;  fOld = f;  gOld = g;
   
   % Do backtracking linesearch
   alpha = 1; lsIter = 1;
   while 1
      % Note: The solution to the subproblem ensures that x+d is
      % within the domain, so no projection is required here.
      x = xOld + alpha * d;

      if lsIter == 1
          % It seems a step length of one is often acceptable,
          % avoid another function evaluation below by also
          % requesting the gradient at the first line search
          % iteration.
          [f,g,data] = objective(x,data);
      else
          [f,data] = objective(x,data);
      end
   
      dx = x - xOld;

      % Check sufficient descent
      if f <= fOld + 1e-3 * gOld' * dx
         break;
      end      
   
      % Reduce step length
      alpha = alpha / 2; lsIter = lsIter + 1;
      if alpha < 1e-6
        stat = EXIT_LINE_SEARCH;
        break;
      end
   end
   
   % Update gradient
   if lsIter > 1
      [f,g,data] = objective(x,data);
   end
   
   
   
   % HACK
   if (iter > 1000) && (spgInfo.iter < 3)
     fprintf('Resetting Hessian!!!\n');
      H.delta = 1;
      H.gamma = 1;
   end
   
% $$$    if (iter > 1000) && (spgInfo.iter < 3)
% $$$       spgData = data;
% $$$        
% $$$       spgOptions = struct();
% $$$       spgOptions.optTol     = 0; 
% $$$       spgOptions.dxTol      = 0.5;
% $$$       spgOptions.iterations = 100;
% $$$       spgOptions.verbosity  = 2;
% $$$       spgOptions.prefix     = [options.prefix,'     |'];
% $$$ 
% $$$       [spgx,spgInfo,spgData] = spgLasso(funObj,funProj,x,spgOptions,spgData);
% $$$       
% $$$       x = spgx;
% $$$       fprintf('-------------------------------------------------------------\n');
% $$$    end
   
   
end % while 1


info = struct();
info.iter          = iter;
info.stat          = stat;
info.timeTotal     = toc - t0;
info.timeProject   = timeProject;
info.timeFunEval   = timeFunEval;
info.nFunctionEval = nFunctionEval;
info.nGradientEval = nGradientEval;
info.nProjections  = nProjections;
info.H             = H;

% Print final output.
switch (stat)
   case EXIT_OPTIMAL
      info.statMsg = 'Optimal solution found';
   case EXIT_ITERATIONS
      info.statMsg = 'Too many iterations';
   case EXIT_FUNEVAL_LIMIT
      info.statMsg = 'Maximum function evaluations reached';
   case EXIT_SEARCH_DIRECTION
      info.statMsg = 'Invalid search direction';
   case EXIT_CALLBACK
      info.statMsg = 'Callback function exit';
   case EXIT_LINE_SEARCH
      info.statMsg = 'Line search error';
   case EXIT_RUNTIME_LIMIT
      info.statMsg = 'Maximum runtime reached';
   otherwise
      info.statMsg = 'Unknown termination condition';
end

if logLevel > 0
   fprintf(fid,'%s\n', prefix);
   fprintf(fid,'%s EXIT -- %s\n', prefix, info.statMsg)
   fprintf(fid,'%s\n', prefix);
   fprintf(fid,'%s %-20s:  %6i %2s %-20s:  %6.1f/%-6.1f\n', prefix, ...
           'Function evals.',nFunctionEval,'',...
           'Total time   (secs)',info.timeTotal, timeSubTotal);
   fprintf(fid,'%s %-20s:  %6i %2s %-20s:  %6.1f/%-6.1f\n', prefix, ...
           'Gradient evals.',nGradientEval,'',...
           'Project time (secs)',timeProject,timeSubProj);
   fprintf(fid,'%s %-20s:  %6i %2s %-20s:  %6.1f/%-6.1f\n', prefix, ...
           'Iterations',info.iter,'',...
           'Fun eval time (secs)',timeFunEval,timeSubEval);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NESTED FUNCTIONS.  These share some vars with workspace above.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% ----------------------------------------------------------------------
function varargout = objective(varargin)
% ----------------------------------------------------------------------
   if (nFunctionEval >= options.maxFunEval)
     error('PQN:MaximumFunEval','');
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
function printf(varargin)
% ----------------------------------------------------------------------
  if logLevel > 0
     fprintf(fid,options.prefix);
     fprintf(fid,varargin{:});
  end
end % function printf


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

end % function pqnLasso



% ----------------------------------------------------------------------
function [f,varargout] = pqnLassoQuadFun(x,data)
% ----------------------------------------------------------------------

% Compute f
x  = x - data.x; % Point x is relative to data.x
Ax = data.funH(x);
f  = x'*Ax/2 + x'*data.g + data.f; % Quadratic model

% Compute gradient (optional)
if nargout == 3
  varargout{1} = Ax + data.g; % gradient;
end

% Output data
if nargout > 1
   varargout{nargout-1} = data;
end

end % function pqnLassoQuadFun
