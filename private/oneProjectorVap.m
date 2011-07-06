function [x,itn] = oneProjectorVap(b,d,tau,eps)
% ONEPROJECTOR  Projects b onto the weighted one-norm ball of radius tau
%
%    [X,ITN] = ONEPROJECTOR(B,TAU) returns the orthogonal projection
%    of the vector b onto the one-norm ball of radius tau. The return
%    vector X which solves the problem
%
%            minimize  ||b-x||_2  st  ||x||_1 <= tau.
%               x
%
%    [X,ITN] = ONEPROJECTOR(B,D,TAU) returns the orthogonal
%    projection of the vector b onto the weighted one-norm ball of
%    radius tau, which solves the problem
%
%            minimize  ||b-x||_2  st  || Dx ||_1 <= tau.
%               x
%
%    If D is empty, all weights are set to one, i.e., D = I.


%    [X,ITN] = ONEPROJECTOR(B,D,TAU, eps) returns the orthogonal
%    projection of the vector b onto the weighted vapnik ball of
%    radius tau, which solves the problem
%
%            minimize  ||b-x||_2  st  vapnik( Dx ) <= tau.
%               x
%
%    If D is empty, all weights are set to one, i.e., D = I.
%
%    In all cases, the return value ITN given the number of elements
%    of B that were thresholded.
%
% See also spgl1.

%   oneProjector.m
%   $Id: oneProjector.m 1200 2008-11-21 19:58:28Z mpf $
%
%   ----------------------------------------------------------------------
%   This file is part of SPGL1 (Spectral Projected Gradient for L1).
%
%   Copyright (C) 2007 Ewout van den Berg and Michael P. Friedlander,
%   Department of Computer Science, University of British Columbia, Canada.
%   All rights reserved. E-mail: <{ewout78,mpf}@cs.ubc.ca>.
%
%   SPGL1 is free software; you can redistribute it and/or modify it
%   under the terms of the GNU Lesser General Public License as
%   published by the Free Software Foundation; either version 2.1 of the
%   License, or (at your option) any later version.
%
%   SPGL1 is distributed in the hope that it will be useful, but WITHOUT
%   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
%   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General
%   Public License for more details.
%
%   You should have received a copy of the GNU Lesser General Public
%   License along with SPGL1; if not, write to the Free Software
%   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301
%   USA
%   ----------------------------------------------------------------------

% Check arguments
if nargin < 2
  error('The oneProjector function requires at least two parameters');
end
if nargin < 3
  tau = d;
  d   = [];
end
if nargin < 4
    eps = 0;
end

% Check weight vector
if isempty(d), d = 1; end;

if ~isscalar(d) && ( length(b) ~= length(d) )
  error('Vectors b and d must have the same length');
end

% Quick return for the easy cases.
if isscalar(d)  &&  d == 0
   x   = b;
   itn = 0;
   return
end

% Get sign of b and set to absolute values
s = sign(b);
small = abs(b) <= eps;
bs    = b.*small;

bMod = abs(b) - eps; % subtract off epsilon from abs values
b = abs(bMod).*(abs(bMod) > 0); % kill entries smaller than eps (bs)


% Perform the projection
if isscalar(d)
  [x,itn] = oneProjectorMex(b,tau/d);
else
  d   = abs(d);
  idx = find(d > eps); % Get index of all non-zero entries of d
  x   = b;             % Ensure x_i = b_i for all i not in index set idx
  [x(idx),itn] = oneProjectorMex(b(idx),d(idx),tau);
end

% Restore signs in x, add eps back to large entries, and add small entries back in
x =(1-small).*(x+eps).*s + bs ; 
