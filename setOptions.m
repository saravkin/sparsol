function opt = setOptions(opt,varargin)
   for i=1:2:length(varargin)
      if ~isfield(opt,varargin{i})
         opt = setfield(opt,varargin{i},varargin{i+1});
      end      
   end   
end
