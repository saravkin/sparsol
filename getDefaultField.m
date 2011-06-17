function opt = getDefaultField(data,field,default)
   if isfield(data,field)
      opt = getfield(data,field);
   else
      opt = default;
   end
end
