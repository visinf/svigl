classdef hstruct < handle
	properties
		data
	end
	
	methods
		function obj = hstruct(data)
			obj.data = data;
		end
	end
end