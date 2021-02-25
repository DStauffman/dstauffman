classdef(Enumeration) Gender < int32

% Gender is a enumerator class definition for the possible genders.
%
% Input:
%     None
%
% Output:
%     Gender ............ : (enum) possible values
%         null        = 0 : Not a valid setting, used to preallocate
%         female      = 1 : Female
%         male        = 2 : Male
%         uncirc_male = 3 : Uncircumcised male
%         circ_male   = 4 : Circumcised male
%
% Methods:
%     is_female : returns a boolean for whether the person is female or not.
%     is_male   : returns a boolean for whether the person is male or not.
%
% Prototype:
%     Gender.female           % returns female as an enumerated Gender type
%     double(Gender.female)   % returns 1, which is the enumerated value of Gender.female
%     char(Gender.female)     % returns 'female' as a character string
%     class(Gender.female)    % returns 'Gender' as a character string
%     Gender.female.is_female % return 1 (or true), as this is a female
%
% Change Log:
%     1.  Written by David C. Stauffer in June 2013.
%     2.  Updated by David C. Stauffer in April 2016 to include methods for determing male/female.

    enumeration
        null(0)
        female(1)
        male(2)
        uncirc_male(3)
        circ_male(4)
        non_binary(5)
    end

    methods
        function out = is_female(obj)
            out = obj == Gender.female;
        end
        function out = is_male(obj)
            out = obj == Gender.male | obj == Gender.uncirc_male | obj == Gender.circ_male;
        end
    end
end