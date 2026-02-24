function struct2csv(s,fn)
% STRUCT2CSV(s,fn)
%
% Output a structure to a comma delimited file with column headers
%
%       s : any structure composed of one or more matrices and cell arrays
%      fn : file name
%
%      Given s:
%
%          s.Alpha = { 'First', 'Second';
%                      'Third', 'Fourth'};
%
%          s.Beta  = [[      1,       2;
%                            3,       4]];
%          
%          s.Gamma = {       1,       2;
%                            3,       4};
%
%          s.Epsln = [     abc;
%                          def;
%                          ghi];
% 
%      STRUCT2CSV(s,'any.csv') will produce a file 'any.csv' containing:
%
%         "Alpha",        , "Beta",   ,"Gamma",   , "Epsln",
%         "First","Second",      1,  2,      1,  2,   "abc",
%         "Third","Fourth",      3,  4,      3,  4,   "def",
%                ,        ,       ,   ,       ,   ,   "ghi",
%    
%      v.0.9 - Rewrote most of the code, now accommodates a wider variety
%              of structure children
%
% Written by James Slegers, james.slegers_at_gmail.com
% Covered by the BSD License
%
s = rmfield(s, 't1');
s = rmfield(s, 'L_s');
s = rmfield(s, 'bigs_s');
s = rmfield(s, 'isline_s');
s = rmfield(s, 'meds_s');
%s = rmfield(s, 'ecc_s');
%s = rmfield(s, 'wig_s');
s = rmfield(s, 'l_s');
s = rmfield(s, 'wid_s');
s = rmfield(s, 'g');
s = rmfield(s, 'g_std');
s = rmfield(s, 'backG');
s = rmfield(s, 'backStd');
s = rmfield(s, 'backStd_s');
s = rmfield(s, 'backG_s');
s = rmfield(s, 'm_im');
s = rmfield(s, 'p_im');
s = rmfield(s, 'patch');
s = rmfield(s, 'sf');
s = rmfield(s, 'pct1');
s = rmfield(s, 'pct2');
s = rmfield(s, 'patnum');
s = rmfield(s, 'fn_mask');
s = rmfield(s, 'fn_im');
s = rmfield(s, 'sl');
s = rmfield(s, 'th1');
s = rmfield(s, 'thf');
s = rmfield(s, 'thf_s');
s = rmfield(s, 'th2');
s = rmfield(s, 'th1_s');
s = rmfield(s, 'th2_s');
s = rmfield(s, 'stren');
s = rmfield(s, 's_lev');
s = rmfield(s, 'ptype');


for c = 1:length(s) % added this, testing it not sure
    s(c).ecc_s = sum(s(c).ecc_s);
    s(c).wig_s = sum(s(c).wig_s);
end
FID = fopen(fn,'w');
headers = fieldnames(s);
m = length(headers);
sz = zeros(m,2);

t = length(s);

for rr = 1:t
    

    l = '';
    for ii = 1:m
        sz(ii,:) = size(s(rr).(headers{ii}));   
        if ischar(s(rr).(headers{ii}))
            sz(ii,2) = 1;
        end
        l = [l,'"',headers{ii},'",',repmat(',',1,sz(ii,2)-1)];
    end

    l = [l,'\n'];
    %repeating header change made here
    if rr == 1
        fprintf(FID,l);
    end
    %repeating header change made here
    n = max(sz(:,1));

    for ii = 1:n
        l = '';
        for jj = 1:m
            c = s(rr).(headers{jj});
            str = '';

            if sz(jj,1)<ii
                str = repmat(',',1,sz(jj,2));
            else
                if isnumeric(c)
                    for kk = 1:sz(jj,2)
                        str = [str,num2str(c(ii,kk)),','];
                    end
                elseif islogical(c)
                    for kk = 1:sz(jj,2)
                        str = [str,num2str(double(c(ii,kk))),','];
                    end
%                elseif ismatrix(c) %testing this
%                        str = [sum(c), ','];
                elseif ischar(c)
                    str = ['"',c(ii,:),'",'];
                elseif iscell(c)
                    if isnumeric(c{1,1})
                        for kk = 1:sz(jj,2)
                            str = [str,num2str(c{ii,kk}),','];
                        end
                    elseif islogical(c{1,1})
                        for kk = 1:sz(jj,2)
                            str = [str,num2str(double(c{ii,kk})),','];
                        end
                    elseif ischar(c{1,1})
                        for kk = 1:sz(jj,2)
                            str = [str,'"',c{ii,kk},'",'];
                        end
                    end
                end
            end
            l = [l,str];
        end
        l = [l,'\n'];
        fprintf(FID,l);
    end
    %fprintf(FID,'\n');
end

fclose(FID);
