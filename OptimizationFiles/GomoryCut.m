% % Test problem
% clc;clear all;close all
% x=[55/14;10/7];
% A = [2/5,1;2/5,-2/5];
% b = [3;1];
% Aeq = [];
% beq = [];

function [A_up,b_up,eflag] = GomoryCut(x,A,b,Aeq,beq)
num_des = length(x);
slack = [];
if isempty(b)~=1
    slack = b - A*x;
    x_up = [x;slack];
    Ain_com = [A,eye(length(slack))];
else
    x_up = x;
    Ain_com = [];
end
if isempty(beq)~=1
    Aeq_com = [Aeq,zeros(size(Aeq,1),length(slack))];
else
    Aeq_com = [];
end
Acom = [Ain_com;Aeq_com];
bcom = [b;beq];

%Generate the Simplex optimal tableau
aaa = find((x_up-0)>1e-6);
B =[];
for ii = 1:length(aaa)
    B = [B,Acom(:,aaa(ii))];
end
tab = [B\Acom,B\bcom];

% Generate cut 
% Select the row from the optimal tableau corresponding 
% to the basic design variable that has the highest fractional part
b_end = tab(:,end);
aa = find(abs(round(b_end) - b_end)>1e-6);
[~,rw_sel] = max(rem(abs(tab(aa,end)),1));
% rw_sel = input('row select: ');
eflag = 0;
if isempty(rw_sel)~=1
    % Apply Gomory Cut
    equ_cut = tab(rw_sel,:);
    lhs = floor(equ_cut); 
    rhs = -(equ_cut-lhs);
    lhs(end) = -lhs(end);rhs(end) = -rhs(end);
    % Cut: rhs <= 0
    a_x = rhs(1:num_des);
    a_s = rhs(num_des+1:end-1);
    A_new = a_x - a_s*A;
    b_new = -(rhs(end) + a_s*b);

    aa = find(abs(A_new-0)<=1e-8);A_new(aa) = 0;
    bb = find(abs(b_new-0)<=1e-8);b_new(bb) = 0;
     %% Update and print cut information
    if (sum(A_new)~=0) && (sum(isnan(A_new))==0)
        eflag = 1;
        A_up = [A;A_new];
        b_up = [b;b_new];

        cut_stat = [];
        for ii = 1:length(A_new)+1
                if ii == length(A_new)+1
                    symbol = ' <= ';
                    cut_stat = [cut_stat,symbol,num2str(b_new(end))];
                    break
                end
                if A_new(ii)~=0
                    if A_new(ii) < 0
                        symbol = ' - ';
                    else
                        if isempty(cut_stat) == 1
                            symbol = '';
                        else
                            symbol = ' + ';
                        end
                    end
                cut_stat = [cut_stat,symbol,num2str(abs(A_new(ii))),'x',num2str(ii)];
                end
        end
    end
end
if eflag == 1
    fprintf('\n%s%s\n','Applying cut: ', cut_stat)
else
    A_up = A;
    b_up = b;
    fprintf('\n%s\n','No cut applied!!')
end
