%% Branch and bound algorithm

% % Test problem
% clc;close all;clear all
% f= [-8;-5];
% A=[1,1;9,5];
% b=[6;45];
% Aeq=[];
% beq=[];
% lb=[0;0];
% ub=[inf;inf];
% x0=[];
% num_int = 2;
% num_con = 0;

%% Main function Branch and Bound
function [xopt,fopt,can_x,can_F,x_best_relax,f_best_relax,funCall,eflag] = branch_cut...
    (f_int,f_con,A,b,Aeq,beq,lb,ub,x0,ind_conCon,ind_intCon,indeq_conCon,indeq_intCon)
%% This is the branch and cut algorithm

% INPUTS: 

% f_int, f_con - linear objective coefficents for the integer type and
% continuous type design variables

% A, b - Coefficient matrix for linear inequality constraints Ax <= b

% Aeq, beq - Coefficient matrix for linear equality constraints Aeqx = beq

% lb, ub - Lower and upper bounds on the design variables

% x0 -  Initial x

% ind_conCon - indices in the A matrix correspoding to the
% constraints containing only continuous type design variables

% ind_intCon - indices in the A matrix correspoding to the
% constraints containing integer and continuous (if any) type design variables

% OUTPUTS:
% xopt - optimal x with integer soltuion.
% fopt - optimal objective funtion value
% can_x - list of candidate solutions x that are feasible (i.e satisfies
% integer constraint)

% can_F - Corresponding list of objective function values
% x_best_relax - x value of the relaxed problem (i.e no integer constraint)
% f_best_relax - Objective fucntion value of the relaxed problem
% funCall -  total number of times the optimizer is executed
% eflag -  status of the run. 1- Solution exists. 0 - no solution found

f = [f_int;f_con];
num_int = length(f_int);num_con = length(f_con);
num_des = num_int + num_con;

iter = 0;funCall = 0;eflag=0;U_best = Inf;
xopt = [];fopt = [];can_x = [];can_F = [];
ter_crit = 0;opt_cr = 0.03;
node_num = 1;tree = 1;

Aset.f = f;Aset.A = A;Aset.b = b;Aset.Aeq = Aeq;Aset.beq = beq;
Aset.lb = lb;Aset.ub = ub;Aset.x0 = x0;Aset.b_F = 0;Aset.x_F=[];
Aset.node = node_num;Aset.tree = tree;

while isempty(Aset)~=1 && ter_crit~=2
    iter = iter + 1;
    
    % Pick A subproblem 
    % Preference given to nodes with higher objective value
    Fsub = -Inf;
    for ii = 1:length(Aset)
        if Aset(ii).b_F >= Fsub
            Fsub_i = ii;
            Fsub = Aset(ii).b_F;
        end
    end

    %Solve subproblem

    % Using Linprog  
    [Aset(Fsub_i).x_F,Aset(Fsub_i).b_F,...
        Aset(Fsub_i).eflag] = linprog(Aset(Fsub_i).f,...
        Aset(Fsub_i).A,Aset(Fsub_i).b,Aset(Fsub_i).Aeq,...
        Aset(Fsub_i).beq,Aset(Fsub_i).lb,Aset(Fsub_i).ub,...
        Aset(Fsub_i).x0);

    % Using other solver
%         options = optimset('Algorithm','active-set');
%         [x_sol,b_F(Fsub_i),exitflag(Fsub_i)] = fmincon(@(x_sol) f'*x_sol,Aset(Fsub_i).x0,Aset(Fsub_i).A,Aset(Fsub_i).b,Aset(Fsub_i).Aeq,...
%              Aset(Fsub_i).beq,Aset(Fsub_i).lb,Aset(Fsub_i).ub,[],options);  
      
    funCall = funCall + 1;
    %Rounding integers
    aa = find(abs(round(Aset(Fsub_i).x_F)-Aset(Fsub_i).x_F)<=1e-6);
    Aset(Fsub_i).x_F(aa) = round(Aset(Fsub_i).x_F(aa));        
    
    if iter == 1
        %Saving the best relaxed solution
        x_best_relax = Aset(Fsub_i).x_F;
        f_best_relax = Aset(Fsub_i).b_F;
    end
    
    if ((Aset(Fsub_i).eflag >= 1) && (Aset(Fsub_i).b_F < U_best))
        if norm(Aset(Fsub_i).x_F(1:num_int)-round(Aset(Fsub_i).x_F(1:num_int))) <= 1e-6    
            can_x = [can_x,Aset(Fsub_i).x_F]; 
            can_F = [can_F,Aset(Fsub_i).b_F];    
            x_best = Aset(Fsub_i).x_F;
            U_best = Aset(Fsub_i).b_F;
            fprintf('\n%s', '=======================')
            fprintf('\n%s', 'New solution found!')
            fprintf('\n%s\n', '=======================') 
            Aset(Fsub_i) = []; %Fathom by integrality
            ter_crit = 1;
            if (abs(U_best-f_best_relax)/abs(f_best_relax))<=opt_cr
                ter_crit = 2;
            end
        else % Cut and Branch
            
            % Apply cut to the subproblem
            if Aset(Fsub_i).node~=1
                [Aset(Fsub_i).A,Aset(Fsub_i).b] = ...
                call_Cutplane(Aset(Fsub_i).x_F,Aset(Fsub_i).A,Aset(Fsub_i).b,...
                Aset(Fsub_i).Aeq,Aset(Fsub_i).beq,ind_conCon,ind_intCon,...
                indeq_conCon,indeq_intCon,num_int);
            end    

            %Branching
            [~,x_ind_maxfrac] = max(rem(abs(Aset(Fsub_i).x_F(1:num_int)),1));
            x_split = Aset(Fsub_i).x_F(x_ind_maxfrac);
            fprintf('\n%s%d%s%d%s%f\n', 'Branching at tree: ',Aset(Fsub_i).tree,' at x',x_ind_maxfrac, ' = ',x_split)
            for jj = 1:2
                F_sub(jj) = Aset(Fsub_i);
                A_rw_add = zeros(1,length(Aset(Fsub_i).x_F));
                if jj ==1
                    A_con = 1;
                    b_con = floor(x_split);
                elseif jj==2
                    A_con = -1;
                    b_con = -ceil(x_split);
                end
                A_rw_add(x_ind_maxfrac) = A_con;
                A_up = [F_sub(jj).A;A_rw_add];
                b_up = [F_sub(jj).b;b_con];
                F_sub(jj).A = A_up;
                F_sub(jj).b = b_up;
                F_sub(jj).tree = 10*F_sub(jj).tree + jj;
                node_num = node_num + 1;
                F_sub(jj).node = node_num;
            end
            Aset(Fsub_i) = [];
            Aset = [Aset,F_sub];
        end
    else
        Aset(Fsub_i) = []; %Fathomed by infeasibility or bounds
    end
end

if ter_crit>0
    eflag = 1;
    xopt = x_best;
    fopt = U_best;
    if ter_crit == 1
        fprintf('\n%s%0.1f%s\n', 'Solution found but is not within ',opt_cr*100,'% of the best relaxed solution!')
    elseif ter_crit == 2
        fprintf('\n%s%0.1f%s\n', 'Solution found and is within ',opt_cr*100,'% of the best relaxed solution!')
    end
else
    fprintf('\n%s\n', 'No solution found!!')
end

       

    
    

     