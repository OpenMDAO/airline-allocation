"""
    optimization.py

    optimization related functions from:
    NASA_LEARN_AirlineAllocation_Branch_Cut/OptimizationFiles
"""

import numpy as np

try:
    from scipy.optimize import linprog
except ImportError, e:
    print "SciPy version >= 0.15.0 is required for linprog support!!"
    pass


def range_extract(RVector, distance):
    """ find the closest match in RVector for each value in distance
        returns an index into RVector for each value in distance
    """
    indices = np.zeros(len(distance))

    for cc in xrange(len(distance)):
        dist = distance[cc]
        diff_min = np.inf

        for r_id, r_dist in np.ndenumerate(RVector):
            diff = np.abs(r_dist - dist)
            if diff < diff_min:
                indices[cc] = r_id[0]
                diff_min = diff

    return indices


def get_objective(inputs, outputs, constants, coefficients):
    """ generate the objective matrix for linprog
        returns the coefficients for the integer and continuous design variables
    """

    J = inputs.DVector.shape[0]  # number of routes
    K = len(inputs.AvailPax)     # number of aircraft types
    KJ = K*J

    fuelburn  = coefficients.Fuelburn
    docnofuel = coefficients.Doc
    price     = outputs.TicketPrice
    fuelcost  = constants.FuelCost

    obj_int = np.zeros((KJ, 1))
    obj_con = np.zeros((KJ, 1))

    for kk in xrange(K):
        for jj in xrange(J):
            col = kk*J + jj
            obj_int[col] = docnofuel[kk, jj] + fuelcost * fuelburn[kk, jj]
            obj_con[col] = -price[kk, jj]

    return obj_int.flatten(), obj_con.flatten()


def get_constraints(inputs, constants, coefficients):
    """ generate the constraint matrix/vector for linprog
    """

    J = inputs.DVector.shape[0]  # number of routes
    K = len(inputs.AvailPax)     # number of aircraft types
    KJ  = K*J
    KJ2 = KJ*2

    dem   = inputs.DVector[:, 1].reshape(-1, 1)
    BH    = coefficients.BlockTime
    MH    = constants.MH.reshape(-1, 1)
    cap   = inputs.AvailPax.flatten()
    fleet = inputs.ACNum.reshape(-1, 1)
    t     = inputs.TurnAround

    # Upper demand constraint
    A1 = np.zeros((J, KJ2))
    b1 = dem.copy()
    for jj in xrange(J):
        for kk in xrange(K):
            col = K*J + kk*J + jj
            A1[jj, col] = 1

    # Lower demand constraint
    A2 = np.zeros((J, KJ2))
    b2 = -0.2 * dem
    for jj in xrange(J):
        for kk in xrange(K):
            col = K*J + kk*J + jj
            A2[jj, col] = -1

    # Aircraft utilization constraint
    A3 = np.zeros((K, KJ2))
    b3 = np.zeros((K, 1))
    for kk in xrange(K):
        for jj in xrange(J):
            col = kk*J + jj
            A3[kk, col] = BH[kk, jj]*(1 + MH[kk, 0]) + t
        b3[kk, 0] = 12*fleet[kk]

    # Aircraft capacity constraint
    A4 = np.zeros((KJ, KJ2))
    b4 = np.zeros((KJ, 1))
    rw = 0
    for kk in xrange(K):
        for jj in xrange(J):
            col1 = kk*J + jj
            A4[rw, col1] = 0.-cap[kk]
            col2 = K*J + kk*J + jj
            A4[rw, col2] = 1
            rw = rw + 1

    A = np.concatenate((A1, A2, A3, A4))
    b = np.concatenate((b1, b2, b3, b4))

    return A, b


def gomory_cut(x, A, b, Aeq, beq):
    """ Gomory Cut (from 'GomoryCut.m')
    """
    num_des = len(x)

    slack = []
    if b.size > 0:
        slack = b - A.dot(x)
        x_up = np.concatenate((x, slack))
        Ain_com = np.concatenate((A, np.eye(len(slack))), axis=1)
    else:
        x_up = x.copy()
        Ain_com = np.array([])

    if len(beq) > 0:
        Aeq_com = np.concatenate((Aeq, np.zeros((Aeq.shape[0], len(slack)))))
    else:
        Aeq_com = np.array([])

    if Aeq_com.size > 0:
        Acom = np.concatenate((Ain_com, Aeq_com))
    else:
        Acom = Ain_com

    if beq.size > 0:
        bcom = np.concatenate((b, beq))
    else:
        bcom = b

    # Generate the Simplex optimal tableau
    aaa = np.where(np.subtract(x_up, 0.) > 1e-06)
    cols = Acom.shape[0]
    rows = len(aaa[0])
    B = np.zeros((rows, cols))
    for ii in range(cols):
        for jj in range(rows):
            B[jj, aaa[ii]] = Acom[jj, aaa[ii]]

    tab = np.concatenate((np.linalg.solve(B, Acom), np.linalg.solve(B, bcom)), axis=1)

    # Generate cut
    # Select the row from the optimal tableau corresponding
    # to the basic design variable that has the highest fractional part
    b_end = tab[:, tab.shape[1]-1].reshape(-1, 1)
    delta = np.subtract(np.round(b_end), b_end)
    aa = np.where(np.abs(delta) > 1e-06)

    rw_sel = -1
    max_rem = 0
    for ii in aa[0]:  # rows in tab with fractional part
        rems = np.remainder(np.abs(tab[ii, :]), 1)
        # FIXME: remainder of 1.0/1  = 1.0  ????  (replace with 0)
        aa = np.where(abs(rems - 1.0) <= 1e-08)
        rems[aa] = 0
        if rems.max() > max_rem:
            max_rem = rems.max()
            rw_sel = ii

    eflag = 0

    if rw_sel >= 0:
        # apply Gomory cut
        equ_cut = tab[rw_sel, :]
        lhs = np.floor(equ_cut)
        rhs = -(equ_cut - lhs)
        lhs[-1] = -lhs[-1]
        rhs[-1] = -rhs[-1]

        # cut: rhs < 0
        a_x = rhs[0:num_des]
        a_s = rhs[num_des:rhs.shape[0] - 1]
        A_new = a_x - a_s.dot(A)
        b_new = -(rhs[-1] + a_s.dot(b))

        aa = np.where(abs(A_new - 0.) <= 1e-08)
        A_new[aa] = 0
        bb = np.where(abs(b_new - 0.) <= 1e-08)
        b_new[bb] = 0

        # Update and print cut information
        if (np.sum(A_new) != 0.) and (np.sum(np.isnan(A_new)) == 0.):
            eflag = 1
            A_up = np.concatenate((A, [A_new]))
            b_up = np.concatenate((b, [b_new]))

            cut_stat = ''
            for ii in range(len(A_new)+1):
                if ii == len(A_new):
                    symbol = ' <= '
                    cut_stat = cut_stat + symbol + str(b_new[-1])
                    break
                if A_new[ii] != 0:
                    if A_new[ii] < 0:
                        symbol = ' - '
                    else:
                        if len(cut_stat) == 0:
                            symbol = ''
                        else:
                            symbol = ' + '
                    cut_stat = cut_stat + symbol + str(abs(A_new[ii])) + 'x' + str(ii)

    if eflag == 1:
        print '\nApplying cut: %s\n' % cut_stat
    else:
        A_up = A.copy()
        b_up = b.copy()
        print '\nNo cut applied!!\n'

    return A_up, b_up, eflag


def cut_plane(x, A, b, Aeq, beq, ind_con, ind_int, indeq_con, indeq_int, num_int):
    num_con = x.size - num_int
    x_trip = x[0:num_int]
    pax = x[num_int:]

    if b.size > 0:
        # A can subdivided into 4 matrices
        # A = [A_x_con, A_pax_con;
        #      A_x_int, A_pax_int]
        A_x_int   = A[ind_int, 0:num_int]
        A_pax_int = A[ind_int, num_int:A.shape[1]]
        b_x_int   = b[ind_int] - A_pax_int.dot(pax)
    else:
        A_x_int = np.array([])
        b_x_int = np.array([])

    if beq.size > 0:
        Aeq_x_int   = Aeq[indeq_int, 0:num_int]
        Aeq_pax_int = Aeq[indeq_int, num_int:Aeq.shape[1]]
        beq_x_int   = beq[indeq_int] - Aeq_pax_int.dot(pax)
    else:
        Aeq_x_int = np.array([])
        beq_x_int = np.array([])

    A_x_int_up, b_x_int_up, eflag = gomory_cut(x_trip, A_x_int, b_x_int, Aeq_x_int, beq_x_int)

    if eflag == 1:
        A_new = np.concatenate((A_x_int_up[-1, :], np.ones(num_con)))
        b_new = b_x_int_up[-1] + np.ones((1, num_con)).dot(pax)
    else:
        A_new = np.array([])
        b_new = np.array([])

    A_up = np.concatenate((A, [A_new]))
    b_up = np.concatenate((b, b_new))
    return A_up, b_up


def branch_cut(f_int, f_con, A, b, Aeq, beq, lb, ub, x0, ind_conCon, ind_intCon, indeq_conCon, indeq_intCon):
    """ branch and bound algorithm
    """
    f = np.concatenate((f_int, f_con))
    num_int = len(f_int)

    _iter = 0
    funCall = 0
    eflag = 0
    U_best = np.inf
    xopt = []
    fopt = []
    can_x = []
    can_F = []
    ter_crit = 0
    opt_cr = 0.03
    node_num = 1
    tree = 1

    class Problem(object):
        pass

    prob = Problem()
    prob.f = f
    prob.A = A
    prob.b = b
    prob.Aeq = Aeq
    prob.beq = beq
    prob.lb = lb
    prob.ub = ub
    prob.x0 = x0
    prob.b_F = 0
    prob.x_F = []
    prob.node = node_num
    prob.tree = tree

    Aset = []
    Aset.append(prob)

    while len(Aset) > 0 and ter_crit != 2:

        _iter = _iter + 1

        # pick a subproblem
        # preference given to nodes with higher objective value
        Fsub = -np.inf
        for ii in range(len(Aset)):
            if Aset[ii].b_F >= Fsub:
                Fsub_i = ii
                Fsub = Aset[ii].b_F

        # solve subproblem using linprog
        # print 'Aeq:\n', Aset[Fsub_i].Aeq
        # print 'beq:\n', Aset[Fsub_i].beq
        print 'f:\n', Aset[Fsub_i].f
        print 'A:\n', Aset[Fsub_i].A
        print 'b:\n', Aset[Fsub_i].b
        bounds = zip(Aset[Fsub_i].lb.flatten(), Aset[Fsub_i].ub.flatten())
        # print 'bounds:\n', bounds

        # Aset[Fsub_i].x_F,Aset[Fsub_i].b_F, Aset[Fsub_i].eflag = \
        # linprog(Aset[Fsub_i].f,
        #         Aset[Fsub_i].A, Aset[Fsub_i].b,
        #         Aset[Fsub_i].Aeq, Aset[Fsub_i].beq,
        #         Aset[Fsub_i].lb, Aset[Fsub_i].ub,
        #         Aset[Fsub_i].x0)
        results = linprog(Aset[Fsub_i].f,
                          A_eq=None,
                          b_eq=None,
                          A_ub=Aset[Fsub_i].A,
                          b_ub=Aset[Fsub_i].b,
                          bounds=bounds,
                          method='simplex',
                          callback=None,
                          options={ 'maxiter': 100, 'disp': True })
        print 'results:\n', results
        Aset[Fsub_i].x_F = results.x
        Aset[Fsub_i].b_F = results.fun
        Aset[Fsub_i].eflag = 1 if results.success else 0

        funCall = funCall + 1

        # rounding integers
        aa = np.where(np.abs(np.round(Aset[Fsub_i].x_F) - Aset[Fsub_i].x_F) <= 1e-06)
        Aset[Fsub_i].x_F[aa] = np.round(Aset[Fsub_i].x_F[aa])

        if _iter == 1:
            x_best_relax = Aset[Fsub_i].x_F
            f_best_relax = Aset[Fsub_i].b_F

        if ((Aset[Fsub_i].eflag >= 1) and (Aset[Fsub_i].b_F < U_best)):
            if np.linalg.norm(Aset[Fsub_i].x_F[range(num_int)] - np.round(Aset[Fsub_i].x_F[range(num_int)])) <= 1e-06:
                can_x = [can_x, Aset[Fsub_i].x_F]
                can_F = [can_F, Aset[Fsub_i].b_F]
                x_best = Aset[Fsub_i].x_F
                U_best = Aset[Fsub_i].b_F
                print '======================='
                print 'New solution found!'
                print '======================='
                del Aset[Fsub_i]  # Fathom by integrality
                ter_crit = 1
                if (abs(U_best - f_best_relax) / abs(f_best_relax)) <= opt_cr:
                    ter_crit = 2
            else:  # cut and branch
                # apply cut to subproblem
                if Aset[Fsub_i].node != 1:
                    Aset[Fsub_i].A, Aset[Fsub_i].b = cut_plane(Aset[Fsub_i].x_F,
                        Aset[Fsub_i].A, Aset[Fsub_i].b,
                        Aset[Fsub_i].Aeq, Aset[Fsub_i].beq,
                        ind_conCon, ind_intCon,
                        indeq_conCon, indeq_intCon,
                        num_int)
                print 'after cut, A:\n', Aset[Fsub_i].A

                # branching
                x_ind_maxfrac = np.argmax(np.remainder(np.abs(Aset[Fsub_i].x_F[range(num_int)]), 1))
                x_split = Aset[Fsub_i].x_F[x_ind_maxfrac]
                print '\nBranching at tree: %d at x%d = %f\n' % (Aset[Fsub_i].tree, x_ind_maxfrac, x_split)
                F_sub = []
                for jj in range(2):
                    print jj
                    F_sub.append(Aset[Fsub_i])
                    A_rw_add = np.zeros(len(Aset[Fsub_i].x_F))
                    print 'A_rw_add:', A_rw_add
                    if jj == 0:
                        A_con = 1
                        b_con = np.floor(x_split)
                    else:
                        if jj == 1:
                            A_con = -1
                            b_con = -np.ceil(x_split)

                    A_rw_add[x_ind_maxfrac] = A_con
                    print 'A_rw_add:', A_rw_add
                    print 'F_sub[jj].A:', F_sub[jj].A
                    A_up = np.concatenate((F_sub[jj].A, [A_rw_add]))
                    print 'A_up:', A_up

                    print 'F_sub[jj].b:', F_sub[jj].b
                    print 'b_con:', b_con
                    b_up = np.append(F_sub[jj].b, b_con)
                    print 'b_up:', b_up
                    F_sub[jj].A = A_up
                    F_sub[jj].b = b_up
                    F_sub[jj].tree = 10 * F_sub[jj].tree + jj
                    node_num = node_num + 1
                    F_sub[jj].node = node_num
                del Aset[Fsub_i]
                Aset.extend(F_sub)
        else:
            del Aset[Fsub_i]  # Fathomed by infeasibility or bounds

    if ter_crit > 0:
        eflag = 1
        xopt = x_best.copy()
        fopt = U_best.copy()
        if ter_crit == 1:
            print '\nSolution found but is not within %0.1f%% of the best relaxed solution!\n' % opt_cr*100
        elif ter_crit == 2:
            print '\nSolution found and is within %0.1f%% of the best relaxed solution!\n' % opt_cr*100
    else:
        print '\nNo solution found!!\n'

    return xopt, fopt, can_x, can_F, x_best_relax, f_best_relax, funCall, eflag


if __name__ == "__main__":

    def load_data(file_name):
        # utility function to load MATLAB data
        from os.path import dirname, pardir, join
        from scipy.io import loadmat

        data_path = join(dirname(__file__),pardir,pardir,'MATLAB','Data')

        return loadmat(join(data_path,file_name),
                       squeeze_me=True, struct_as_record=False)

    # smaller network with 3 routes
    inputs       = load_data('inputs_after_3routes.mat')['Inputs']
    outputs      = load_data('outputs_after_3routes.mat')['Outputs']
    constants    = load_data('constants_after_3routes.mat')['Constants']
    coefficients = load_data('coefficients_after_3routes.mat')['Coefficients']

    # linear objective coefficients
    objective   = get_objective(inputs, outputs, constants, coefficients)
    f_int = objective[0]    # integer type design variables
    f_con = objective[1]    # continuous type design variables

    # coefficient matrix for linear inequality constraints, Ax <= b
    constraints = get_constraints(inputs, constants, coefficients)
    A = constraints[0]
    b = constraints[1]

    J = inputs.DVector.shape[0]  # number of routes
    K = len(inputs.AvailPax)     # number of aircraft types

    # lower and upper bounds
    lb = np.zeros((2*K*J, 1))
    ub = np.concatenate((
        np.ones((K*J, 1)) * inputs.MaxTrip.reshape(-1, 1),
        np.ones((K*J, 1)) * np.inf
    ))

    # initial x
    x0 = []

    # indices into A matrix for continuous & integer/continuous variables
    ind_conCon = range(2*J)
    ind_intCon = range(2*J, len(constraints[0])+1)

    # call the branch and cut algorithm to solve the MILP problem
    branch_cut(f_int, f_con, A, b, [], [], lb, ub, x0,
               ind_conCon, ind_intCon, [], [])
