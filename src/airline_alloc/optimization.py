"""
    optimization.py

    optimization related functions from:
    NASA_LEARN_AirlineAllocation_Branch_Cut/OptimizationFiles
"""

import numpy as np


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
    f = [[f_int], [f_con]]
    num_int = len(f_int)
    num_con = len(f_con)
    num_des = num_int + num_con

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

    Aset = object()
    Aset.f = f
    Aset.A = A
    Aset.b = b
    Aset.Aeq = Aeq
    Aset.beq = beq
    Aset.lb = lb
    Aset.ub = ub
    Aset.x0 = x0
    Aset.b_F = 0
    Aset.x_F = []
    Aset.node = node_num
    Aset.tree = tree

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
        Aset[Fsub_i].x_F, Aset[Fsub_i].b_F, Aset[Fsub_i].eflag = \
            np.linprog(Aset[Fsub_i].f,
                       Aset[Fsub_i].A, Aset[Fsub_i].b,
                       Aset[Fsub_i].Aeq, Aset[Fsub_i].beq,
                       Aset[Fsub_i].lb, Aset[Fsub_i].ub,
                       Aset[Fsub_i].x0)

        funCall = funCall + 1

        # rounding integers
        aa = np.where(abs(round(Aset[Fsub_i].x_F) - Aset[Fsub_i].x_F) <= 1e-06)
        Aset[Fsub_i].x_F[aa] = round(Aset[Fsub_i].x_F(aa))

        if _iter == 1:
            x_best_relax = Aset[Fsub_i].x_F
            f_best_relax = Aset[Fsub_i].b_F

        if ((Aset[Fsub_i].eflag >= 1) and (Aset[Fsub_i].b_F < U_best)):
            if np.norm(Aset[Fsub_i].x_F(range(num_int)) - round(Aset[Fsub_i].x_F(range(num_int)))) <= 1e-06:
                can_x = [can_x, Aset[Fsub_i].x_F]
                can_F = [can_F, Aset[Fsub_i].b_F]
                x_best = Aset[Fsub_i].x_F
                U_best = Aset[Fsub_i].b_F
                print '======================='
                print 'New solution found!'
                print '======================='
                Aset[Fsub_i] = []  # Fathom by integrality
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

                # branching
                __, x_ind_maxfrac = max(np.remainder(abs(Aset[Fsub_i].x_F(range(num_int))), 1))
                x_split = Aset[Fsub_i].x_F(x_ind_maxfrac)
                print '\n%s%d%s%d%s%f\n' % ('Branching at tree: ', Aset[Fsub_i].tree, ' at x', x_ind_maxfrac, ' = ', x_split)
                F_sub = []
                for jj in range(2):
                    F_sub[jj] = Aset[Fsub_i]
                    A_rw_add = np.zeros(1, len(Aset[Fsub_i].x_F))
                    if jj == 0:
                        A_con = 1
                        b_con = np.floor(x_split)
                    else:
                        if jj == 2:
                            A_con = -1
                            b_con = -np.ceil(x_split)

                    A_rw_add[x_ind_maxfrac] = A_con
                    A_up = [[F_sub[jj].A], [A_rw_add]]
                    b_up = [[F_sub[jj].b], [b_con]]
                    F_sub[jj].A = A_up
                    F_sub[jj].b = b_up
                    F_sub[jj].tree = 10 * F_sub[jj].tree + jj
                    node_num = node_num + 1
                    F_sub[jj].node = node_num
                Aset[Fsub_i] = []
                Aset = [Aset, F_sub]
        else:
            Aset[Fsub_i] = []  # Fathomed by infeasibility or bounds

    if ter_crit > 0:
        eflag = 1
        xopt = x_best.copy()
        fopt = U_best.copy()
        if ter_crit == 1:
            print '\n%s%0.1f%s\n' % ('Solution found but is not within ', opt_cr*100, '%% of the best relaxed solution!')
        elif ter_crit == 2:
            print '\n%s%0.1f%s\n' % ('Solution found and is within ', opt_cr*100, '%% of the best relaxed solution!')
    else:
        print '\n%s\n' % 'No solution found!!'

    return xopt, fopt, can_x, can_F, x_best_relax, f_best_relax, funCall, eflag
