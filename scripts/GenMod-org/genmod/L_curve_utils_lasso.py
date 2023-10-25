import numpy as np
import sklearn.linear_model as lm
import plotly.graph_objects as go


def find_curvature(x_temp, y_temp, extrema):
    """Given three points with x cordinates and y coordinates in x_temp and
    y_temp, respectively, calculate the normalized curvature. The normalization
    is done using values in extrema=[xmin,ymin,xmax,ymax]"""

    # Change to a log scale and normalize on [0,1]
    x = (np.log10(x_temp) - np.log10(extrema[0])) / (
        np.log10(extrema[2]) - np.log10(extrema[0]))
    y = (np.log10(y_temp) - np.log10(extrema[1])) / (
        np.log10(extrema[3]) - np.log10(extrema[1]))

    # If points are on straight line return error
    if (x[0] == x[1] and x[1] == x[2]):
        print('x equal')
        return('err')
    if (y[0] == y[1] and y[1] == y[2]):
        print('y equal')
        return('err')

    # Calculate curvature
    d1 = np.sqrt((y[1] - y[0])**2 + (x[1] - x[0])**2)
    d2 = np.sqrt((y[2] - y[0])**2 + (x[2] - x[0])**2)
    d3 = np.sqrt((y[1] - y[2])**2 + (x[1] - x[2])**2)

    area = 1 / 2 * (
        x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1])
    )
    curvature = 4 * area / (d1 * d2 * d3)

    return(curvature)


def find_alpha_vec(A, y, alpha_min, alpha_max, method, D=None, tol=1e-12):
    # Initiate vectors to store the residuals
    sol_res = np.zeros(4)  # ||Ax-y||_2
    reg_res = np.zeros(4)  # Regularization residual (||x||_2 or ||Dx||_1)

    # Calculate the least squares solution and residual
    ls_sol = np.linalg.pinv(np.transpose(A) @ A) @ np.transpose(A) @ y
    ls_res = np.linalg.norm(A @ ls_sol - y)
    # print('Least Squares Residual: ' + str(ls_res))

    # Golden ratio: Used to search hyperparameter space
    pGS = (1 + np.sqrt(5)) / 2

    # Update alpha_min and alpha_max so minimum regularization residual
    # does not equal zero and the minimum solution residual does not equal the
    # least squares solution. This helps the algorithm converge properly.
    while np.min(reg_res) == 0 or np.min(sol_res - ls_res) < tol:

        # Specify the two interior alpha values using the golden ratio.
        alpha2 = alpha_max**(1 / (1 + pGS)) * alpha_min**(pGS / (1 + pGS))
        alpha3 = alpha_max**(pGS / (1 + pGS)) * alpha_min**(1 / (1 + pGS))
        # Construct vector containing the 4 alpha values
        alpha_vec = np.array([alpha_min, alpha2, alpha3, alpha_max])
        # Perform lasso at the 4 alpha values (based on method)
        for i in range(4):
            if method == '1':
                las = lm.Lasso(alpha=alpha_vec[i], fit_intercept=False,
                               tol=tol, max_iter=100000)
                las.fit(A, y)
                sol_res[i] = np.linalg.norm(A @ las.coef_ - y)
                reg_res[i] = np.sum(np.abs(las.coef_))
            elif method == '2':
                coef = np.linalg.inv(
                    np.transpose(A) @ A + alpha_vec[i] * np.transpose(D) @ D
                ) @ np.transpose(A) @ y
                sol_res[i] = np.linalg.norm(A @ coef - y)
                reg_res[i] = np.linalg.norm(D @ coef)
        if np.min(reg_res) == 0:
            alpha_max = alpha_max / 2
        if np.min(sol_res - ls_res) < tol:
            alpha_min = alpha_min * 2

    # Return the final alpha_vec and residuals
    return alpha_vec, sol_res, reg_res


def lassoLCurve(A, y, alpha_min, alpha_max, D=None, method='2',
                max_iter=100, tol=1e-12):
    """Run the L curve algorithm for selection the hyperparameter alpha used in
    optimization. Two optimzation methods are possible:
    -method='1': ||Ax-y||_2 + alpha||x||_1
    -method='2': ||Ax-y||_2 + alpha||Dx||_2 """

    # Golden ratio: Used to search hyperparameter space
    pGS = (1 + np.sqrt(5)) / 2

    for iter in range(max_iter):
        # On first iteration find initial alpha vec and corresponding extrema
        if iter == 0:
            alpha_vec, sol_res, reg_res = find_alpha_vec(
                A, y, alpha_min, alpha_max, method, D=D, tol=tol
            )
            extrema = np.array([
                np.min(sol_res), np.min(reg_res),
                np.max(sol_res), np.max(reg_res)
            ])
        # Set alpha_min and alpha_max values
        alpha_min = alpha_vec[0]
        alpha_max = alpha_vec[3]
        # Find curvature using the calculated solution and regularization residuals
        curvature1 = find_curvature(sol_res[:3], reg_res[:3], extrema)
        curvature2 = find_curvature(sol_res[1:], reg_res[1:], extrema)

        # If error results calculate new alpha_vec
        while curvature1 == 'err' or curvature2 == 'err':
            print('Curvature 1: ' + str(curvature1))
            print('Curvature 2: ' + str(curvature2))
            if curvature1 == 'err':
                alpha_min = alpha_min * 2
            if curvature2 == 'err':
                alpha_max = alpha_max / 2
            if alpha_max < alpha_min:
                print('error: alpha max less than alpha min')
                print(alpha_vec)
                return alpha_vec[0]
            alpha_vec, sol_res, reg_res = find_alpha_vec(
                A, y, alpha_min, alpha_max, method, D=D, tol=tol
            )
            alpha_min = alpha_vec[0]
            alpha_max = alpha_vec[3]
            curvature1 = find_curvature(sol_res[:3], reg_res[:3], extrema)
            curvature2 = find_curvature(sol_res[1:], reg_res[1:], extrema)

        # If curvature is negative, modify alpha_vec
        while curvature2 < 0 and curvature1 < 0:
            # if curvature1 < 0:
            #     alpha_min = alpha_min*2
            if curvature2 < 0:
                alpha_max = alpha_max / 2
            if alpha_max < alpha_min:
                print('error: alpha max less than alpha min')
                print(alpha_vec)
                return alpha_vec[0]
            alpha_vec, sol_res, reg_res = find_alpha_vec(
                A, y, alpha_min, alpha_max, method, D=D, tol=tol
            )
            alpha_min = alpha_vec[0]
            alpha_max = alpha_vec[3]
            curvature1 = find_curvature(sol_res[:3], reg_res[:3], extrema)
            curvature2 = find_curvature(sol_res[1:], reg_res[1:], extrema)

        # Best on region of maximum curvature, modify alpha_vec
        if curvature1 > curvature2:
            alpha_vec[2:4] = alpha_vec[1:3]
            sol_res[2:4] = sol_res[1:3]
            reg_res[2:4] = reg_res[1:3]
            alpha_vec[1] = alpha_vec[3]**(1 / (1 + pGS)) * alpha_vec[0]**(pGS / (1 + pGS))
            lasso_idx = 1
            c_best = curvature1
        else:
            alpha_vec[0:2] = alpha_vec[1:3]
            sol_res[0:2] = sol_res[1:3]
            reg_res[0:2] = reg_res[1:3]
            alpha_vec[2] = alpha_vec[0]**(1 / (1 + pGS)) * alpha_vec[3]**(pGS / (1 + pGS))
            lasso_idx = 2
            c_best = curvature2

        # For new value of alpha added, perform lasso based on the necessary method
        if method == '1':
            las = lm.Lasso(alpha=alpha_vec[lasso_idx], fit_intercept=False,
                           tol=tol, max_iter=100000)
            las.fit(A, y)
            sol_res[lasso_idx] = np.linalg.norm(A @ las.coef_ - y)
            reg_res[lasso_idx] = np.sum(np.abs(las.coef_))
        elif method == '2':
            coef = np.linalg.inv(np.transpose(A) @ A + alpha_vec[lasso_idx] * np.transpose(D) @ D) @ np.transpose(A) @ y
            sol_res[lasso_idx] = np.linalg.norm(A @ coef - y)
            reg_res[lasso_idx] = np.linalg.norm(D @ coef)

        # Once alpha_vec has converged break and return value
        if (alpha_vec[3] - alpha_vec[0]) / alpha_vec[0] < 1e-4:
            # print('Breaking at iteration: ' + str(iter))
            # print(alpha_vec)
            break

    return alpha_vec[0]


def get_L_curve_data(A, y, alpha_min, alpha_max, a_num=50, method='1', D=None,
                     c_actual=None, tol=1e-12):
    '''Gives the value of the solution and regularization residuals at a_num
    alpha values rangin from alpha_min to alpha_max. Two optimzation methods are possible:
    -method='1': ||Ax-y||_2 + alpha||x||_1
    -method='2': ||Ax-y||_2 + alpha||Dx||_2 """'''
    #Generate list of alphas at which to perform lasso
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), a_num)

    norm_min = 10000
    reg_res = np.zeros(a_num)

    sol_res = np.zeros(a_num)
    for k in range(a_num):
        # print(k)
        a = alphas[k]
        if method == '1':
            las = lm.Lasso(alpha=a, fit_intercept=False, max_iter=100000,
                           tol=tol)
            las.fit(A, y)
            x = las.coef_
            reg_res[k] = np.sum(np.abs(x))
            sol_res[k] = np.linalg.norm(A @ x - y)
        elif method == '2':
            x = np.linalg.inv(
                np.transpose(A) @ A + a * np.transpose(D) @ D
            ) @ np.transpose(A) @ y
            reg_res[k] = np.linalg.norm(D @ x)
            sol_res[k] = np.linalg.norm(A @ x - y)

        # Determine if current alpha gives the optimal solution
        if c_actual is not None:
            norm_cur = np.linalg.norm(x - c_actual)
            if norm_cur < norm_min:
                alpha_best = a
                norm_min = norm_cur
                x_best = x

    if c_actual is None:
        return(reg_res, sol_res, alphas)
    else:
        return(reg_res, sol_res, alphas, alpha_best, x_best)


def findAlphaMaxCurve(A, udot, tol, title, c_actual=None, plot_results=False,
                      a_num=200, d_lim=0.01, D=None, method='1'):
    """Find the hyperparameter using L curve criteria.

    Two optimzation methods are possible:
    -method='1': ||Ax-y||_2 + alpha||x||_1
    -method='2': ||Ax-y||_2 + alpha||Dx||_2
    """
    if plot_results:
        alpha_vec, sol_res, reg_res = find_alpha_vec(A, udot, 1e-8, 1, method,
                                                     D=D, tol=tol)

        if c_actual is None:
            x, y, alphas = get_L_curve_data(
                A, udot, alpha_vec[0], alpha_vec[-1], a_num=a_num,
                method=method, c_actual=c_actual, D=D, tol=tol
            )
        else:
            x, y, alphas, alpha_best, signal_best = get_L_curve_data(
                A, udot, alpha_vec[0], alpha_vec[-1], a_num=a_num,
                method=method, c_actual=c_actual, D=D, tol=tol
            )

        # subsample y/x so the points aren't too dense
        y_new = np.array([y[0]])
        x_new = np.array([x[0]])
        alphas_new = np.array([alphas[0]])
        xlog = np.log10(x)
        ylog = np.log10(y)
        xlog_new = np.array([xlog[0]])
        ylog_new = np.array([ylog[0]])

        for i in np.arange(1, np.size(y) - 1):
            d = np.sqrt(
                (ylog[i] - ylog_new[-1])**2 + (xlog[i + 1] - xlog_new[-1])**2
            )
            if d > d_lim:
                y_new = np.append(y_new, y[i])
                x_new = np.append(x_new, x[i])
                ylog_new = np.append(y_new, ylog[i])
                xlog_new = np.append(x_new, xlog[i])
                alphas_new = np.append(alphas_new, alphas[i])

        # extrema = np.array([np.min(sol_res), np.min(reg_res),
        #                     np.max(sol_res), np.max(reg_res)])
        # curvature_max = 0
        # for ii in range(np.size(x_new)-2):
        #     curvature = find_curvature(y_new[ii:ii+3],x_new[ii:ii+3],extrema)
        #     if isinstance(curvature,str):
        #         continue
        #     if curvature > curvature_max:
        #         curvature_max = curvature
        #         x_final = x_new[ii+1]
        #         y_final = y_new[ii+1]
        #         alpha_final = alphas_new[ii+1]
        # #print('Final Curvature: ' + str(curvature_max))

    # Alternatively find alpha_final using the L curve algorithm
    alpha_final = lassoLCurve(A, udot, 1e-14, 100, D='null', method='1',
                              tol=tol)
    las = lm.Lasso(alpha=alpha_final, fit_intercept=False, tol=1e-18,
                   max_iter=15000)
    las.fit(A, udot)
    y_final = np.linalg.norm(A @ las.coef_ - udot)
    x_final = np.sum(np.abs(las.coef_))

    if plot_results:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_new, y=y_new, text=alphas_new,
                                 name='scikit-learn'))
        fig.add_trace(go.Scatter(x=[x_final], y=[y_final], text=alpha_final))
        if c_actual is not None:
            fig.add_trace(go.Scatter(
                x=[np.sum(np.abs(signal_best))],
                y=[np.linalg.norm(A @ signal_best - udot)],
                mode='markers', text=alpha_best
            ))
        fig.update_xaxes(type='log', title_text='||c||')
        fig.update_yaxes(type='log', title_text='||Phi c - udot||')
        fig.update_layout(title_text=title, width=600, height=600,
                          showlegend=False)
        fig.show()

    if c_actual is None:
        return(alpha_final, x_final, y_final)
    else:
        return(alpha_final, x_final, y_final, alpha_best, signal_best)
