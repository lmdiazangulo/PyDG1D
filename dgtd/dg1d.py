import numpy as np
import scipy.special
import math

# n_faces, n_fp
n_faces = 2
n_fp = 1
   
def jacobiGL(alpha, beta, n_order):
    """
    Compute the order N Gauss Lobatto quadrature points, x, associated
    with the Jacobi polynomial.
    
    >>> jacobi_gauss_lobatto(0.0, 0.0, 1)
    array([-1.,  1.])
    
    >>> jacobi_gauss_lobatto(0,0,3)
    array([-1.       , -0.4472136,  0.4472136,  1.       ])

    >>> jacobi_gauss_lobatto(0,0,4)
    array([-1.        , -0.65465367,  0.        ,  0.65465367,  1.        ])
    
    """
    if n_order==0:
        return np.array([0.0])
    if n_order==1:
        return np.array([-1.0, 1.0])
    if n_order>1:
        x, w = scipy.special.roots_jacobi(n_order-1, alpha+1, beta+1);
        r_gl = np.concatenate(([-1.0], x, [1.0]))
        return r_gl
    
    raise ValueError('n_order must be positive.')

def jacobi_gauss(alpha, beta, n_order):
    """
    Compute the order n_order Gauss quadrature points, x, 
    and weights, w, associated with the Jacobi 
    polynomial, of type (alpha,beta) > -1 ( <> -0.5).
    >>> s1 = jacobi_gauss(2,1,0)
    >>> s2 = [-0.2,  2]
    >>> np.allclose(s1,s2)
    True
    >>> s1 = jacobi_gauss(2,1,1)
    >>> s2 = [([-0.54691816,  0.26120387]), ([0.76094757, 0.57238576])]
    >>> np.allclose(s1,s2)
    True
    >>> s1 = jacobi_gauss(2,1,2)
    >>> s2 = [([-0.70882014, -0.13230082,  0.50778763]), ([0.39524241,  0.72312171,  0.21496922])]
    >>> np.allclose(s1,s2)
    True
    """
    points = np.zeros(n_order)
    weight = np.zeros(n_order)
    if n_order==0:
        points = -(alpha-beta)/(alpha+beta+2)
        weight = 2
        return [points,weight]
    
    # Form symmetric matrix from recurrence.
    j_matrix = np.zeros([n_order+1,n_order+1])
    h1 = np.zeros(n_order+1)
    aux = np.zeros(n_order)
    
    for i in range(n_order):
        aux[i] = 1+i

    for i in range(n_order+1):
        h1[i] = 2*i+alpha+beta

    j_matrix = np.diag(-0.5*(alpha**2-beta**2)/(h1+2)/h1) \
        + np.diag(2/(h1[0:n_order]+2) \
        * np.sqrt(aux*(aux+alpha+beta)*(aux+alpha) \
        * (aux+beta)/(h1[0:n_order]+1)/(h1[0:n_order]+3)),1)

    eps = np.finfo(np.float).eps

    if (alpha+beta < 10*eps):
        j_matrix[0,0] = 0.0
    
    j_matrix = j_matrix + np.transpose(j_matrix)
    
    [e_val,e_vec] = np.linalg.eig(j_matrix)
    
    points = e_val 
    
    weight = e_vec[0,:]**2*2**(alpha+beta+1)/(alpha+beta+1)*scipy.special.gamma(alpha+1) \
        * scipy.special.gamma(beta+1)/scipy.special.gamma(alpha+beta+1)
    
    return [points,weight]


def jacobi_polynomial(r, alpha, beta, n_order):
    """
    Evaluate Jacobi Polynomial
    
    >>> r = jacobi_gauss_lobatto(0,0,1)
    >>> jacobi_polynomial(r, 0, 0, 1)
    array([-1.2247,  1.2247])
    
    >>> r = jacobi_gauss_lobatto(0,0,2)
    >>> jacobi_polynomial(r, 0, 0, 2)
    array([1.5811, -0.7906,  1.5811])
    
    >>> r = jacobi_gauss_lobatto(0,0,3)
    >>> jacobi_polynomial(r, 0, 0, 3)
    array([-1.87082869,  0.83666003, -0.83666003,  1.87082869])
    
    >>> r = jacobi_gauss_lobatto(0,0,4)
    >>> jacobi_polynomial(r, 0, 0, 4)
    array([ 2.12132034, -0.90913729,  0.79549513, -0.90913729,  2.12132034])
    
    """
    PL = np.zeros([n_order+1,len(r)]) 
    # Initial values P_0(x) and P_1(x)
    gamma0 = 2**(alpha+beta+1) \
             / (alpha+beta+1) \
             * scipy.special.gamma(alpha+1) \
             * scipy.special.gamma(beta+1) \
             / scipy.special.gamma(alpha+beta+1);
    PL[0] = 1.0 / math.sqrt(gamma0);
    if n_order == 0:
        return PL.transpose()
    
    gamma1 = (alpha+1.) * (beta+1.) / (alpha+beta+3.) * gamma0;
    PL[1] = ((alpha+beta+2.)*r/2. + (alpha-beta)/2.) / math.sqrt(gamma1);
    
    if n_order == 1:
        return np.transpose(PL[n_order])

    # Repeat value in recurrence.
    aold = 2. / (2.+alpha+beta) \
           * math.sqrt( (alpha+1.)*(beta+1.) / (alpha+beta+3.));

    # Forward recurrence using the symmetry of the recurrence.
    for i in range(n_order-1):
        h1 = 2.*(i+1.) + alpha + beta;
        anew = 2. / (h1+2.) \
               * math.sqrt((i+2.)*(i+2.+ alpha+beta)*(i+2.+alpha)*(i+2.+beta) \
                           / (h1+1.)/(h1+3.));
        bnew = - (alpha**2 - beta**2) / h1 / (h1+2.);
        PL[i+2] = 1. / anew * (-aold * PL[i] + (r-bnew) * PL[i+1]);
        aold = anew;

    return np.transpose(PL[n_order]);
    #return scipy.special.eval_jacobi(N,alpha,beta,r);

def vandermonde_1d(n_order, r):
    """
    Initialize Vandermonde matrix
    """
#    res = jacobi_polynomial(r, 0, 0, 0)
    res = np.zeros((len(r), n_order+1))
    for j in range(len(r)):
        res[:,j] = np.transpose(jacobi_polynomial(r, 0, 0, j))
        #res = np.hstack((res, jacobi_polynomial(r, 0, 0, j)))
    return res
    
def jacobi_polynomial_grad(r, alpha, beta, n_order):
    """Evaluate the derivative of the Jacobi pol. of type (alpha,beta) > -1
    at points r for order n_order
    >>> r = jacobi_gauss_lobatto(0,0,1)
    >>> jacobi_polynomial_grad(r,0,0,1)
    array([1.22474487, 1.22474487])
    >>> r = jacobi_gauss_lobatto(0,0,3)
    >>> jacobi_polynomial_grad(r,0,0,3)
    array([11.22497216,  0.        ,  0.        , 11.22497216])
    """
    
    der_jacobi_pol = np.zeros([len(r)])

    if n_order == 0:
        return der_jacobi_pol
       
    jacobi_pol = jacobi_polynomial(r,alpha+1,beta+1,n_order-1)

    for i in range(len(r)):
        der_jacobi_pol[i] = math.sqrt(n_order*(n_order+alpha+beta+1))*jacobi_pol[i]

    return der_jacobi_pol    


def vandermonde_grad(n_order,r):
    """
    Initialize the gradient of the modal basis (i) at (r)
    at order (n_order)
    >>> r = jacobi_gauss_lobatto(0,0,2)
    >>> vandermonde_grad(2,r)
    array([[ 0.        ,  1.22474487, -4.74341649],
           [ 0.        ,  1.22474487,  0.        ],
           [ 0.        ,  1.22474487,  4.74341649]])
    """
    grad_vander =  np.zeros([len(r),n_order+1])
    for i in range(n_order+1):
        grad_vander[:,i] = jacobi_polynomial_grad(r,0,0,i)
        
    return grad_vander


def differentiation_matrix(n_order,r,vander):
    """
    Initialize the (r) differentiation matrices
    of the interval evaluated at (r) at order n_order
    V is the 1d Vandermonde matrix
    >>> r = jacobi_gauss_lobatto(0,0,2)
    >>> v = vandermonde(2,r)
    >>> differentiation_matrix(2,r,v)
    array([[-1.5,  2. , -0.5],
           [-0.5,  0. ,  0.5],
           [ 0.5, -2. ,  1.5]])
    >>> r = jacobi_gauss_lobatto(0,0,3)
    >>> v = vandermonde(3,r)
    >>> A1 = differentiation_matrix(3,r,v)
    >>> A2 = ([[-3.00000000e+00,  4.04508497e+00, -1.54508497e+00,  5.00000000e-01], \
               [-8.09016994e-01, -4.05396129e-16,  1.11803399e+00, -3.09016994e-01], \
               [ 3.09016994e-01, -1.11803399e+00,  6.28036983e-16,  8.09016994e-01], \
               [-5.00000000e-01,  1.54508497e+00, -4.04508497e+00,  3.00000000e+00]])
    >>> np.allclose(A1,A2)
    True
    """
    v_r = vandermonde_grad(n_order,r)
    v_inv = np.linalg.inv(vander)
    diff_matrix = np.matmul(v_r,v_inv)
    return diff_matrix


def surface_integral_dg(n_order,vander):
    """ LIFT1D
    Compute surface integral term in DG formulation
    >>> r = jacobi_gauss_lobatto(0,0,2)
    >>> v = vandermonde(2,r)
    >>> surface_integral_dg(2,v)
    array([[ 4.5 ,  1.5 ],
           [-0.75, -0.75],
           [ 1.5 ,  4.5 ]])
    >>> r = jacobi_gauss_lobatto(0,0,3)
    >>> v = vandermonde(3,r)
    >>> surface_integral_dg(3,v)
    array([[ 8.        , -2.        ],
           [-0.89442719,  0.89442719],
           [ 0.89442719, -0.89442719],
           [-2.        ,  8.        ]])
    """
    n_p = n_order+1

    emat = np.zeros([n_p, n_faces*n_fp])
    emat[0, 0] = 1.0
    emat[n_p-1, 1] = 1.0

    v_trans = np.transpose(vander)
    v_i = np.matmul(v_trans,emat)
    lift = np.matmul(vander,v_i)

    return lift

def normals(k_elem):
    """
    Compute outward pointing normals at element faces
    >>> normals(4)
    array([[-1., -1., -1., -1.],
           [ 1.,  1.,  1.,  1.]])
    """
    # K is the number of elements, derived from the grid info
    # n_faces and n_fp are defined as global variables
    n_faces = 2
    n_fp = 1
    nx = np.zeros([n_fp * n_faces, k_elem])
    nx[0,:] = -1.0
    nx[1,:] = 1.0
    return nx 

def filter(n_order,n_c,s,vander):
    """
    Initialize 1D filter matrix of size n_order.
    Order of exponential filter is (even) s with cutoff at n_c;
    >>> r = jacobiGL(0,0,3)
    >>> v = vandermonde(3,r)
    >>> filter(3,1,1,v)
    array([[ 0.3333842 ,  0.9756328 , -0.14240119, -0.1666158 ],
           [ 0.19512656,  0.66667684,  0.16667684, -0.02848024],
           [-0.02848024,  0.16667684,  0.66667684,  0.19512656],
           [-0.1666158 , -0.14240119,  0.9756328 ,  0.3333842 ]])
    """
    s_even = 2*s
    f_diagonal = np.ones(n_order+1)
    alpha = -np.log(np.finfo(np.float).eps)

    for i in range(n_c,n_order+1):
        f_diagonal[i] = np.exp(-alpha*((i-n_c)/(n_order-n_c))**s_even)

    # F = V*diag(Fdiagonal)*Vinv
    f_i = np.matmul(vander,np.diag(f_diagonal))
    v_inv = np.linalg.inv(vander)
    filter = np.matmul(f_i,v_inv)
    return filter


def nodes_coordinates(n_order,EToV,vx):
    """
    Part of StartUp1D.m. De+fined to be able to define
    methods depedent grid properties
    >>> [Nv,vx,K,etov] = mesh_generator(0,10,4)
    >>> x = nodes_coordinates(4,etov,vx)
    >>> x_test = ([[0.00000000,    2.50000000,    5.00000000,    7.50000000], \
                   [0.43168291,    2.93168291,    5.43168291,    7.93168291], \
                   [1.25000000,    3.75000000,    6.25000000,    8.75000000], \
                   [2.06831709,    4.56831709,    7.06831709,    9.56831709], \
                   [2.50000000,    5.00000000,    7.50000000,   10.00000000]])
    >>> np.allclose(x,x_test)
    True
    """

    jgl = jacobiGL(0,0,n_order)
    
    va = EToV[:,0]
    vb = EToV[:,1]
    vx_va = np.zeros([1,len(va)])
    vx_vb = np.zeros([1,len(va)])
    for i in range(len(va)):
        vx_va[0,i] = vx[va[i]]
        vx_vb[0,i] = vx[vb[i]]

    nodes_coord = np.matmul(np.ones([n_order+1,1]),vx_va)+0.5*np.matmul((jgl.reshape(n_order+1,1)+1),(vx_vb-vx_va))
    return nodes_coord

def geometric_factors(nodes_coord,diff_matrix):
    """
    Compute the metric elements for the local mappings of the 1D elements 
    >>> [Nv,vx,K,etov] = mesh_generator(0,10,4)
    >>> x = nodes_coordinates(2,etov,vx)
    >>> r = jacobiGL(0,0,2)
    >>> V = vandermonde(2,r)
    >>> Dr = differentiation_matrix(2,r,V)
    >>> [rx,J] = geometric_factors(x,Dr)
    >>> rx_test = ([[0.80000,   0.80000,   0.80000,   0.80000], \
                    [0.80000,   0.80000,   0.80000,   0.80000], \
                    [0.80000,   0.80000,   0.80000,   0.80000]])
    >>> J_test =   ([[1.2500,   1.2500,   1.2500,   1.2500], \
                     [1.2500,   1.2500,   1.2500,   1.2500], \
                     [1.2500,   1.2500,   1.2500,   1.2500]])
    >>> np.allclose(rx,rx_test)
    True
    >>> np.allclose(J,J_test)
    True
    """
    xr = np.matmul(diff_matrix,nodes_coord)
    jacobian = xr
    rx = 1/jacobian

    return [rx,jacobian]

def connect(EToV):
    """
    Build global connectivity arrays for 1D grid based on standard 
    etov input array from grid generator
    >>> [Nv,vx,K,etov] = mesh_generator(0,10,4)
    >>> [etoe, etof] = connect(etov)
    >>> etoe_test =  ([[1,2], \
                       [1,3], \
                       [2,4], \
                       [3,4]])
    >>> etof_test =  ([[1,1], \
                       [2,1], \
                       [2,1], \
                       [2,2]])
    >>> np.allclose(etoe,etoe_test)
    True
    >>> np.allclose(etof,etof_test)
    True
    >>> [Nv,vx,K,etov] = mesh_generator(-1,22,7)
    >>> [etoe, etof] = connect(etov)
    >>> etoe_test = ([[1,2],\
                      [1,3],\
                      [2,4],\
                      [3,5],\
                      [4,6],\
                      [5,7],\
                      [6,7]])
    >>> etof_test = ([[1,1],\
                      [2,1],\
                      [2,1],\
                      [2,1],\
                      [2,1],\
                      [2,1],\
                      [2,2]])
    >>> np.allclose(etoe,etoe_test)
    True
    >>> np.allclose(etof,etof_test)
    True
    """
    n_faces = 2
    k_elem = np.shape(EToV)[0] 
    total_faces = n_faces*k_elem
    nv = k_elem+1
    vn = np.arange(0,2)
    sp_ftov = np.zeros([total_faces,nv])

    sk = 0

    for i in range(k_elem):
        for face in range(n_faces):
            sp_ftov[sk][EToV[i][vn[face]]-1] = 1
            sk += 1
    
    sp_ftof = np.matmul(sp_ftov,np.transpose(sp_ftov))-np.identity(total_faces)
    [faces_2,faces_1] = np.where(sp_ftof==1)
    #numpy floor returns floats

    element_1 = np.int64(np.floor(faces_1/n_faces))
    element_2 = np.int64(np.floor(faces_2/n_faces))

    face_1 = np.mod(faces_1,n_faces)
    face_2 = np.mod(faces_2,n_faces)

    ind = np.arange(len(element_1))

    for i in range(len(element_1)):
        ind[i] = np.ravel_multi_index((element_1[i],face_1[i]),dims=(k_elem,n_faces))

    etoe_1 = np.transpose(np.arange(1,k_elem+1).reshape(1,k_elem))
    etoe_2 = np.full([1,n_faces],1)
    etoe = np.matmul(etoe_1,etoe_2)

    etof_1 = np.full([k_elem,1],1)
    etof_2 = np.arange(1,n_faces+1).reshape(1,n_faces)
    etof = np.matmul(etof_1,etof_2)

    for i in range(len(ind)):
        etoe.ravel()[ind[i]] = element_2[i]+1
        etof.ravel()[ind[i]] = face_2[i]+1 

    return [etoe, etof]

def build_maps(n_order,nodes_coord,etoe,etof):
    """
    Connectivity and boundary tables for nodes given in the K # of elements,
    each with n_order+1 degrees of freedom.
    >>> [Nv,vx,K,etov] = mesh_generator(0,10,4)
    >>> x = nodes_coordinates(4,etov,vx)
    >>> [etoe, etof] = connect(etov)
    >>> [vmap_m,vmap_p,vmap_b,map_b] = build_maps(4,x,etoe,etof)
    >>> vmap_m_test = ([[1,5,6,10,11,15,16,20]])
    >>> np.allclose(vmap_m,vmap_m_test)
    True
    >>> vmap_p_test = ([[1,6,5,11,10,16,15,20]])
    >>> np.allclose(vmap_p,vmap_p_test)
    True
    >>> vmap_b_test = ([[1,20]])
    >>> np.allclose(vmap_b,vmap_b_test)
    True
    >>> map_b_test = ([[1,8]])
    >>> np.allclose(map_b,map_b_test)
    True
    """
    jgl = jacobiGL(0,0,n_order)
    k_elem = np.size(etoe,0)
    n_p = n_order+1
    n_faces = 2
    n_fp = 1
    #mask defined in globals
    fmask_1 = np.where(np.abs(jgl+1)<1e-10)[0][0]
    fmask_2 = np.where(np.abs(jgl-1)<1e-10)[0][0]
    fmask = [fmask_1,fmask_2]

    node_ids = np.reshape(np.arange(k_elem*n_p),[n_p,k_elem],'F')
    vmap_m = np.full([k_elem,n_fp,n_faces],0)
    vmap_p = np.full([k_elem,n_fp,n_faces],0)

    for k1 in range(k_elem):
        for f1 in range(n_faces):
            vmap_m[k1,:,f1] = node_ids[fmask[f1],k1]

    for k1 in range(k_elem):

        for f1 in range(n_faces):
            k2 = etoe[k1,f1]-1
            f2 = etof[k1,f1]-1

            vid_m = vmap_m[k1,:,f1][0]
            vid_p = vmap_m[k2,:,f2][0]

            x1 = nodes_coord.ravel('F')[vid_m]
            x2 = nodes_coord.ravel('F')[vid_p]

            distance = (x2-x1)**2  
            if (distance < 1e-10):
                vmap_p[k1,:,f1] = vid_p

    vmap_m+=1
    vmap_p+=1

    vmap_p = vmap_p.ravel()
    vmap_m = vmap_m.ravel()

    map_b = np.where(vmap_p==vmap_m)[0]
    vmap_b = vmap_m[map_b]

    map_b+=1
    vmap_b

    map_i = 1
    map_o = k_elem*n_faces
    vmap_i = 1
    vmap_0 = k_elem*n_p
    
    return [vmap_m-1, vmap_p-1, vmap_b-1, map_b-1]


def set_nodes_1d(N, vertices):
    """ 
    Sets N+1 nodes in equispaced positions using the vertices indicated
    by vx.
    """
    K = vertices.shape[0] # vertices columns 
    x = np.zeros([N+1, K])
    for k in range(K):
        x[:,k] = np.linspace(vertices[k,0], vertices[k,1], num=N+1)
             
    return x


def node_indices_1d(N):
    """
    Generates number of node Indices for order N.
    
    >>> _node_indices_1d(1)
    array([[1, 0],
           [0, 1]])
           
    >>> _node_indices_1d(2)
    array([[2, 0],
           [1, 1],
           [0, 2]])
    """
    Np = N+1
    nId = np.zeros([Np, 2])
    for i in range(Np):
        nId[i] = [N-i, i]     
    return nId.astype(int)

